"""
rl/shadow.py — Trade Optimizer shadow observer.

Runs alongside live_trader.py. Each bar it receives the same feature data the
ML bot sees, runs the RL model, and logs what the RL *would* have done — entry,
SL, and TP included. Never touches the broker.

Action space: MultiDiscrete([2, 3, 3])
  action[0]  0=SKIP, 1=TAKE
  action[1]  SL_MODE: 0=ML_SL, 1=TIGHT (0.65×), 2=WIDE (1.4×)
  action[2]  TP_MODE: 0=ML_TP, 1=CONSERVATIVE (1.5× RR), 2=EXTENDED (2.5× RR)

DB writes:
  Every bar → rl_shadow row (rl_suggested_sl/tp filled for TAKE decisions)
  On resolve → outcome columns updated (max_favourable, pnl_difference, rl_was_correct)

Model path (first found):
  rl/models/shadow_model/best_model.zip   (preferred)
  rl/models/final_shadow_model.zip        (fallback)

Usage:
  shadow = RLShadow.load(feature_columns=bundle["feature_columns"])
  rl_sig = shadow.observe(feat_df, ml_signal=sig)
  # rl_sig: {action, signal, sl, tp, rr, reason, rl_suggested_sl, rl_suggested_tp}

Retrain:
  When 50+ resolved outcomes accumulate a warning is logged.
  Run manually: python rl/retrain_shadow.py
"""

from __future__ import annotations

import logging
import os
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple

from utils.db_writer import write_rl_decision, update_rl_outcome

logger = logging.getLogger("rl.shadow")

_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(_THIS_DIR, "models")

_MODEL_CANDIDATES = [
    os.path.join(MODELS_DIR, "shadow_model", "best_model.zip"),
    os.path.join(MODELS_DIR, "final_shadow_model.zip"),
]

PIP_SIZE            = 0.10
OUTCOME_TIMEOUT     = 60        # bars before unresolved outcome is written as timeout
RETRAIN_THRESHOLD   = 50        # log warning after this many resolved outcomes

_SL_SCALE = {0: 1.00, 1: 0.65, 2: 1.40}
_TP_RR    = {1: 1.5,  2: 2.5}
_SL_NAMES = {0: "ML_SL", 1: "TIGHT_SL", 2: "WIDE_SL"}
_TP_NAMES = {0: "ML_TP", 1: "CONS_TP",  2: "EXT_TP"}

_SESSION_SL_BUFFER = {
    "Asian": 0.80, "Off": 0.80,
    "London": 0.50, "NY": 0.50, "Overlap": 0.45,
}

_TF_BARS_PER_HOUR = {
    "1min": 60, "5min": 12, "15min": 4, "30min": 2,
    "1H": 1, "4H": 0.25, "1D": 0.042,
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(val) -> datetime:
    if isinstance(val, datetime):
        return val if val.tzinfo else val.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(val))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return _utcnow()


def _session(row: pd.Series) -> str:
    london = float(row.get("london_session", row.get("in_london", 0)) or 0) > 0.5
    ny     = float(row.get("ny_session",     row.get("in_newyork", row.get("in_ny", 0))) or 0) > 0.5
    asian  = float(row.get("asian_session",  row.get("in_asian", 0)) or 0) > 0.5
    if london and ny:
        return "Overlap"
    if london:
        return "London"
    if ny:
        return "NY"
    if asian:
        return "Asian"
    if float(row.get("in_session", 0) or 0) > 0.5:
        return "Session"
    return "Off"


def _htf_bias_int(htf4h: float) -> int:
    if htf4h > 0.3:
        return 1
    if htf4h < -0.3:
        return -1
    return 0


def _market_structure(htf4h: float) -> str:
    if htf4h > 0.3:
        return "bullish"
    if htf4h < -0.3:
        return "bearish"
    return "neutral"


def _signed_pips(price_diff: float, direction: int) -> float:
    return round(price_diff * direction / PIP_SIZE, 1)


class RLShadow:
    """Passive trade-optimizer observer. Predicts, tracks virtually, logs. Never executes."""

    def __init__(
        self,
        model,
        feature_columns: list,
        timeframe: str         = "15min",
        symbol: str            = "XAUUSDm",
        initial_balance: float = 5_000.0,
        lot_size: float        = 0.01,
        contract_size: float   = 10.0,
        min_rr: float          = 1.5,
        secondary_tfs: list    = None,
    ):
        self.model           = model
        self.feature_columns = feature_columns
        self.timeframe       = timeframe
        self.symbol          = symbol
        self.initial_balance = initial_balance
        self.lot_size        = lot_size
        self.contract_size   = contract_size
        self.min_rr          = min_rr
        self.secondary_tfs   = sorted(secondary_tfs or ["5min", "1H", "4H"])

        _bph = _TF_BARS_PER_HOUR.get(timeframe, 4)
        self._bars_1h  = max(1, round(_bph))
        self._bars_4h  = max(1, round(_bph * 4))
        self._bars_24h = max(1, round(_bph * 24))

        # Virtual position
        self._position      = 0
        self._entry_price   = 0.0
        self._sl            = 0.0
        self._tp            = 0.0
        self._entry_rr      = 0.0
        self._bars_in_trade = 0
        self._stall_count   = 0
        self._hi_in_trade   = 0.0
        self._lo_in_trade   = float("inf")
        self._max_favorable = 0.0
        self._max_adverse   = 0.0
        self._trade_quality = 0.0
        self._balance       = float(initial_balance)
        self._trade_history: list = []

        # ML virtual position for differential tracking
        self._ml_vpos   = 0
        self._ml_ventry = 0.0
        self._ml_vsl    = 0.0
        self._ml_vtp    = 0.0

        # Pending outcome queue
        self._pending_outcomes: List[Dict[str, Any]] = []
        self._resolved_count = 0

        # LSTM state
        self._lstm_state = None

    # ── Load ──────────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, feature_columns: list, timeframe: str = "15min", **kwargs) -> "RLShadow":
        try:
            from stable_baselines3 import PPO
        except ImportError:
            raise ImportError("pip install stable-baselines3")

        from config.pipeline_config import RL_FEATURE_COLUMNS as _RL_FC
        feature_columns = list(_RL_FC)

        for path in _MODEL_CANDIDATES:
            if os.path.exists(path):
                logger.info("RLShadow loading: %s", path)
                model = PPO.load(path)
                logger.info("Model loaded. Obs dim: %d  Action: %s",
                            model.observation_space.shape[0],
                            model.action_space)
                return cls(model=model, feature_columns=feature_columns,
                           timeframe=timeframe, **kwargs)

        raise FileNotFoundError(
            "No shadow model found. Train one:\n"
            "  python rl/train_shadow.py --steps 200000"
        )

    # ── Observation builder ───────────────────────────────────────────────────

    def _build_obs(
        self,
        primary_row: pd.Series,
        secondary_rows: Optional[Dict[str, pd.Series]] = None,
        ml_signal_dict: Optional[dict] = None,
    ) -> np.ndarray:
        fc    = self.feature_columns
        parts = [primary_row.reindex(fc, fill_value=0).fillna(0).values.astype(np.float32)]

        for tf in self.secondary_tfs:
            if secondary_rows and tf in secondary_rows:
                sec = secondary_rows[tf]
                # Accept either a DataFrame (take last row) or a Series
                if isinstance(sec, pd.DataFrame):
                    sec = sec.iloc[-1]
                parts.append(sec.reindex(fc, fill_value=0).fillna(0).values.astype(np.float32))
            else:
                parts.append(np.zeros(len(fc), dtype=np.float32))

        close = float(primary_row.get("close", 0) or 0)
        ml    = ml_signal_dict or {}

        ml_vpnl = self._ml_virtual_pnl(close) / max(self.initial_balance, 1)
        ml_ctx = np.array([
            np.clip(float(ml.get("signal", 0) or 0),                          -1, 1),
            np.clip(float(ml.get("prob",   0) or 0),                           0, 1),
            {"A": 1.0, "B": 0.67, "C": 0.33}.get(
                str(ml.get("grade", "") or "").strip().upper(), 0.0),
            np.clip(ml_vpnl,                                                   -1, 1),
            np.clip(float(primary_row.get("htf_4h_bias",       0) or 0),      -1, 1),
            np.clip(float(primary_row.get("macro_atr_ratio", 1.0) or 1.0) - 1.0, -1, 2),
            np.clip(float(primary_row.get("active_zone_quality", 0) or 0) / 5.0,  0, 1),
            np.clip(float(primary_row.get("volume_ratio", 1.0) or 1.0) - 1.0,    -1, 2),
        ], dtype=np.float32)
        parts.append(ml_ctx)

        upnl  = self._unrealized_pnl(close) / max(self.initial_balance, 1)
        risk  = abs(close - self._sl) if (self._position != 0 and self._sl != 0) else 1.0
        mfe_r = self._max_favorable / max(risk, 1e-6) if self._position != 0 else 0.0
        mom5  = float(primary_row.get("momentum_5", 0) or 0)
        mom_n = float(np.clip(mom5 * self._position, -1, 1)) if self._position != 0 else 0.0
        pos_ctx = np.array([
            float(self._position),
            np.clip(upnl,    -1, 1),
            min(self._bars_in_trade / 50.0, 1.0),
            mom_n,
            float(self._stall_count < 8 and self._position != 0),
            np.clip(mfe_r, 0, 3),
            np.clip(self._trade_quality / 5.0, 0, 1),
        ], dtype=np.float32)
        parts.append(pos_ctx)

        return np.concatenate(parts)

    # ── PnL helpers ───────────────────────────────────────────────────────────

    def _unrealized_pnl(self, price: float) -> float:
        if self._position == 0:
            return 0.0
        return (price - self._entry_price) * self._position * self.lot_size * self.contract_size

    def _ml_virtual_pnl(self, price: float) -> float:
        if self._ml_vpos == 0:
            return 0.0
        return (price - self._ml_ventry) * self._ml_vpos * self.lot_size * self.contract_size

    # ── SL/TP computation ─────────────────────────────────────────────────────

    def _compute_levels(
        self, row: pd.Series, direction: int, sl_mode: int, tp_mode: int
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Return (sl, tp, rr, risk) or (None, None, None, None)."""
        close = float(row.get("close", 0) or 0)
        atr   = float(row.get("atr_14", 0) or 0)
        if atr <= 0:
            return None, None, None, None

        sess     = _session(row)
        base_buf = _SESSION_SL_BUFFER.get(sess, 0.50)
        sl_buf   = base_buf * _SL_SCALE[sl_mode]

        def _f(k: str) -> float:
            v = row.get(k, np.nan)
            try:
                v = float(v)
                return np.nan if np.isnan(v) else v
            except (TypeError, ValueError):
                return np.nan

        if direction == 1:
            d_bot = _f("demand_zone_bottom")
            if np.isnan(d_bot):
                return None, None, None, None
            sl   = d_bot - sl_buf * atr
            risk = close - sl
            if risk <= 0:
                return None, None, None, None
            if tp_mode == 0:
                s_bot = _f("supply_zone_bottom")
                tp = s_bot if (not np.isnan(s_bot) and s_bot > close) \
                    else close + max(self.min_rr * risk, 3.0 * atr)
            else:
                tp = close + _TP_RR[tp_mode] * risk
        else:
            s_top = _f("supply_zone_top")
            if np.isnan(s_top):
                return None, None, None, None
            sl   = s_top + sl_buf * atr
            risk = sl - close
            if risk <= 0:
                return None, None, None, None
            if tp_mode == 0:
                d_top = _f("demand_zone_top")
                tp = d_top if (not np.isnan(d_top) and d_top < close) \
                    else close - max(self.min_rr * risk, 3.0 * atr)
            else:
                tp = close - _TP_RR[tp_mode] * risk

        rr = abs(tp - close) / risk
        if rr < self.min_rr:
            return None, None, None, None
        return round(sl, 5), round(tp, 5), round(rr, 3), round(risk, 5)

    # ── Virtual position management ───────────────────────────────────────────

    def _check_sl_tp_hit(self, pos: int, sl: float, tp: float, row: pd.Series) -> Tuple[bool, bool]:
        if pos == 0:
            return False, False
        hi = float(row.get("high", 0) or 0)
        lo = float(row.get("low",  0) or 0)
        return (lo <= sl, hi >= tp) if pos == 1 else (hi >= sl, lo <= tp)

    def _update_virtual_position(self, row: pd.Series) -> Optional[dict]:
        if self._position == 0:
            return None
        hi = float(row.get("high", 0) or 0)
        lo = float(row.get("low",  0) or 0)
        if self._position == 1:
            if hi > self._hi_in_trade:
                self._hi_in_trade = hi
                self._stall_count = 0
            else:
                self._stall_count += 1
            fav = hi - self._entry_price
            adv = self._entry_price - lo
        else:
            if lo < self._lo_in_trade:
                self._lo_in_trade = lo
                self._stall_count = 0
            else:
                self._stall_count += 1
            fav = self._entry_price - lo
            adv = hi - self._entry_price
        self._max_favorable = max(self._max_favorable, max(fav, 0.0))
        self._max_adverse   = max(self._max_adverse,   max(adv, 0.0))

        sl_hit, tp_hit = self._check_sl_tp_hit(self._position, self._sl, self._tp, row)
        if not (sl_hit or tp_hit):
            return None

        exit_price = self._sl if sl_hit else self._tp
        pnl = (exit_price - self._entry_price) * self._position \
              * self.lot_size * self.contract_size
        self._balance += pnl
        close_event = {"pnl": pnl, "exit": "sl" if sl_hit else "tp", "exit_price": exit_price}
        self._trade_history.append(close_event)
        self._position      = 0
        self._bars_in_trade = 0
        self._entry_rr      = 0.0
        self._stall_count   = 0
        self._max_favorable = 0.0
        self._max_adverse   = 0.0
        return close_event

    def _open_virtual_position(
        self, row: pd.Series, direction: int, sl: float, tp: float, rr: float
    ) -> None:
        close = float(row.get("close", 0) or 0)
        self._position      = direction
        self._entry_price   = close
        self._sl            = sl
        self._tp            = tp
        self._entry_rr      = rr
        self._bars_in_trade = 0
        self._stall_count   = 0
        self._max_favorable = 0.0
        self._max_adverse   = 0.0
        self._hi_in_trade   = float(row.get("high", close) or close)
        self._lo_in_trade   = float(row.get("low",  close) or close)
        self._trade_quality = float(row.get("active_zone_quality", 0) or 0)

    # ── Pending outcome tracking ──────────────────────────────────────────────

    def _enqueue_outcome(
        self, db_id: Optional[int], direction: int, entry: float,
        rl_sl: Optional[float], rl_tp: Optional[float], rr: Optional[float],
        ml_sl: Optional[float], ml_tp: Optional[float],
        agreement: str, decision_ts: str,
    ) -> None:
        if rl_sl is None and ml_sl is None:
            return
        self._pending_outcomes.append({
            "db_id":        db_id,
            "decision_ts":  decision_ts,
            "agreement":    agreement,
            "direction":    direction,
            "entry_price":  entry,
            "rl_sl":        rl_sl,
            "rl_tp":        rl_tp,
            "rr":           rr or 0.0,
            "ml_sl":        ml_sl,
            "ml_tp":        ml_tp,
            "age":          0,
            "price_1h":     None,
            "price_4h":     None,
            "price_24h":    None,
            "mfe_price":    0.0,
            "mae_price":    0.0,
            "ml_resolved":  False,
            "ml_exit":      None,
        })

    def _resolve_pending_outcomes(self, row: pd.Series) -> None:
        hi    = float(row.get("high",  0) or 0)
        lo    = float(row.get("low",   0) or 0)
        close = float(row.get("close", 0) or 0)
        still = []

        for p in self._pending_outcomes:
            p["age"] += 1
            if p["price_1h"]  is None and p["age"] >= self._bars_1h:
                p["price_1h"]  = round(close, 5)
            if p["price_4h"]  is None and p["age"] >= self._bars_4h:
                p["price_4h"]  = round(close, 5)
            if p["price_24h"] is None and p["age"] >= self._bars_24h:
                p["price_24h"] = round(close, 5)

            pos = p["direction"]
            # MFE / MAE
            if pos == 1:
                p["mfe_price"] = max(p["mfe_price"], max(hi - p["entry_price"], 0.0))
                p["mae_price"] = max(p["mae_price"], max(p["entry_price"] - lo, 0.0))
            else:
                p["mfe_price"] = max(p["mfe_price"], max(p["entry_price"] - lo, 0.0))
                p["mae_price"] = max(p["mae_price"], max(hi - p["entry_price"], 0.0))

            # RL hit
            rl_sl = p["rl_sl"]
            rl_tp = p["rl_tp"]
            rl_sl_hit = rl_tp_hit = False
            if rl_sl is not None and rl_tp is not None:
                rl_sl_hit = (lo <= rl_sl) if pos == 1 else (hi >= rl_sl)
                rl_tp_hit = (hi >= rl_tp) if pos == 1 else (lo <= rl_tp)

            # ML hit (for comparison)
            if not p["ml_resolved"] and p["ml_sl"] and p["ml_tp"]:
                ml_sl_hit = (lo <= p["ml_sl"]) if pos == 1 else (hi >= p["ml_sl"])
                ml_tp_hit = (hi >= p["ml_tp"]) if pos == 1 else (lo <= p["ml_tp"])
                if ml_tp_hit or ml_sl_hit:
                    p["ml_exit"]    = p["ml_tp"] if ml_tp_hit else p["ml_sl"]
                    p["ml_resolved"] = True
                elif p["age"] >= OUTCOME_TIMEOUT:
                    p["ml_exit"]    = close
                    p["ml_resolved"] = True

            resolved = rl_tp_hit or rl_sl_hit or p["age"] >= OUTCOME_TIMEOUT
            if resolved:
                exit_price = rl_tp if rl_tp_hit else (rl_sl if rl_sl_hit else close)
                pnl = (exit_price - p["entry_price"]) * pos \
                      * self.lot_size * self.contract_size if rl_sl else 0.0
                market_did = "won" if rl_tp_hit else ("lost" if rl_sl_hit else "stalled")

                ml_exit = p["ml_exit"] if p["ml_exit"] is not None else close
                ml_pnl = (ml_exit - p["entry_price"]) * pos \
                         * self.lot_size * self.contract_size if p["ml_sl"] else None

                agreement = p["agreement"]
                if agreement == "SKIP":
                    rl_correct: Optional[bool] = True if rl_sl_hit else (False if rl_tp_hit else None)
                else:
                    rl_correct = True if rl_tp_hit else (False if rl_sl_hit else None)

                pnl_diff = round(pnl - ml_pnl, 2) if ml_pnl is not None else None

                if p.get("db_id") is not None:
                    mfe_usd = round(p["mfe_price"] * self.lot_size * self.contract_size, 2)
                    mae_usd = round(p["mae_price"] * self.lot_size * self.contract_size, 2)
                    update_rl_outcome(
                        row_id              = p["db_id"],
                        price_1h_later      = p["price_1h"],
                        price_4h_later      = p["price_4h"],
                        price_24h_later     = p["price_24h"],
                        max_favourable      = mfe_usd,
                        max_adverse         = mae_usd,
                        ml_actual_pnl       = round(ml_pnl, 2) if ml_pnl is not None else None,
                        rl_hypothetical_pnl = round(pnl, 2),
                        pnl_difference      = pnl_diff,
                        rl_was_correct      = rl_correct,
                    )
                    self._resolved_count += 1
                    if self._resolved_count >= RETRAIN_THRESHOLD:
                        logger.warning(
                            "%d resolved outcomes accumulated. "
                            "Run: python rl/retrain_shadow.py",
                            self._resolved_count,
                        )
                        self._resolved_count = 0   # reset so warning fires again later
            else:
                still.append(p)

        self._pending_outcomes = still

    # ── Reason string ─────────────────────────────────────────────────────────

    @staticmethod
    def _reason(
        take: int, sl_mode: int, tp_mode: int,
        ml_dir: int, row: pd.Series, ml_prob: float, zone_q: float,
    ) -> str:
        dir_name = {1: "BUY", -1: "SELL", 0: "FLAT"}.get(ml_dir, "FLAT")
        if ml_dir == 0:
            return "HOLD: No ML signal this bar"
        if take == 0:
            htf4h   = float(row.get("htf_4h_bias", 0) or 0)
            in_sess = float(row.get("in_session",   0) or 0) == 1.0
            atr_r   = float(row.get("macro_atr_ratio", 1.0) or 1.0)
            counter = (ml_dir == 1 and htf4h < -0.5) or (ml_dir == -1 and htf4h > 0.5)
            if counter:
                return f"SKIP {dir_name} | counter-HTF (4H {htf4h:+.2f})"
            if not in_sess:
                return f"SKIP {dir_name} | off-session"
            if atr_r < 0.75:
                return f"SKIP {dir_name} | low momentum (ATR×{atr_r:.2f})"
            if zone_q < 1.5:
                return f"SKIP {dir_name} | weak zone ({zone_q:.1f}/5)"
            return f"SKIP {dir_name} | RL confidence below threshold ({ml_prob:.0%})"
        return (
            f"TAKE {dir_name} | {_SL_NAMES[sl_mode]} | {_TP_NAMES[tp_mode]} "
            f"| zone={zone_q:.1f} prob={ml_prob:.0%}"
        )

    # ── Main observe() ────────────────────────────────────────────────────────

    def observe(
        self,
        feat_df: pd.DataFrame,
        ml_signal: Optional[dict] = None,
        secondary_rows: Optional[Dict[str, pd.Series]] = None,
    ) -> dict:
        """
        Called once per bar by live_trader.py.

        feat_df:        DataFrame with the current bar's features (15min primary TF).
        ml_signal:      dict from ML evaluate_bar(): {signal, prob, grade, sl, tp, rr, reason}
        secondary_rows: {tf: pd.Series} for 5min/1H/4H feature rows (optional)

        Returns dict:
          action, signal, sl, tp, rr, reason,
          rl_suggested_sl, rl_suggested_tp, agreement, rl_confidence
        """
        ml = ml_signal or {}
        row = feat_df.iloc[-1] if len(feat_df) > 1 else feat_df.iloc[0]

        # Resolve any pending outcome tracking
        self._resolve_pending_outcomes(row)

        # Update virtual ML position for obs builder
        if self._ml_vpos != 0:
            ml_sl_h = ml_tp_h = False
            hi = float(row.get("high", 0) or 0)
            lo = float(row.get("low",  0) or 0)
            if self._ml_vpos == 1:
                ml_sl_h = lo <= self._ml_vsl
                ml_tp_h = hi >= self._ml_vtp
            else:
                ml_sl_h = hi >= self._ml_vsl
                ml_tp_h = lo <= self._ml_vtp
            if ml_sl_h or ml_tp_h:
                self._ml_vpos = 0

        # Build observation
        obs = self._build_obs(row, secondary_rows, ml)
        obs_batch = obs[np.newaxis, :]

        # Run model — RecurrentPPO needs LSTM state
        try:
            from sb3_contrib import RecurrentPPO
            action_arr, self._lstm_state = self.model.predict(
                obs_batch,
                state=self._lstm_state,
                deterministic=True,
            )
        except Exception:
            action_arr, _ = self.model.predict(obs_batch, deterministic=True)

        action_arr = np.array(action_arr).flatten()
        take    = int(action_arr[0]) if len(action_arr) > 0 else 0
        sl_mode = int(action_arr[1]) if len(action_arr) > 1 else 0
        tp_mode = int(action_arr[2]) if len(action_arr) > 2 else 0

        # RL confidence proxy (max action prob if available)
        rl_confidence: Optional[float] = None
        try:
            import torch
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs_batch)
                dist  = self.model.policy.get_distribution(obs_t)
                probs = [d.probs.squeeze() for d in dist.distributions]
                rl_confidence = round(float(probs[0][take].item()), 4)
        except Exception:
            pass

        ml_dir  = int(ml.get("signal", 0) or 0)
        ml_prob = float(ml.get("prob",   0) or 0)
        zone_q  = float(row.get("active_zone_quality", 0) or 0)
        close   = float(row.get("close", 0) or 0)

        # Compute RL's suggested levels
        rl_sl: Optional[float] = None
        rl_tp: Optional[float] = None
        rl_rr: Optional[float] = None
        ml_sl_level: Optional[float] = None
        ml_tp_level: Optional[float] = None

        if ml_dir != 0:
            rl_sl, rl_tp, rl_rr, _ = self._compute_levels(row, ml_dir, sl_mode, tp_mode)
            ml_sl_level, ml_tp_level, _, _ = self._compute_levels(row, ml_dir, 0, 0)

        agreement = "HOLD" if ml_dir == 0 else ("SKIP" if take == 0 else "TAKE")
        reason    = self._reason(take, sl_mode, tp_mode, ml_dir, row, ml_prob, zone_q)

        # Update virtual position
        close_event = self._update_virtual_position(row)

        if take == 1 and ml_dir != 0 and rl_sl is not None:
            # Implicit rotate: if holding opposite, close it
            if self._position != 0 and self._position != ml_dir and self._stall_count >= 8:
                exit_p = close
                pnl    = (exit_p - self._entry_price) * self._position \
                         * self.lot_size * self.contract_size
                self._balance  += pnl
                self._position  = 0
                self._bars_in_trade = 0

            if self._position == 0:
                self._open_virtual_position(row, ml_dir, rl_sl, rl_tp, rl_rr or 0.0)
                # Track ML baseline
                if self._ml_vpos == 0 and ml_sl_level is not None:
                    self._ml_vpos   = ml_dir
                    self._ml_ventry = close
                    self._ml_vsl    = ml_sl_level
                    self._ml_vtp    = ml_tp_level

        if self._position != 0:
            self._bars_in_trade += 1

        # Write to DB
        db_id = self._write_decision(
            row        = row,
            ml_sig     = ml,
            take       = take,
            sl_mode    = sl_mode,
            tp_mode    = tp_mode,
            rl_sl      = rl_sl if take == 1 else None,
            rl_tp      = rl_tp if take == 1 else None,
            agreement  = agreement,
            reason     = reason,
            rl_conf    = rl_confidence,
        )

        # Enqueue outcome tracking for bars where a signal existed
        if ml_dir != 0:
            ts_str = str(row.get("timestamp", _utcnow()))
            self._enqueue_outcome(
                db_id      = db_id,
                direction  = ml_dir,
                entry      = close,
                rl_sl      = rl_sl if take == 1 else None,
                rl_tp      = rl_tp if take == 1 else None,
                rr         = rl_rr,
                ml_sl      = ml_sl_level,
                ml_tp      = ml_tp_level,
                agreement  = agreement,
                decision_ts = ts_str,
            )

        return {
            "action":          take,
            "signal":          ml_dir if take == 1 else 0,
            "sl":              rl_sl,
            "tp":              rl_tp,
            "rr":              rl_rr,
            "reason":          reason,
            "rl_suggested_sl": rl_sl if take == 1 else None,
            "rl_suggested_tp": rl_tp if take == 1 else None,
            "agreement":       agreement,
            "rl_confidence":   rl_confidence,
        }

    # ── DB write ──────────────────────────────────────────────────────────────

    def _write_decision(
        self,
        row: pd.Series, ml_sig: dict,
        take: int, sl_mode: int, tp_mode: int,
        rl_sl: Optional[float], rl_tp: Optional[float],
        agreement: str, reason: str, rl_conf: Optional[float],
    ) -> Optional[int]:
        ml    = ml_sig or {}
        ts    = _parse_ts(row.get("timestamp", ""))
        close = float(row.get("close", 0) or 0)
        ml_dir = int(ml.get("signal", 0) or 0)
        htf4h  = float(row.get("htf_4h_bias", 0) or 0)
        body   = abs(float(row.get("body_size",    0) or 0))
        candle = max(float(row.get("candle_size", 1e-9) or 1e-9), 1e-9)

        # RL decision: encode take/skip + sl_mode + tp_mode as a single int (0=skip, 1-9=take combos)
        # For DB rl_decision column: keep 0=skip, 1=take (simplified)
        rl_decision_db = 1 if take == 1 else 0

        return write_rl_decision(
            timestamp               = ts,
            symbol                  = self.symbol,
            ml_signal               = ml_dir,
            ml_confidence           = round(float(ml.get("prob", 0) or 0), 4),
            ml_entry_price          = close if ml_dir != 0 else None,
            ml_sl_price             = ml.get("sl"),
            ml_tp_price             = ml.get("tp"),
            ml_rr_ratio             = ml.get("rr"),
            ml_triggered_rule       = str(ml.get("rule", ml.get("reason", "")) or ""),
            ml_triggered_tf         = str(ml.get("trigger_tf", ml.get("timeframe", self.timeframe)) or ""),
            htf_bias                = _htf_bias_int(htf4h),
            market_structure        = _market_structure(htf4h),
            session                 = _session(row),
            momentum_score          = round(float(row.get("momentum_5",       0) or 0), 4),
            atr_ratio               = round(float(row.get("macro_atr_ratio", 1.0) or 1.0), 3),
            volume_ratio            = round(float(row.get("volume_ratio",    1.0) or 1.0), 3),
            candle_body_ratio       = round(body / candle, 4),
            in_trade                = self._position != 0,
            trade_duration          = self._bars_in_trade,
            unrealised_pnl          = round(self._unrealized_pnl(close), 2),
            is_stalling             = self._stall_count >= 8,
            better_setup_available  = False,
            better_setup_direction  = None,
            better_setup_quality    = round(float(row.get("active_zone_quality", 0) or 0), 4),
            rl_decision             = rl_decision_db,
            rl_confidence           = rl_conf,
            rl_reason               = reason,
            rl_suggested_entry      = close if take == 1 and ml_dir != 0 else None,
            rl_suggested_sl         = rl_sl,
            rl_suggested_tp         = rl_tp,
            rl_recommended_rotation = False,
        )

    # ── Session summary ───────────────────────────────────────────────────────

    def session_summary(self) -> dict:
        wins   = [t for t in self._trade_history if t["pnl"] > 0]
        losses = [t for t in self._trade_history if t["pnl"] <= 0]
        n      = len(self._trade_history)
        return {
            "virtual_trades":   n,
            "virtual_wins":     len(wins),
            "virtual_losses":   len(losses),
            "virtual_wr":       round(len(wins) / max(n, 1), 3),
            "virtual_pnl":      round(sum(t["pnl"] for t in self._trade_history), 2),
            "pending_outcomes": len(self._pending_outcomes),
            "resolved_count":   self._resolved_count,
        }
