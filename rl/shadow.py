"""
rl/shadow.py — Parallel RL observer that runs alongside the live ML bot.

Receives the same feature data that live_trader.py builds each candle, runs
the trained shadow PPO model, and logs what the RL agent *would* have done —
but NEVER touches the broker or places any order.

Shadow log (one row per bar, logs/rl_shadow_YYYY-MM-DD.csv):
  • ML signal vs RL decision (agree / skip / override / rotate)
  • Reason for the decision (human-readable rule explanation)
  • RL confidence proxy (action probability)
  • Virtual position state
  • 4H HTF bias, zone quality, momentum context

Outcome log (one row per resolved virtual trade, logs/rl_outcomes_YYYY-MM-DD.csv):
  • Tracks what market actually did after each RL/ML signal decision
  • Filled when virtual TP or SL is hit (or after OUTCOME_TIMEOUT bars)

Obs dims:
  Shadow model  (is_shadow_model=True):  4 × 81 + 8 + 7 = 339
  Generic MTF   (is_shadow_model=False): 4 × 81 + 3     = 327
  (81 = 66 REQUIRED_FEATURE_COLUMNS + 15 RL_MACRO_FEATURE_COLUMNS)

Model priority (first found wins):
  1. rl/models/shadow_model/best_model.zip   (shadow-specific, preferred)
  2. rl/models/final_shadow_model.zip        (shadow final)
  3. rl/models/best_model_mtf/best_model.zip (generic multi-TF)
  4. rl/models/final_model_mtf.zip           (generic multi-TF)

Usage in live_trader.py (already wired — no changes needed there):
    from rl.shadow import RLShadow
    shadow = RLShadow.load(feature_columns=bundle["feature_columns"])
    rl_sig = shadow.observe(feat_df, ml_signal=sig)
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

# ── Paths ──────────────────────────────────────────────────────────────────────
_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(_THIS_DIR, "models")

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


def _htf_bias_int(htf4h: float) -> Optional[int]:
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


_MODEL_CANDIDATES = [
    # (path, is_multi_tf, is_shadow_model)
    (os.path.join(MODELS_DIR, "shadow_model", "best_model.zip"), True,  True),
    (os.path.join(MODELS_DIR, "final_shadow_model.zip"),         True,  True),
    (os.path.join(MODELS_DIR, "best_model_mtf", "best_model.zip"), True, False),
    (os.path.join(MODELS_DIR, "final_model_mtf.zip"),            True,  False),
    (os.path.join(MODELS_DIR, "best_model", "best_model.zip"),   False, False),
    (os.path.join(MODELS_DIR, "final_model.zip"),                False, False),
]


_ACTION_NAMES = {0: "agree", 1: "skip", 2: "rotate"}

# Timeout (bars) before we write an unresolved outcome as "unknown"
OUTCOME_TIMEOUT = 60

# 1 pip in price units for XAU/USD (0.10 = standard broker convention)
PIP_SIZE = 0.10


def _to_pips(price_diff: float) -> float:
    """Convert a raw price difference to pips (rounds to 1 decimal)."""
    return round(abs(price_diff) / PIP_SIZE, 1)


def _signed_pips(price_diff: float, direction: int) -> float:
    """Signed pips: positive = in direction's favour."""
    return round(price_diff * direction / PIP_SIZE, 1)


def _session_name(row: "pd.Series") -> str:
    """Derive session label from feature row (prefers explicit columns, falls back to timestamp)."""
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
    # Fallback: derive from in_session flag
    if float(row.get("in_session", 0) or 0) > 0.5:
        return "Session"
    return "Off"

# Bars-per-hour by timeframe — used to compute 1H/4H/24H price-snapshot offsets
_TF_BARS_PER_HOUR = {
    "1min": 60, "5min": 12, "15min": 4, "30min": 2,
    "1H": 1, "4H": 0.25, "1D": 0.042,
}

# HTF thresholds for generating reasons
COUNTER_HTF_THRESHOLD  = 0.5
ALIGNED_HTF_THRESHOLD  = 0.3
LOW_MOMENTUM_ATR_RATIO = 0.75
LOW_VOLUME_RATIO       = 0.70
WEAK_ZONE_QUALITY      = 1.5
STRONG_ZONE_QUALITY    = 3.5
STALL_BARS             = 8


class RLShadow:
    """Passive observer — predicts, tracks virtually, logs. Never executes orders."""

    def __init__(
        self,
        model,
        feature_columns: list,
        timeframe: str          = "15min",
        symbol: str             = "XAUUSDm",
        initial_balance: float  = 5_000.0,
        lot_size: float         = 0.01,
        contract_size: float    = 10.0,
        sl_atr_buffer: float    = 0.5,
        min_rr: float           = 1.5,
        is_multi_tf: bool       = False,
        is_shadow_model: bool   = False,
        secondary_tfs: list     = None,
        pip_size: float         = PIP_SIZE,
    ):
        self.model           = model
        self.feature_columns = feature_columns
        self.timeframe       = timeframe
        self.symbol          = symbol
        self.initial_balance = initial_balance
        self.lot_size        = lot_size
        self.contract_size   = contract_size
        self.sl_atr_buffer   = sl_atr_buffer
        self.min_rr          = min_rr
        self.is_multi_tf     = is_multi_tf
        self.is_shadow_model = is_shadow_model
        self.secondary_tfs   = sorted(secondary_tfs or ["5min", "1H", "4H"])
        self.pip_size        = pip_size

        # Price-snapshot bar offsets (how many bars = 1H / 4H / 24H for this TF)
        _bph = _TF_BARS_PER_HOUR.get(timeframe, 4)
        self._bars_1h  = max(1, round(_bph))
        self._bars_4h  = max(1, round(_bph * 4))
        self._bars_24h = max(1, round(_bph * 24))

        # Virtual RL position (never executed)
        self._position      = 0        # -1 / 0 / +1
        self._entry_price   = 0.0
        self._entry_rr      = 0.0
        self._sl            = 0.0
        self._tp            = 0.0
        self._bars_in_trade = 0
        self._stall_count   = 0
        self._hi_in_trade       = 0.0
        self._lo_in_trade       = float("inf")
        self._max_favorable     = 0.0
        self._trade_quality     = 0.0
        self._peak_pnl_usd      = 0.0   # best USD PnL reached in current trade
        self._trough_pnl_usd    = 0.0   # worst USD PnL reached in current trade
        self._retraced_past_entry = False  # price crossed entry in wrong direction
        self._balance       = float(initial_balance)
        self._equity        = float(initial_balance)
        self._trade_history: list = []
        # ML virtual position tracking (for shadow model obs builder)
        self._ml_vpos   = 0
        self._ml_ventry = 0.0

        # Pending outcome queue for deferred market-did tracking
        # Each entry keeps "db_id" (rl_shadow row id) for later UPDATE
        self._pending_outcomes: List[Dict[str, Any]] = []
        self._outcome_counter = 0

    # ── Constructor from saved model ─────────────────────────────────────────

    @classmethod
    def load(
        cls,
        feature_columns: list,
        timeframe: str = "15min",
        **kwargs,
    ) -> "RLShadow":
        """Load the best available model (shadow-specific preferred over generic MTF)."""
        try:
            from stable_baselines3 import PPO
        except ImportError:
            raise ImportError(
                "stable-baselines3 is required.  pip install stable-baselines3"
            )

        model_path = is_mtf = is_shadow = None
        for path, mtf, shadow in _MODEL_CANDIDATES:
            if os.path.exists(path):
                model_path = path
                is_mtf     = mtf
                is_shadow  = shadow
                break

        if model_path is None:
            raise FileNotFoundError(
                "No trained RL model found. Train one first:\n"
                "  python rl/train_shadow.py          (shadow-specific, recommended)\n"
                "  python rl/train_rl_mtf.py --from-cache  (generic multi-TF)"
            )

        logger.info(
            "RLShadow loading: %s  (multi_tf=%s  shadow_model=%s)",
            model_path, is_mtf, is_shadow,
        )

        # Shadow model and generic PPO both use PPO.load()
        model = PPO.load(model_path)
        logger.info("Model loaded. Obs dim: %d", model.observation_space.shape[0])

        # Multi-TF models are trained on RL_FEATURE_COLUMNS (81), not the ML
        # bundle's feature list (68) — override to keep obs dims consistent
        if is_mtf:
            from config.pipeline_config import RL_FEATURE_COLUMNS as _RL_FC
            feature_columns = list(_RL_FC)
            logger.info("Multi-TF model: using RL_FEATURE_COLUMNS (%d features)", len(feature_columns))

        return cls(
            model           = model,
            feature_columns = feature_columns,
            timeframe       = timeframe,
            is_multi_tf     = is_mtf,
            is_shadow_model = is_shadow,
            **kwargs,
        )

    # ── DB log initialisation (no-op — engine is created lazily in db_writer) ──

    def _init_logs(self) -> None:
        pass

    # ── Observation builder ──────────────────────────────────────────────────

    def _build_obs(
        self,
        primary_row: pd.Series,
        secondary_rows: Optional[Dict[str, pd.Series]] = None,
        ml_signal_dict: Optional[dict] = None,
    ) -> np.ndarray:
        fc = self.feature_columns
        parts = [primary_row.reindex(fc, fill_value=0).fillna(0).values.astype(np.float32)]

        if self.is_multi_tf:
            # Always build one block per secondary TF; use zeros when not provided
            for tf in self.secondary_tfs:
                if secondary_rows and tf in secondary_rows:
                    parts.append(
                        secondary_rows[tf].reindex(fc, fill_value=0).fillna(0).values.astype(np.float32)
                    )
                else:
                    parts.append(np.zeros(len(fc), dtype=np.float32))

        close = float(primary_row.get("close", 0) or 0)

        if self.is_shadow_model:
            # Shadow model expects ML context block + position context block
            ml = ml_signal_dict or {}
            ml_sig  = float(ml.get("signal", 0) or 0)
            ml_prob = float(ml.get("prob",   0) or 0)
            ml_gr   = {"A": 1.0, "B": 0.67, "C": 0.33}.get(
                str(ml.get("grade", "") or "").strip().upper(), 0.0
            )
            # ML virtual PnL from current virtual position
            ml_vpnl = self._ml_virtual_pnl(close) / max(self.initial_balance, 1)
            htf4h   = float(primary_row.get("htf_4h_bias",       0) or 0)
            atr_r   = float(primary_row.get("macro_atr_ratio", 1.0) or 1.0)
            zone_q  = float(primary_row.get("active_zone_quality", 0) or 0) / 5.0
            vol_r   = float(primary_row.get("volume_ratio", 1.0) or 1.0)

            ml_ctx = np.array([
                np.clip(ml_sig,          -1,  1),
                np.clip(ml_prob,          0,  1),
                ml_gr,
                np.clip(ml_vpnl,         -1,  1),
                np.clip(htf4h,           -1,  1),
                np.clip(atr_r - 1.0,     -1,  2),
                np.clip(zone_q,           0,  1),
                np.clip(vol_r - 1.0,     -1,  2),
            ], dtype=np.float32)
            parts.append(ml_ctx)

            upnl  = self._unrealized_pnl(close) / max(self.initial_balance, 1)
            risk  = abs(close - self._sl) if (self._position != 0 and self._sl != 0) else 1.0
            max_fav = self._max_favorable / max(risk, 1e-6) if self._position != 0 else 0.0
            mom5  = float(primary_row.get("momentum_5", 0) or 0)
            mom_n = float(np.clip(mom5 * self._position, -1, 1)) if self._position != 0 else 0.0
            still = float(self._stall_count < STALL_BARS and self._position != 0)
            tq    = self._trade_quality / 5.0

            pos_ctx = np.array([
                float(self._position),
                np.clip(upnl,    -1, 1),
                min(self._bars_in_trade / 50.0, 1.0),
                mom_n,
                still,
                np.clip(max_fav, 0, 3),
                np.clip(tq, 0, 1),
            ], dtype=np.float32)
            parts.append(pos_ctx)

        else:
            # Generic MTF model: position context only (3 values)
            ctx = np.array([
                float(self._position),
                np.clip(self._unrealized_pnl(close) / self.initial_balance, -1.0, 1.0),
                min(self._bars_in_trade / 100.0, 1.0),
            ], dtype=np.float32)
            parts.append(ctx)

        return np.concatenate(parts)

    # ── PnL helpers ──────────────────────────────────────────────────────────

    def _unrealized_pnl(self, price: float) -> float:
        if self._position == 0:
            return 0.0
        return (price - self._entry_price) * self._position * self.lot_size * self.contract_size

    def _ml_virtual_pnl(self, price: float) -> float:
        if self._ml_vpos == 0:
            return 0.0
        return (price - self._ml_ventry) * self._ml_vpos * self.lot_size * self.contract_size

    # ── SL/TP computation ────────────────────────────────────────────────────

    def _compute_sl_tp(self, row: pd.Series, direction: int) -> Tuple:
        close = float(row.get("close", 0) or 0)
        atr   = float(row.get("atr_14", 0) or 0)
        if atr <= 0:
            return None, None, None

        def _f(k: str) -> float:
            v = row.get(k, np.nan)
            try:
                v = float(v)
                return np.nan if np.isnan(v) else v
            except (TypeError, ValueError):
                return np.nan

        if direction == 1:
            d_bot = _f("demand_zone_bottom")
            s_bot = _f("supply_zone_bottom")
            if np.isnan(d_bot):
                return None, None, None
            sl   = d_bot - self.sl_atr_buffer * atr
            risk = close - sl
            if risk <= 0:
                return None, None, None
            tp = (close + s_bot - close) if (not np.isnan(s_bot) and s_bot > close) \
                 else close + max(self.min_rr * risk, 3.0 * atr)
            rr = (tp - close) / risk
        else:
            s_top = _f("supply_zone_top")
            d_top = _f("demand_zone_top")
            if np.isnan(s_top):
                return None, None, None
            sl   = s_top + self.sl_atr_buffer * atr
            risk = sl - close
            if risk <= 0:
                return None, None, None
            tp = (close - (close - d_top)) if (not np.isnan(d_top) and d_top < close) \
                 else close - max(self.min_rr * risk, 3.0 * atr)
            rr = (close - tp) / risk

        if rr < self.min_rr:
            return None, None, None
        return round(sl, 5), round(tp, 5), round(rr, 3)

    # ── Virtual position management ──────────────────────────────────────────

    def _check_sl_tp_hit(self, row: pd.Series) -> Tuple[bool, bool]:
        if self._position == 0:
            return False, False
        hi = float(row.get("high", 0) or 0)
        lo = float(row.get("low",  0) or 0)
        return (lo <= self._sl, hi >= self._tp) if self._position == 1 \
               else (hi >= self._sl, lo <= self._tp)

    def _update_virtual_position(self, row: pd.Series) -> Optional[dict]:
        """
        Resolve virtual SL/TP on this bar.
        Returns close event dict or None if still open.
        """
        if self._position == 0:
            return None

        # Update stall tracking
        hi = float(row.get("high", 0) or 0)
        lo = float(row.get("low",  0) or 0)
        if self._position == 1:
            if hi > self._hi_in_trade:
                self._hi_in_trade = hi
                self._stall_count = 0
            else:
                self._stall_count += 1
        else:
            if lo < self._lo_in_trade:
                self._lo_in_trade = lo
                self._stall_count = 0
            else:
                self._stall_count += 1
        fav = (hi - self._entry_price) if self._position == 1 else (self._entry_price - lo)
        self._max_favorable = max(self._max_favorable, max(fav, 0.0))

        # Track peak and trough PnL using bar high/low for accuracy
        bar_best  = hi if self._position == 1 else lo
        bar_worst = lo if self._position == 1 else hi
        pnl_best  = (bar_best  - self._entry_price) * self._position * self.lot_size * self.contract_size
        pnl_worst = (bar_worst - self._entry_price) * self._position * self.lot_size * self.contract_size
        self._peak_pnl_usd   = max(self._peak_pnl_usd,   pnl_best)
        self._trough_pnl_usd = min(self._trough_pnl_usd, pnl_worst)

        # Flag if price crossed entry in the wrong direction
        if not self._retraced_past_entry:
            retraced = (lo < self._entry_price) if self._position == 1 else (hi > self._entry_price)
            if retraced:
                self._retraced_past_entry = True

        sl_hit, tp_hit = self._check_sl_tp_hit(row)
        if not (sl_hit or tp_hit):
            return None

        exit_price = self._sl if sl_hit else self._tp
        pnl = (exit_price - self._entry_price) * self._position \
              * self.lot_size * self.contract_size
        self._balance += pnl
        close_event = {
            "pnl":        pnl,
            "exit":       "sl" if sl_hit else "tp",
            "exit_price": exit_price,
            "direction":  self._position,
        }
        self._trade_history.append(close_event)
        self._position              = 0
        self._bars_in_trade         = 0
        self._entry_rr              = 0.0
        self._stall_count           = 0
        self._max_favorable         = 0.0
        self._peak_pnl_usd          = 0.0
        self._trough_pnl_usd        = 0.0
        self._retraced_past_entry   = False
        return close_event

    def _open_virtual_position(self, row: pd.Series, direction: int) -> bool:
        close = float(row.get("close", 0) or 0)
        sl, tp, rr = self._compute_sl_tp(row, direction)
        if sl is None:
            return False
        self._position              = direction
        self._entry_price           = close
        self._sl                    = sl
        self._tp                    = tp
        self._entry_rr              = rr or 0.0
        self._bars_in_trade         = 0
        self._stall_count           = 0
        self._max_favorable         = 0.0
        self._peak_pnl_usd          = 0.0
        self._trough_pnl_usd        = 0.0
        self._retraced_past_entry   = False
        self._hi_in_trade           = float(row.get("high", close) or close)
        self._lo_in_trade           = float(row.get("low",  close) or close)
        self._trade_quality         = float(row.get("active_zone_quality", 0) or 0)
        if self._ml_vpos == 0:
            self._ml_vpos   = direction
            self._ml_ventry = close
        return True

    # ── Pending outcome tracking ─────────────────────────────────────────────

    def _enqueue_outcome(
        self,
        outcome_id:  int,
        action_name: str,
        agreement:   str,
        direction:   int,
        entry:       float,
        sl:          Optional[float],
        tp:          Optional[float],
        rr:          Optional[float],
        decision_ts: str,
        db_id:       Optional[int]   = None,   # rl_shadow row id for UPDATE on resolve
        ml_sl:       Optional[float] = None,
        ml_tp:       Optional[float] = None,
    ) -> None:
        if sl is None:
            return
        self._pending_outcomes.append({
            "outcome_id":         outcome_id,
            "db_id":              db_id,           # rl_shadow row id
            "decision_timestamp": decision_ts,
            "rl_action_name":     action_name,
            "agreement":          agreement,
            "direction":          direction,
            "entry_price":        entry,
            "sl":                 sl,
            "tp":                 tp,
            "rr":                 rr or 0.0,
            "ml_sl":              ml_sl,
            "ml_tp":              ml_tp,
            "age":                0,
            "price_1h_later":     None,
            "price_4h_later":     None,
            "price_24h_later":    None,
            "mfe_price":          0.0,
            "mae_price":          0.0,
            "ml_resolved":        False,
            "ml_exit_price":      None,
        })

    def _resolve_pending_outcomes(self, row: pd.Series) -> None:
        hi    = float(row.get("high",  0) or 0)
        lo    = float(row.get("low",   0) or 0)
        close = float(row.get("close", 0) or 0)
        still_pending = []

        for p in self._pending_outcomes:
            p["age"] += 1

            # Fill price snapshots at fixed bar offsets regardless of trade state
            if p["price_1h_later"]  is None and p["age"] >= self._bars_1h:
                p["price_1h_later"]  = round(close, 5)
            if p["price_4h_later"]  is None and p["age"] >= self._bars_4h:
                p["price_4h_later"]  = round(close, 5)
            if p["price_24h_later"] is None and p["age"] >= self._bars_24h:
                p["price_24h_later"] = round(close, 5)

            pos    = p["direction"]

            # Update MFE / MAE using bar high/low for accuracy
            if pos == 1:    # long
                p["mfe_price"] = max(p["mfe_price"], max(hi - p["entry_price"], 0.0))
                p["mae_price"] = max(p["mae_price"], max(p["entry_price"] - lo, 0.0))
            else:           # short
                p["mfe_price"] = max(p["mfe_price"], max(p["entry_price"] - lo, 0.0))
                p["mae_price"] = max(p["mae_price"], max(hi - p["entry_price"], 0.0))
            sl_hit = (lo <= p["sl"]) if pos == 1 else (hi >= p["sl"])
            tp_hit = (hi >= p["tp"]) if pos == 1 else (lo <= p["tp"])

            # Track ML SL/TP hit independently for comparison PnL
            if not p["ml_resolved"] and p["ml_sl"] is not None and p["ml_tp"] is not None:
                ml_sl_hit = (lo <= p["ml_sl"]) if pos == 1 else (hi >= p["ml_sl"])
                ml_tp_hit = (hi >= p["ml_tp"]) if pos == 1 else (lo <= p["ml_tp"])
                if ml_tp_hit or ml_sl_hit:
                    p["ml_exit_price"] = p["ml_tp"] if ml_tp_hit else p["ml_sl"]
                    p["ml_resolved"]   = True
                elif p["age"] >= OUTCOME_TIMEOUT:
                    p["ml_exit_price"] = close
                    p["ml_resolved"]   = True

            if tp_hit or sl_hit or p["age"] >= OUTCOME_TIMEOUT:
                exit_type  = "TP" if tp_hit else ("SL" if sl_hit else "timeout")
                exit_price = p["tp"] if tp_hit else (p["sl"] if sl_hit else close)
                pnl        = (exit_price - p["entry_price"]) * pos \
                             * self.lot_size * self.contract_size
                pnl_pips   = _signed_pips(exit_price - p["entry_price"], pos)
                market_did = "won" if tp_hit else ("lost" if sl_hit else "stalled")

                # ML counterfactual PnL
                ml_exit = p["ml_exit_price"] if p["ml_exit_price"] is not None else close
                if p["ml_sl"] is not None:
                    ml_pnl      = (ml_exit - p["entry_price"]) * pos \
                                  * self.lot_size * self.contract_size
                    ml_pnl_pips = _signed_pips(ml_exit - p["entry_price"], pos)
                else:
                    ml_pnl      = ""
                    ml_pnl_pips = ""

                rl_vs_ml = round(pnl - ml_pnl, 2) if ml_pnl != "" else ""

                # RL was right if: agreed+won, OR skipped+lost
                agreement = p["agreement"]
                if agreement in ("AGREE", "ROTATE", "OVERRIDE"):
                    rl_was_right = 1 if tp_hit else (0 if sl_hit else "")
                elif agreement == "SKIP":
                    rl_was_right = 1 if sl_hit else (0 if tp_hit else "")
                else:
                    rl_was_right = ""

                if p.get("db_id") is not None:
                    mfe_usd = round(p["mfe_price"] * self.lot_size * self.contract_size, 2)
                    mae_usd = round(p["mae_price"] * self.lot_size * self.contract_size, 2)
                    rl_hyp  = round(pnl, 2)
                    ml_act  = round(ml_pnl, 2) if ml_pnl != "" else None
                    pnl_diff = round(rl_hyp - ml_act, 2) if ml_act is not None else None
                    _rl_correct: Optional[bool] = None
                    if rl_was_right == 1:
                        _rl_correct = True
                    elif rl_was_right == 0:
                        _rl_correct = False
                    update_rl_outcome(
                        row_id              = p["db_id"],
                        price_1h_later      = p["price_1h_later"],
                        price_4h_later      = p["price_4h_later"],
                        price_24h_later     = p["price_24h_later"],
                        max_favourable      = mfe_usd,
                        max_adverse         = mae_usd,
                        ml_actual_pnl       = ml_act,
                        rl_hypothetical_pnl = rl_hyp,
                        pnl_difference      = pnl_diff,
                        rl_was_correct      = _rl_correct,
                    )
            else:
                still_pending.append(p)

        self._pending_outcomes = still_pending

    # ── Human-readable reason generation ─────────────────────────────────────

    def _generate_reason(
        self,
        action: int,
        row: pd.Series,
        ml_sig: dict,
        rl_direction: int,
    ) -> str:
        ml_dir  = ml_sig.get("signal", 0)
        ml_prob = float(ml_sig.get("prob", 0) or 0)
        ml_grade = str(ml_sig.get("grade", "") or "").strip().upper()

        htf4h    = float(row.get("htf_4h_bias",       0) or 0)
        htf1h    = float(row.get("htf_1h_bias",       0) or 0)
        zone_q   = float(row.get("active_zone_quality", 0) or 0)
        atr_r    = float(row.get("macro_atr_ratio", 1.0) or 1.0)
        vol_r    = float(row.get("volume_ratio",   1.0) or 1.0)
        mom5     = float(row.get("momentum_5",       0) or 0)
        in_sess  = float(row.get("in_session",       0) or 0) == 1.0
        adx      = float(row.get("macro_adx_14",   0.0) or 0) * 100

        dir_name = {1: "BUY", -1: "SELL", 0: "FLAT"}.get(ml_dir, "FLAT")

        if ml_dir == 0:
            if self._position != 0 and action in (1, 2):
                return (f"EXIT: No ML signal, trade stalling "
                        f"({self._stall_count} bars, {self._bars_in_trade} held)")
            return "HOLD: No ML signal this bar"

        counter_htf = (ml_dir == 1 and htf4h < -COUNTER_HTF_THRESHOLD) or \
                      (ml_dir == -1 and htf4h > COUNTER_HTF_THRESHOLD)
        aligned_htf = (ml_dir == 1 and htf4h > ALIGNED_HTF_THRESHOLD) or \
                      (ml_dir == -1 and htf4h < -ALIGNED_HTF_THRESHOLD)
        low_momentum = atr_r < LOW_MOMENTUM_ATR_RATIO
        low_volume   = vol_r < LOW_VOLUME_RATIO
        weak_zone    = zone_q < WEAK_ZONE_QUALITY
        strong_zone  = zone_q >= STRONG_ZONE_QUALITY
        high_conf    = ml_prob >= 0.65

        if action == 0:  # AGREE
            parts = [f"AGREED {dir_name}"]
            if aligned_htf:
                htf_str = "bullish" if htf4h > 0 else "bearish"
                parts.append(f"4H {htf_str} ({htf4h:+.2f})")
            if strong_zone:
                parts.append(f"A-grade zone ({zone_q:.1f}/5)")
            elif zone_q >= 2.5:
                parts.append(f"zone ({zone_q:.1f}/5)")
            if high_conf:
                parts.append(f"high confidence ({ml_prob:.0%})")
            elif ml_grade:
                parts.append(f"Grade {ml_grade} ({ml_prob:.0%})")
            if adx > 25:
                parts.append(f"trending ADX={adx:.0f}")
            if in_sess:
                parts.append("in-session")
            return " | ".join(parts)

        elif action == 1:  # SKIP
            parts = [f"SKIPPED {dir_name}"]
            if counter_htf:
                htf_str = "bearish" if htf4h < 0 else "bullish"
                against = "BUY" if ml_dir == 1 else "SELL"
                parts.append(f"counter-HTF (4H {htf_str} {htf4h:+.2f} vs {against})")
            elif low_momentum:
                parts.append(f"low momentum (ATR×{atr_r:.2f}, shallow move likely)")
            elif low_volume:
                parts.append(f"thin volume (×{vol_r:.2f})")
            elif weak_zone:
                parts.append(f"weak zone ({zone_q:.1f}/5)")
            elif not in_sess:
                parts.append("off-session (London/NY closed)")
            elif ml_grade == "B":
                parts.append(f"Grade B filtered (zone={zone_q:.1f} prob={ml_prob:.0%})")
            else:
                parts.append(f"RL confidence below threshold ({ml_prob:.0%})")
            return " | ".join(parts)

        elif action == 2:  # ROTATE
            if self._position != 0:
                dir_was = "BUY" if self._position == 1 else "SELL"
                return (f"ROTATED: Exiting {dir_was} ({self._bars_in_trade} bars, "
                        f"{self._stall_count} bars stalling) → new {dir_name} "
                        f"Grade={ml_grade} zone={zone_q:.1f} prob={ml_prob:.0%}")
            return f"ROTATE→AGREE {dir_name}: Was flat, entering new setup"

        return f"action={_ACTION_NAMES.get(action, '?')} ml={dir_name}"

    # ── Agreement classification ──────────────────────────────────────────────

    @staticmethod
    def _classify_agreement(ml_signal: int, rl_direction: int, action: int) -> str:
        if ml_signal == 0:
            return "HOLD"
        if action == 1:
            return "SKIP"
        if action == 2:
            return "ROTATE"
        # action == 0
        if rl_direction == ml_signal:
            return "AGREE"
        if rl_direction == -ml_signal and rl_direction != 0:
            return "OVERRIDE"
        return "AGREE"

    # ── DB writer ─────────────────────────────────────────────────────────────

    def _write_decision(self, row: pd.Series, rl_sig: dict, ml_sig: dict,
                        agreement: str, reason: str, outcome_id: int,
                        rl_confidence: Optional[float] = None) -> Optional[int]:
        """INSERT rl_shadow row via SQLAlchemy. Returns DB id for outcome tracking."""
        ml    = ml_sig or {}
        ts    = _parse_ts(row.get("timestamp", ""))
        close = float(row.get("close", 0) or 0)
        ml_dir = int(ml.get("signal", 0) or 0)
        htf4h  = float(row.get("htf_4h_bias", 0) or 0)

        better_avail = (rl_sig.get("action") == 2 and self._position != 0
                        and self._position != rl_sig.get("signal", 0))
        body   = abs(float(row.get("body_size", 0) or 0))
        candle = max(float(row.get("candle_size", 1e-9) or 1e-9), 1e-9)

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
            session                 = _session_name(row),
            momentum_score          = round(float(row.get("momentum_5", 0) or 0), 4),
            atr_ratio               = round(float(row.get("macro_atr_ratio", 1.0) or 1.0), 3),
            volume_ratio            = round(float(row.get("volume_ratio", 1.0) or 1.0), 3),
            candle_body_ratio       = round(body / candle, 4),
            in_trade                = (self._position != 0),
            trade_duration          = self._bars_in_trade,
            unrealised_pnl          = round(self._unrealized_pnl(close), 2),
            is_stalling             = (self._stall_count >= STALL_BARS),
            better_setup_available  = better_avail,
            better_setup_direction  = rl_sig.get("signal") if better_avail else None,
            better_setup_quality    = float(row.get("active_zone_quality", 0) or 0),
            rl_decision             = int(rl_sig.get("action", 0)),
            rl_confidence           = rl_confidence,
            rl_reason               = reason,
            rl_suggested_entry      = close if rl_sig.get("signal", 0) != 0 else None,
            rl_suggested_sl         = rl_sig.get("sl"),
            rl_suggested_tp         = rl_sig.get("tp"),
            rl_recommended_rotation = (rl_sig.get("action") == 2),
        )

    # ── Console shadow summary ────────────────────────────────────────────────

    def _log_console_summary(
        self,
        ts: str,
        ml_sig: dict,
        rl_action: int,
        rl_direction: int,
        agreement: str,
        reason: str,
        sl: Optional[float],
        tp: Optional[float],
        rr: Optional[float],
    ) -> None:
        ml_dir_str = {1: "BUY", -1: "SELL", 0: "flat"}.get(int(ml_sig.get("signal", 0) or 0), "flat")
        rl_dir_str = {1: "BUY", -1: "SELL", 0: "hold"}.get(rl_direction, "hold")

        logger.info(
            "━━ RL SHADOW ━━  %s\n"
            "   ML:  %-6s  grade=%-2s  prob=%.2f  sl=%s tp=%s rr=%s\n"
            "   RL:  %-6s  action=%s  sl=%s tp=%s rr=%s\n"
            "   %-8s  %s\n"
            "   vpos=%+d  veq=%.2f  stall=%d  trades=%d",
            ts,
            ml_dir_str,
            ml_sig.get("grade", "–"),
            float(ml_sig.get("prob", 0) or 0),
            ml_sig.get("sl") or "–",
            ml_sig.get("tp") or "–",
            ml_sig.get("rr") or "–",
            rl_dir_str,
            _ACTION_NAMES.get(rl_action, "?"),
            f"{sl:.5f}" if sl is not None else "–",
            f"{tp:.5f}" if tp is not None else "–",
            f"{rr:.2f}" if rr is not None else "–",
            agreement,
            reason,
            self._position,
            self._equity,
            self._stall_count,
            len(self._trade_history),
        )

    # ── Main observe() entry point ────────────────────────────────────────────

    def observe(
        self,
        feat_df: Optional[pd.DataFrame],
        secondary_feat_dfs: Optional[Dict[str, pd.DataFrame]] = None,
        ml_signal: Optional[dict] = None,
    ) -> dict:
        """
        Run the RL model on the latest completed bar.

        Args:
            feat_df:            Feature DataFrame for the primary TF (from build_features).
                                Uses the last row as the current bar.
            secondary_feat_dfs: {tf_name: feature_df} for multi-TF model.
            ml_signal:          Signal dict from evaluate_bar() for comparison logging.
                                Expected keys: signal, sl, tp, rr, prob, grade, reason.

        Returns:
            {
              "action":    0 (agree/hold) | 1 (skip) | 2 (rotate),
              "signal":    0 | 1 (buy) | -1 (sell),
              "sl":        float | None,
              "tp":        float | None,
              "rr":        float | None,
              "reason":    str  (human-readable shadow explanation),
              "agreement": str  (AGREE | SKIP | ROTATE | OVERRIDE | HOLD),
            }
        """
        no_trade = {
            "action": 0, "signal": 0, "sl": None, "tp": None, "rr": None,
            "reason": "empty feat_df", "agreement": "HOLD",
        }

        if feat_df is None or feat_df.empty:
            return no_trade

        row = feat_df.iloc[-1]
        ts  = str(row.get("timestamp", ""))

        # Resolve pending outcome tracking
        self._resolve_pending_outcomes(row)

        # Settle any open virtual position
        close_event = self._update_virtual_position(row)
        if close_event is not None:
            logger.info(
                "Virtual position closed: %s  pnl=%.2f",
                close_event["exit"].upper(), close_event["pnl"],
            )

        # Build secondary rows for multi-TF obs
        secondary_rows: Optional[Dict[str, pd.Series]] = None
        if self.is_multi_tf and secondary_feat_dfs:
            secondary_rows = {
                tf: df.iloc[-1]
                for tf, df in secondary_feat_dfs.items()
                if df is not None and not df.empty
            }

        ml_sig = ml_signal or {}

        # Build observation and get model prediction
        obs    = self._build_obs(row, secondary_rows, ml_sig)
        action = int(self.model.predict(obs, deterministic=True)[0])

        # Extract action probability as confidence score
        rl_confidence = None
        try:
            import torch
            obs_t = torch.as_tensor(obs.reshape(1, -1), device=self.model.device)
            with torch.no_grad():
                dist = self.model.policy.get_distribution(obs_t)
                rl_confidence = round(float(dist.distribution.probs[0, action].cpu()), 4)
        except Exception:
            pass

        # Determine RL signal direction and SL/TP
        close      = float(row.get("close", 0) or 0)
        rl_signal  = 0
        sl = tp = rr = None

        ml_dir = int(ml_sig.get("signal", 0) or 0)

        if action == 0 and ml_dir != 0:
            # AGREE: take the ML direction
            if self._position == 0:
                sl, tp, rr = self._compute_sl_tp(row, ml_dir)
                if sl is not None:
                    rl_signal = ml_dir
                    self._open_virtual_position(row, ml_dir)

        elif action == 1:
            # SKIP: no entry — possibly early exit of current position
            if self._position != 0:
                atr_r      = float(row.get("macro_atr_ratio", 1.0) or 1.0)
                upnl       = self._unrealized_pnl(close)
                low_mom    = atr_r < LOW_MOMENTUM_ATR_RATIO
                if upnl > 0 and low_mom:
                    pnl = upnl
                    self._balance += pnl
                    self._trade_history.append({"pnl": pnl, "exit": "conservative_tp"})
                    self._position = 0
                    self._bars_in_trade = 0
                    logger.info("Conservative TP: closed virtual position  pnl=%.2f  (low momentum)", pnl)

        elif action == 2 and ml_dir != 0:
            # ROTATE: exit stalling trade, enter new ML setup
            if self._position != 0 and self._position != ml_dir:
                stalling = self._stall_count >= STALL_BARS or self._bars_in_trade >= 25
                if stalling:
                    pnl = self._unrealized_pnl(close)
                    self._balance += pnl
                    self._trade_history.append({"pnl": pnl, "exit": "rotate"})
                    self._position = 0
                    self._bars_in_trade = 0
                    self._stall_count   = 0
                    # Enter new setup
                    sl, tp, rr = self._compute_sl_tp(row, ml_dir)
                    if sl is not None:
                        rl_signal = ml_dir
                        self._open_virtual_position(row, ml_dir)
            elif self._position == 0:
                # Was flat — treat as AGREE
                sl, tp, rr = self._compute_sl_tp(row, ml_dir)
                if sl is not None:
                    rl_signal = ml_dir
                    self._open_virtual_position(row, ml_dir)

        # Update bars in trade
        if self._position != 0:
            self._bars_in_trade += 1

        self._equity = self._balance + self._unrealized_pnl(close)

        # Generate reason and agreement
        reason    = self._generate_reason(action, row, ml_sig, rl_signal)
        agreement = self._classify_agreement(ml_dir, rl_signal, action)

        # Assign outcome_id for tracking what market does after this decision
        outcome_id = -1
        _ml_sl = ml_sig.get("sl")
        _ml_tp = ml_sig.get("tp")
        _ml_rr = ml_sig.get("rr")
        _ml_sl_f = float(_ml_sl) if _ml_sl is not None else None
        _ml_tp_f = float(_ml_tp) if _ml_tp is not None else None

        if rl_signal != 0 and sl is not None:
            self._outcome_counter += 1
            outcome_id = self._outcome_counter
            self._enqueue_outcome(
                outcome_id  = outcome_id,
                action_name = _ACTION_NAMES.get(action, "?"),
                agreement   = agreement,
                direction   = rl_signal,
                entry       = close,
                sl          = sl,
                tp          = tp,
                rr          = rr,
                decision_ts = ts,
                ml_sl       = _ml_sl_f,
                ml_tp       = _ml_tp_f,
            )
        elif action == 1 and ml_dir != 0:
            # Track skipped signals using ML SL/TP to see what would have happened
            if _ml_sl_f is not None:
                self._outcome_counter += 1
                outcome_id = self._outcome_counter
                self._enqueue_outcome(
                    outcome_id  = outcome_id,
                    action_name = "skip",
                    agreement   = "SKIP",
                    direction   = ml_dir,
                    entry       = close,
                    sl          = _ml_sl_f,
                    tp          = _ml_tp_f,
                    rr          = float(_ml_rr) if _ml_rr else None,
                    decision_ts = ts,
                    ml_sl       = _ml_sl_f,
                    ml_tp       = _ml_tp_f,
                )

        # Write decision row to DB; returned id is used to UPDATE outcome later
        rl_result = {"action": action, "signal": rl_signal, "sl": sl, "tp": tp, "rr": rr}
        db_row_id = self._write_decision(row, rl_result, ml_sig, agreement, reason, outcome_id, rl_confidence)

        # Back-fill db_id into the pending outcome we just enqueued (last item)
        if db_row_id is not None and self._pending_outcomes:
            last = self._pending_outcomes[-1]
            if last.get("outcome_id") == outcome_id and last.get("db_id") is None:
                last["db_id"] = db_row_id

        # Console output
        self._log_console_summary(ts, ml_sig, action, rl_signal, agreement, reason, sl, tp, rr)

        return {**rl_result, "reason": reason, "agreement": agreement}

    # ── Performance summary ───────────────────────────────────────────────────

    def performance_summary(self) -> dict:
        wins   = [t for t in self._trade_history if t["pnl"] > 0]
        losses = [t for t in self._trade_history if t["pnl"] <= 0]
        total  = sum(t["pnl"] for t in self._trade_history)
        n      = len(self._trade_history)
        wr     = len(wins) / max(n, 1)
        pf     = abs(sum(t["pnl"] for t in wins) / sum(t["pnl"] for t in losses)) \
                 if losses else float("inf")
        return {
            "n_trades":       n,
            "win_rate":       round(wr, 3),
            "total_pnl":      round(total, 2),
            "profit_factor":  round(pf, 2),
            "virtual_equity": round(self._equity, 2),
        }

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def close(self) -> None:
        pass  # DB connections managed by SQLAlchemy engine pool — no handles to close
