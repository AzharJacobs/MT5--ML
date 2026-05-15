"""
rl/shadow.py — Passive RL observer that runs alongside the live ML bot.

Receives the same feature data that live_trader.py builds each candle, runs
the trained PPO model, and logs what the RL agent *would* have done — but
NEVER touches the broker or places any order.

Virtual position state is tracked so the model's context vector (position,
unrealised PnL, bars held) stays coherent across candles, matching the training
environment exactly.

Model priority (first found wins):
  1. rl/models/best_model_mtf/best_model.zip  (multi-TF)
  2. rl/models/final_model_mtf.zip            (multi-TF)
  3. rl/models/best_model/best_model.zip      (single-TF)
  4. rl/models/final_model.zip                (single-TF)

Usage in live_trader.py:
    from rl.shadow import RLShadow
    shadow = RLShadow.load(feature_columns=bundle["feature_columns"])
    ...
    # inside run_once(), after evaluate_bar():
    rl_sig = shadow.observe(feat_df, ml_signal=sig)
    logger.info("RL shadow: %s", rl_sig["reason"])

Output: logs/rl_shadow_YYYY-MM-DD.csv  (one row per bar)
"""

import os
import csv
import logging
import numpy as np
import pandas as pd
from datetime import date
from typing import Optional

logger = logging.getLogger("rl.shadow")

_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(_THIS_DIR, "models")
LOGS_DIR   = os.path.join(os.path.dirname(_THIS_DIR), "logs")

_MODEL_CANDIDATES = [
    (os.path.join(MODELS_DIR, "best_model_mtf", "best_model.zip"), True),
    (os.path.join(MODELS_DIR, "final_model_mtf.zip"),              True),
    (os.path.join(MODELS_DIR, "best_model", "best_model.zip"),     False),
    (os.path.join(MODELS_DIR, "final_model.zip"),                  False),
]

_LOG_COLS = [
    "timestamp", "symbol", "timeframe",
    "rl_action", "rl_signal", "rl_sl", "rl_tp", "rl_rr",
    "ml_signal", "ml_grade", "ml_prob",
    "virtual_position", "virtual_equity", "virtual_n_trades",
    "bar_close", "in_demand_zone", "in_supply_zone", "zone_quality",
]

_ACTION_NAMES = {0: "hold", 1: "buy", 2: "sell"}


class RLShadow:
    """Passive observer — predicts, tracks virtually, logs. Never executes."""

    def __init__(
        self,
        model,
        feature_columns: list,
        timeframe: str         = "15min",
        symbol: str            = "XAUUSDm",
        initial_balance: float = 5_000.0,
        lot_size: float        = 0.01,
        contract_size: float   = 10.0,
        sl_atr_buffer: float   = 0.5,
        min_rr: float          = 1.5,
        is_multi_tf: bool      = False,
        secondary_tfs: list    = None,
    ):
        self.model            = model
        self.feature_columns  = feature_columns
        self.timeframe        = timeframe
        self.symbol           = symbol
        self.initial_balance  = initial_balance
        self.lot_size         = lot_size
        self.contract_size    = contract_size
        self.sl_atr_buffer    = sl_atr_buffer
        self.min_rr           = min_rr
        self.is_multi_tf      = is_multi_tf
        self.secondary_tfs    = sorted(secondary_tfs or ["5min", "1H", "4H"])

        # Virtual position state (mirrors training environment)
        self._position      = 0       # -1=short, 0=flat, 1=long
        self._entry_price   = 0.0
        self._entry_rr      = 0.0
        self._sl            = 0.0
        self._tp            = 0.0
        self._bars_in_trade = 0
        self._balance       = float(initial_balance)
        self._equity        = float(initial_balance)
        self._trade_history: list = []

        self._log_file   = None
        self._log_writer = None
        self._init_log()

    # ─────────────────────────────────────────────────────────────────────────
    #  Class constructor from saved model
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        feature_columns: list,
        timeframe: str = "15min",
        **kwargs,
    ) -> "RLShadow":
        """Load the best available PPO model (MTF preferred over single-TF)."""
        try:
            from stable_baselines3 import PPO
        except ImportError:
            raise ImportError("stable-baselines3 is required for RLShadow. pip install stable-baselines3")

        model_path = None
        is_multi_tf = False
        for path, mtf in _MODEL_CANDIDATES:
            if os.path.exists(path):
                model_path  = path
                is_multi_tf = mtf
                break

        if model_path is None:
            raise FileNotFoundError(
                "No trained RL model found. Run first:\n"
                "  python rl/train_rl.py      (single-TF)\n"
                "  python rl/train_rl_mtf.py  (multi-TF, recommended)"
            )

        logger.info("RLShadow loading model: %s  (multi_tf=%s)", model_path, is_multi_tf)
        model = PPO.load(model_path)

        return cls(
            model=model,
            feature_columns=feature_columns,
            timeframe=timeframe,
            is_multi_tf=is_multi_tf,
            **kwargs,
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  CSV log
    # ─────────────────────────────────────────────────────────────────────────

    def _init_log(self):
        os.makedirs(LOGS_DIR, exist_ok=True)
        path   = os.path.join(LOGS_DIR, f"rl_shadow_{date.today().isoformat()}.csv")
        exists = os.path.exists(path)
        self._log_file   = open(path, "a", newline="", encoding="utf-8")
        self._log_writer = csv.DictWriter(self._log_file, fieldnames=_LOG_COLS)
        if not exists:
            self._log_writer.writeheader()

    def _log(self, row: pd.Series, rl_sig: dict, ml_sig: Optional[dict]):
        if self._log_writer is None:
            return
        ml = ml_sig or {}
        self._log_writer.writerow({
            "timestamp":         str(row.get("timestamp", "")),
            "symbol":            self.symbol,
            "timeframe":         self.timeframe,
            "rl_action":         rl_sig.get("action", 0),
            "rl_signal":         rl_sig.get("signal", 0),
            "rl_sl":             rl_sig.get("sl", ""),
            "rl_tp":             rl_sig.get("tp", ""),
            "rl_rr":             rl_sig.get("rr", ""),
            "ml_signal":         ml.get("signal", 0),
            "ml_grade":          ml.get("grade", ""),
            "ml_prob":           ml.get("prob", ""),
            "virtual_position":  self._position,
            "virtual_equity":    round(self._equity, 2),
            "virtual_n_trades":  len(self._trade_history),
            "bar_close":         float(row.get("close", 0)),
            "in_demand_zone":    int(float(row.get("in_demand_zone", 0) or 0)),
            "in_supply_zone":    int(float(row.get("in_supply_zone", 0) or 0)),
            "zone_quality":      round(float(row.get("active_zone_quality", 0) or 0), 2),
        })
        self._log_file.flush()

    # ─────────────────────────────────────────────────────────────────────────
    #  Observation builder (matches training environments)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_obs(
        self,
        primary_row: pd.Series,
        secondary_rows: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Build the model's input vector.
        secondary_rows: {tf_name: pd.Series} — only used for multi-TF model.
        """
        fc = self.feature_columns
        parts = [primary_row[fc].fillna(0).values.astype(np.float32)]

        if self.is_multi_tf and secondary_rows:
            for tf in self.secondary_tfs:
                if tf in secondary_rows:
                    parts.append(secondary_rows[tf][fc].fillna(0).values.astype(np.float32))
                else:
                    # Secondary TF not available — zero-pad so obs shape is unchanged
                    parts.append(np.zeros(len(fc), dtype=np.float32))

        close = float(primary_row.get("close", 0))
        ctx = np.array([
            float(self._position),
            np.clip(self._unrealized_pnl(close) / self.initial_balance, -1.0, 1.0),
            min(self._bars_in_trade / 100.0, 1.0),
        ], dtype=np.float32)
        parts.append(ctx)

        return np.concatenate(parts)

    # ─────────────────────────────────────────────────────────────────────────
    #  Virtual position helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _unrealized_pnl(self, price: float) -> float:
        if self._position == 0:
            return 0.0
        return (price - self._entry_price) * self._position * self.lot_size * self.contract_size

    def _compute_sl_tp(self, row: pd.Series, direction: int):
        close = float(row.get("close", 0))
        atr   = float(row.get("atr_14", 0) or 0)
        if atr <= 0:
            return None, None, None

        def _f(key):
            v = row.get(key, np.nan)
            try:
                v = float(v)
                return np.nan if np.isnan(v) else v
            except (TypeError, ValueError):
                return np.nan

        if direction == 1:
            d_bottom = _f("demand_zone_bottom")
            s_bottom = _f("supply_zone_bottom")
            if np.isnan(d_bottom):
                return None, None, None
            sl   = d_bottom - self.sl_atr_buffer * atr
            risk = close - sl
            if risk <= 0:
                return None, None, None
            tp = (close + (s_bottom - close)) if (not np.isnan(s_bottom) and s_bottom > close) \
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

    def _update_virtual_position(self, row: pd.Series):
        """Check if virtual SL/TP was hit on this bar and settle."""
        if self._position == 0:
            return
        high = float(row.get("high", 0))
        low  = float(row.get("low", 0))
        if self._position == 1:
            sl_hit, tp_hit = low <= self._sl, high >= self._tp
        else:
            sl_hit, tp_hit = high >= self._sl, low <= self._tp

        if sl_hit or tp_hit:
            exit_price = self._sl if sl_hit else self._tp
            pnl = (exit_price - self._entry_price) * self._position \
                  * self.lot_size * self.contract_size
            self._balance += pnl
            self._trade_history.append({
                "pnl":  pnl,
                "exit": "sl" if sl_hit else "tp",
                "rr":   self._entry_rr,
            })
            self._position      = 0
            self._bars_in_trade = 0
            self._entry_rr      = 0.0

    # ─────────────────────────────────────────────────────────────────────────
    #  Main observation entry point
    # ─────────────────────────────────────────────────────────────────────────

    def observe(
        self,
        feat_df: pd.DataFrame,
        secondary_feat_dfs: Optional[dict] = None,
        ml_signal: Optional[dict] = None,
    ) -> dict:
        """
        Run the RL model on the latest completed bar.

        Args:
            feat_df:            Feature DataFrame for the primary TF (from build_features).
                                Uses the last row as the current bar.
            secondary_feat_dfs: {tf_name: feature_df} for multi-TF model.
                                Pass None when secondary data is unavailable — the model
                                will receive zero-padded secondary features.
            ml_signal:          Signal dict from evaluate_bar() for comparison logging.

        Returns:
            {
              "action":  0 (hold) | 1 (buy) | 2 (sell),
              "signal":  0 (no trade) | 1 (buy) | -1 (sell),
              "sl":      float | None,
              "tp":      float | None,
              "rr":      float | None,
              "reason":  str,
            }
        """
        no_trade = {"action": 0, "signal": 0, "sl": None, "tp": None, "rr": None}

        if feat_df is None or feat_df.empty:
            return {**no_trade, "reason": "empty feat_df"}

        row = feat_df.iloc[-1]

        # Settle any open virtual position against this bar's high/low
        self._update_virtual_position(row)

        # Build obs
        secondary_rows = None
        if self.is_multi_tf and secondary_feat_dfs:
            secondary_rows = {
                tf: df.iloc[-1]
                for tf, df in secondary_feat_dfs.items()
                if df is not None and not df.empty
            }

        obs    = self._build_obs(row, secondary_rows)
        action = int(self.model.predict(obs, deterministic=True)[0])

        signal  = 0
        sl = tp = rr = None

        if self._position == 0 and action in (1, 2):
            direction = 1 if action == 1 else -1
            sl, tp, rr = self._compute_sl_tp(row, direction)
            if sl is not None:
                signal = direction
                self._position      = direction
                self._entry_price   = float(row.get("close", 0))
                self._sl            = sl
                self._tp            = tp
                self._entry_rr      = rr or 0.0
                self._bars_in_trade = 0

        if self._position != 0:
            self._bars_in_trade += 1

        close         = float(row.get("close", 0))
        self._equity  = self._balance + self._unrealized_pnl(close)

        action_name = _ACTION_NAMES.get(action, "?")
        reason = (
            f"action={action_name} vpos={self._position:+d} "
            f"veq={self._equity:.2f} trades={len(self._trade_history)}"
        )

        result = {"action": action, "signal": signal, "sl": sl, "tp": tp, "rr": rr, "reason": reason}
        self._log(row, result, ml_signal)
        return result

    # ─────────────────────────────────────────────────────────────────────────
    #  Reporting
    # ─────────────────────────────────────────────────────────────────────────

    def performance_summary(self) -> dict:
        wins      = [t for t in self._trade_history if t["pnl"] > 0]
        losses    = [t for t in self._trade_history if t["pnl"] <= 0]
        total_pnl = sum(t["pnl"] for t in self._trade_history)
        n         = len(self._trade_history)
        win_rate  = len(wins) / max(n, 1)
        pf = abs(sum(t["pnl"] for t in wins) / sum(t["pnl"] for t in losses)) \
             if losses else float("inf")
        return {
            "n_trades":      n,
            "win_rate":      round(win_rate, 3),
            "total_pnl":     round(total_pnl, 2),
            "profit_factor": round(pf, 2),
            "virtual_equity": round(self._equity, 2),
        }

    def close(self):
        if self._log_file:
            self._log_file.close()
            self._log_file = None
