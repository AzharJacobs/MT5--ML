"""
rl/environment_raw.py — Self-learning trading environment.

The agent receives ONLY normalised raw OHLCV windows. No zones, no rule columns,
no HTF bias labels, no confirmation patterns. Everything must be learned from price.

Observation per step:
  For each timeframe (15min, 1H, 4H):
    Last `window_size` (default 50) candles ×
    [open_norm, high_norm, low_norm, close_norm, volume_norm]   = 5 values/candle

  Plus time context:  [hour/23, day_of_week/6, session/3]       = 3 values
  Plus position ctx:  [position, unrealised_pnl_norm, bars/100] = 3 values

  Total: 3 × 50 × 5 + 3 + 3 = 756 dims

Normalisation (per-window, done live):
  Prices  → Z-score using that window's close mean/std
  Volume  → ratio to window mean volume

SL/TP: ATR-based. sl = entry ± sl_atr × ATR14, tp = entry ± tp_atr × ATR14.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Optional

_DAY_MAP = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
            "Friday": 4, "Saturday": 5, "Sunday": 6}


class XAUUSDRawEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(
        self,
        tf_dfs: dict,                    # {"15min": df, "1H": df, "4H": df}
        primary_tf: str         = "15min",
        window_size: int        = 50,
        atr_period: int         = 14,
        sl_atr: float           = 1.5,
        tp_atr: float           = 2.5,
        initial_balance: float  = 5_000.0,
        lot_size: float         = 0.01,
        contract_size: float    = 10.0,
        drawdown_penalty: float = 3.0,
        sl_extra_penalty: float = 1.0,
        rr_bonus_factor: float  = 0.25,
        episode_length: int     = 500,
    ):
        super().__init__()

        if primary_tf not in tf_dfs:
            raise ValueError(f"primary_tf '{primary_tf}' not in tf_dfs: {list(tf_dfs.keys())}")

        self.primary_tf       = primary_tf
        self.window_size      = window_size
        self.atr_period       = atr_period
        self.sl_atr           = sl_atr
        self.tp_atr           = tp_atr
        self.initial_balance  = initial_balance
        self.lot_size         = lot_size
        self.contract_size    = contract_size
        self.drawdown_penalty = drawdown_penalty
        self.sl_extra_penalty = sl_extra_penalty
        self.rr_bonus_factor  = rr_bonus_factor
        self.episode_length   = episode_length

        self.primary_df    = tf_dfs[primary_tf].reset_index(drop=True)
        self.secondary_tfs = sorted(tf for tf in tf_dfs if tf != primary_tf)
        self.secondary_dfs = {tf: tf_dfs[tf].reset_index(drop=True) for tf in self.secondary_tfs}

        self._primary_atr = self._compute_atr(self.primary_df, atr_period)
        self._aligned     = self._precompute_alignment()

        n_tfs = 1 + len(self.secondary_tfs)
        n_obs = n_tfs * window_size * 5 + 3 + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self._reset_state(start_step=window_size)

    # ─────────────────────────────────────────────────────────────────────────
    #  ATR precomputation
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int) -> np.ndarray:
        high  = df["high"].values.astype(np.float64)
        low   = df["low"].values.astype(np.float64)
        close = df["close"].values.astype(np.float64)
        n     = len(df)
        atr   = np.zeros(n, dtype=np.float64)
        tr    = np.empty(n)
        tr[0] = high[0] - low[0]
        tr[1:] = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
        )
        atr[period - 1] = tr[:period].mean()
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        first = float(atr[atr > 0][0]) if np.any(atr > 0) else 1.0
        atr[:period - 1] = first
        return atr.astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    #  Timestamp alignment
    # ─────────────────────────────────────────────────────────────────────────

    def _precompute_alignment(self) -> dict:
        primary_ts = pd.to_datetime(self.primary_df["timestamp"]).values.astype(np.int64)
        aligned = {}
        for tf, df in self.secondary_dfs.items():
            sec_ts  = pd.to_datetime(df["timestamp"]).values.astype(np.int64)
            indices = np.searchsorted(sec_ts, primary_ts, side="right") - 1
            indices = np.clip(indices, 0, len(df) - 1)
            aligned[tf] = indices
        return aligned

    # ─────────────────────────────────────────────────────────────────────────
    #  State
    # ─────────────────────────────────────────────────────────────────────────

    def _reset_state(self, start_step: int):
        self.current_step  = start_step
        self.episode_start = start_step
        self.balance       = self.initial_balance
        self.equity        = self.initial_balance
        self.peak_equity   = self.initial_balance
        self.position      = 0
        self.entry_price   = 0.0
        self.entry_rr      = 0.0
        self.sl            = 0.0
        self.tp            = 0.0
        self.bars_in_trade = 0
        self.trade_history = []
        self.max_drawdown  = 0.0

    # ─────────────────────────────────────────────────────────────────────────
    #  Observation
    # ─────────────────────────────────────────────────────────────────────────

    def _window_features(self, df: pd.DataFrame, end_idx: int) -> np.ndarray:
        ws    = self.window_size
        start = max(0, end_idx - ws + 1)
        chunk = df.iloc[start : end_idx + 1]

        closes   = chunk["close"].values.astype(np.float32)
        mu       = closes.mean()
        sigma    = float(closes.std()) + 1e-8
        price    = (chunk[["open", "high", "low", "close"]].values.astype(np.float32) - mu) / sigma
        vol      = chunk["volume"].values.astype(np.float32)
        vol_norm = (vol / (float(vol.mean()) + 1e-9)).reshape(-1, 1)
        features = np.hstack([price, vol_norm])

        if len(features) < ws:
            pad      = np.zeros((ws - len(features), 5), dtype=np.float32)
            features = np.vstack([pad, features])

        return features.flatten().astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        step = self.current_step
        row  = self.primary_df.iloc[step]
        parts = [self._window_features(self.primary_df, step)]

        for tf in self.secondary_tfs:
            idx = self._aligned[tf][step]
            parts.append(self._window_features(self.secondary_dfs[tf], idx))

        hour = float(row.get("hour", 0) or 0)
        _dow = row.get("day_of_week", 0)
        dow  = float(_DAY_MAP.get(_dow, _dow) if isinstance(_dow, str) else (_dow or 0))
        session = float(row.get("session_id", 0) or 0)
        parts.append(np.array([hour / 23.0, dow / 6.0, session / 3.0], dtype=np.float32))

        close = float(row["close"])
        ctx = np.array([
            float(self.position),
            np.clip(self._unrealized_pnl(close) / self.initial_balance, -1.0, 1.0),
            min(self.bars_in_trade / 100.0, 1.0),
        ], dtype=np.float32)
        parts.append(ctx)

        return np.concatenate(parts)

    # ─────────────────────────────────────────────────────────────────────────
    #  SL / TP
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_sl_tp(self, step: int, direction: int):
        atr   = float(self._primary_atr[step])
        close = float(self.primary_df.iloc[step]["close"])
        if atr <= 0:
            return None, None, None
        sl_dist = self.sl_atr * atr
        tp_dist = self.tp_atr * atr
        if direction == 1:
            sl, tp = close - sl_dist, close + tp_dist
        else:
            sl, tp = close + sl_dist, close - tp_dist
        rr = round(tp_dist / sl_dist, 3)
        return round(sl, 5), round(tp, 5), rr

    def _check_sl_tp(self, row: pd.Series):
        if self.position == 0:
            return False, False
        high, low = float(row["high"]), float(row["low"])
        if self.position == 1:
            return low <= self.sl, high >= self.tp
        else:
            return high >= self.sl, low <= self.tp

    def _unrealized_pnl(self, price: float) -> float:
        if self.position == 0:
            return 0.0
        return (price - self.entry_price) * self.position * self.lot_size * self.contract_size

    # ─────────────────────────────────────────────────────────────────────────
    #  Gym interface
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        lo = self.window_size
        hi = max(lo, len(self.primary_df) - self.episode_length - 1)
        start = int(self.np_random.integers(lo, hi + 1)) if hi > lo else lo
        self._reset_state(start_step=start)
        return self._get_obs(), {}

    def step(self, action: int):
        row   = self.primary_df.iloc[self.current_step]
        close = float(row["close"])
        reward = 0.0

        sl_hit, tp_hit = self._check_sl_tp(row)
        if sl_hit or tp_hit:
            exit_price = self.sl if sl_hit else self.tp
            pnl = (exit_price - self.entry_price) * self.position \
                  * self.lot_size * self.contract_size
            self.balance += pnl
            pnl_norm = pnl / self.initial_balance * 100
            if sl_hit:
                reward += pnl_norm * (1.0 + self.sl_extra_penalty)
            else:
                rr_bonus = self.rr_bonus_factor * max(self.entry_rr - (self.tp_atr / self.sl_atr), 0.0)
                reward   += pnl_norm * (1.0 + rr_bonus)
            self.trade_history.append({"pnl": pnl, "rr": self.entry_rr,
                                       "exit": "sl" if sl_hit else "tp",
                                       "step": self.current_step})
            self.position = 0
            self.bars_in_trade = 0
            self.entry_rr = 0.0

        if self.position == 0 and action in (1, 2):
            direction = 1 if action == 1 else -1
            sl, tp, rr = self._compute_sl_tp(self.current_step, direction)
            if sl is not None:
                self.position      = direction
                self.entry_price   = close
                self.sl, self.tp   = sl, tp
                self.entry_rr      = rr
                self.bars_in_trade = 0

        if self.position != 0:
            self.bars_in_trade += 1

        self.equity = self.balance + self._unrealized_pnl(close)
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        current_dd = (self.peak_equity - self.equity) / max(self.peak_equity, 1e-9)
        self.max_drawdown = max(self.max_drawdown, current_dd)

        if current_dd > 0.05:
            reward -= self.drawdown_penalty * (current_dd - 0.05) * 100
        reward -= 0.001

        self.current_step += 1
        steps_taken = self.current_step - self.episode_start
        terminated  = self.current_step >= len(self.primary_df) - 1
        truncated   = (steps_taken >= self.episode_length
                       or self.equity < self.initial_balance * 0.50)

        return self._get_obs(), float(reward), terminated, truncated, {
            "balance": self.balance, "equity": self.equity,
            "position": self.position, "max_drawdown": self.max_drawdown,
            "n_trades": len(self.trade_history),
        }

    def render(self):
        row = self.primary_df.iloc[self.current_step]
        print(f"step={self.current_step} pos={self.position:+d} eq={self.equity:.2f} "
              f"dd={self.max_drawdown:.2%} trades={len(self.trade_history)} "
              f"close={row['close']:.2f} atr={self._primary_atr[self.current_step]:.3f}")

    def summary(self) -> dict:
        wins      = [t for t in self.trade_history if t["pnl"] > 0]
        losses    = [t for t in self.trade_history if t["pnl"] <= 0]
        total_pnl = sum(t["pnl"] for t in self.trade_history)
        win_rate  = len(wins) / max(len(self.trade_history), 1)
        avg_win   = float(np.mean([t["pnl"] for t in wins]))   if wins   else 0.0
        avg_loss  = float(np.mean([t["pnl"] for t in losses])) if losses else 0.0
        pf        = abs(sum(t["pnl"] for t in wins) / sum(t["pnl"] for t in losses)) \
                    if losses else float("inf")
        calmar    = (total_pnl / self.initial_balance) / max(self.max_drawdown, 1e-6)
        return {
            "total_pnl":      round(total_pnl, 2),
            "n_trades":       len(self.trade_history),
            "win_rate":       round(win_rate, 3),
            "avg_win":        round(avg_win, 2),
            "avg_loss":       round(avg_loss, 2),
            "profit_factor":  round(pf, 2),
            "max_drawdown":   round(self.max_drawdown, 4),
            "calmar_ratio":   round(calmar, 3),
            "final_equity":   round(self.equity, 2),
            "theoretical_rr": round(self.tp_atr / self.sl_atr, 2),
        }
