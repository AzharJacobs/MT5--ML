"""
rl/environment_mtf.py — Multi-timeframe Gym environment for XAUUSD PPO training.

Observation space: concatenated feature vectors from all supplied timeframes
(default: 5min, 15min, 1H, 4H) + position context [position, unrealised_pnl, bars_held].
Primary decision clock: 15min bars (or whichever is set as primary_tf).
Secondary TFs are aligned to the primary by timestamp (latest bar <= primary bar ts).

Reward is optimised for:
  - Clean realised PnL (no extra reward for paper gains)
  - Tight SL discipline  (1+sl_extra_penalty multiplier on SL-hit losses)
  - Risk-adjusted returns (bonus proportional to RR above min_rr on TP hits)
  - Drawdown control     (heavy penalty above 5% equity drawdown)
  - Activity incentive   (small per-step cost to prevent staying flat forever)
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Optional


class XAUUSDMultiTFEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(
        self,
        tf_dfs: dict,                   # {"5min": df, "15min": df, "1H": df, "4H": df}
        feature_columns: list,
        primary_tf: str         = "15min",
        initial_balance: float  = 5_000.0,
        lot_size: float         = 0.01,
        contract_size: float    = 10.0,   # XAUUSDm: 10 oz per 0.01 lot
        sl_atr_buffer: float    = 0.5,
        min_rr: float           = 1.5,
        drawdown_penalty: float = 3.0,    # multiplier when drawdown > 5%
        sl_extra_penalty: float = 1.0,    # applied as (1 + sl_extra_penalty) on SL-hit pnl
        rr_bonus_factor: float  = 0.25,   # reward bonus per unit of RR above min_rr on TP hit
        episode_length: int     = 500,
    ):
        super().__init__()

        if primary_tf not in tf_dfs:
            raise ValueError(f"primary_tf '{primary_tf}' not in tf_dfs: {list(tf_dfs.keys())}")

        self.feature_columns  = feature_columns
        self.primary_tf       = primary_tf
        self.initial_balance  = initial_balance
        self.lot_size         = lot_size
        self.contract_size    = contract_size
        self.sl_atr_buffer    = sl_atr_buffer
        self.min_rr           = min_rr
        self.drawdown_penalty = drawdown_penalty
        self.sl_extra_penalty = sl_extra_penalty
        self.rr_bonus_factor  = rr_bonus_factor
        self.episode_length   = episode_length

        self.primary_df    = tf_dfs[primary_tf].reset_index(drop=True)
        # Secondary TFs: sorted for deterministic obs stacking
        self.secondary_tfs = sorted(tf for tf in tf_dfs if tf != primary_tf)
        self.secondary_dfs = {tf: tf_dfs[tf].reset_index(drop=True) for tf in self.secondary_tfs}

        self._aligned = self._precompute_alignment()

        n_tfs  = 1 + len(self.secondary_tfs)
        n_obs  = n_tfs * len(feature_columns) + 3   # +3 for position context
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)   # 0=hold, 1=buy, 2=sell

        self._reset_state(start_step=0)

    # ─────────────────────────────────────────────────────────────────────────
    #  Timestamp alignment
    # ─────────────────────────────────────────────────────────────────────────

    def _precompute_alignment(self) -> dict:
        """
        For each primary bar i, find the most recent index in each secondary TF
        where secondary_ts <= primary_ts[i].  O(n log n) via searchsorted.
        """
        primary_ts = pd.to_datetime(self.primary_df["timestamp"]).values.astype(np.int64)
        aligned = {}
        for tf, df in self.secondary_dfs.items():
            sec_ts  = pd.to_datetime(df["timestamp"]).values.astype(np.int64)
            indices = np.searchsorted(sec_ts, primary_ts, side="right") - 1
            indices = np.clip(indices, 0, len(df) - 1)
            aligned[tf] = indices
        return aligned

    # ─────────────────────────────────────────────────────────────────────────
    #  State management
    # ─────────────────────────────────────────────────────────────────────────

    def _reset_state(self, start_step: int = 0):
        self.current_step  = start_step
        self.episode_start = start_step
        self.balance       = self.initial_balance
        self.equity        = self.initial_balance
        self.peak_equity   = self.initial_balance
        self.position      = 0       # -1=short, 0=flat, 1=long
        self.entry_price   = 0.0
        self.entry_rr      = 0.0
        self.sl            = 0.0
        self.tp            = 0.0
        self.bars_in_trade = 0
        self.trade_history = []
        self.max_drawdown  = 0.0

    # ─────────────────────────────────────────────────────────────────────────
    #  Observation builder
    # ─────────────────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        fc  = self.feature_columns
        row = self.primary_df.iloc[self.current_step]
        parts = [row[fc].fillna(0).values.astype(np.float32)]

        for tf in self.secondary_tfs:
            idx     = self._aligned[tf][self.current_step]
            sec_row = self.secondary_dfs[tf].iloc[idx]
            parts.append(sec_row[fc].fillna(0).values.astype(np.float32))

        close = float(row["close"])
        ctx = np.array([
            float(self.position),
            np.clip(self._unrealized_pnl(close) / self.initial_balance, -1.0, 1.0),
            min(self.bars_in_trade / 100.0, 1.0),
        ], dtype=np.float32)
        parts.append(ctx)

        return np.concatenate(parts)

    def _unrealized_pnl(self, price: float) -> float:
        if self.position == 0:
            return 0.0
        return (price - self.entry_price) * self.position * self.lot_size * self.contract_size

    # ─────────────────────────────────────────────────────────────────────────
    #  SL / TP — mirrors live_trader.py compute_sl_tp exactly
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_sl_tp(self, row: pd.Series, direction: int):
        """Returns (sl, tp, rr) using zone levels or (None, None, None) if unavailable."""
        close = float(row["close"])
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

        if direction == 1:   # buy
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

        else:                # sell
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

    # ─────────────────────────────────────────────────────────────────────────
    #  SL / TP hit detection (bar high/low)
    # ─────────────────────────────────────────────────────────────────────────

    def _check_sl_tp(self, row: pd.Series):
        if self.position == 0:
            return False, False
        high = float(row["high"])
        low  = float(row["low"])
        if self.position == 1:
            return low <= self.sl, high >= self.tp
        else:
            return high >= self.sl, low <= self.tp

    # ─────────────────────────────────────────────────────────────────────────
    #  Gym interface
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        max_start = max(0, len(self.primary_df) - self.episode_length - 1)
        start     = int(self.np_random.integers(0, max_start + 1)) if max_start > 0 else 0
        self._reset_state(start_step=start)
        return self._get_obs(), {}

    def step(self, action: int):
        row   = self.primary_df.iloc[self.current_step]
        close = float(row["close"])
        reward = 0.0

        # ── 1. Resolve SL/TP hit from previous bar ────────────────────────────
        sl_hit, tp_hit = self._check_sl_tp(row)
        if sl_hit or tp_hit:
            exit_price = self.sl if sl_hit else self.tp
            pnl        = (exit_price - self.entry_price) * self.position \
                         * self.lot_size * self.contract_size
            self.balance += pnl
            pnl_norm = pnl / self.initial_balance * 100

            if sl_hit:
                # Extra penalty on SL hit: (1 + sl_extra_penalty) × negative pnl
                reward += pnl_norm * (1.0 + self.sl_extra_penalty)
            else:
                # TP hit: bonus for trades with RR above floor
                rr_bonus = self.rr_bonus_factor * max(self.entry_rr - self.min_rr, 0.0)
                reward   += pnl_norm * (1.0 + rr_bonus)

            self.trade_history.append({
                "pnl":  pnl,
                "rr":   self.entry_rr,
                "exit": "sl" if sl_hit else "tp",
                "step": self.current_step,
            })
            self.position      = 0
            self.bars_in_trade = 0
            self.entry_rr      = 0.0

        # ── 2. Agent action (only when flat) ──────────────────────────────────
        if self.position == 0 and action in (1, 2):
            direction = 1 if action == 1 else -1
            sl, tp, rr = self._compute_sl_tp(row, direction)
            if sl is not None:
                self.position      = direction
                self.entry_price   = close
                self.sl            = sl
                self.tp            = tp
                self.entry_rr      = rr
                self.bars_in_trade = 0

        # ── 3. Update position age ────────────────────────────────────────────
        if self.position != 0:
            self.bars_in_trade += 1

        # ── 4. Equity and drawdown ────────────────────────────────────────────
        self.equity = self.balance + self._unrealized_pnl(close)
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        current_dd = (self.peak_equity - self.equity) / max(self.peak_equity, 1e-9)
        self.max_drawdown = max(self.max_drawdown, current_dd)

        if current_dd > 0.05:
            reward -= self.drawdown_penalty * (current_dd - 0.05) * 100

        # ── 5. Per-step cost (discourages staying flat forever) ───────────────
        reward -= 0.001

        # ── 6. Advance ────────────────────────────────────────────────────────
        self.current_step += 1
        steps_taken = self.current_step - self.episode_start
        terminated  = self.current_step >= len(self.primary_df) - 1
        truncated   = (
            steps_taken >= self.episode_length
            or self.equity < self.initial_balance * 0.50   # -50% → end episode early
        )

        info = {
            "balance":      self.balance,
            "equity":       self.equity,
            "position":     self.position,
            "max_drawdown": self.max_drawdown,
            "n_trades":     len(self.trade_history),
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        row = self.primary_df.iloc[self.current_step]
        print(
            f"step={self.current_step} pos={self.position:+d} "
            f"eq={self.equity:.2f} dd={self.max_drawdown:.2%} "
            f"trades={len(self.trade_history)} close={row['close']:.2f}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  Episode summary (call after episode ends)
    # ─────────────────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        wins      = [t for t in self.trade_history if t["pnl"] > 0]
        losses    = [t for t in self.trade_history if t["pnl"] <= 0]
        total_pnl = sum(t["pnl"] for t in self.trade_history)
        win_rate  = len(wins) / max(len(self.trade_history), 1)
        avg_win   = float(np.mean([t["pnl"] for t in wins]))   if wins   else 0.0
        avg_loss  = float(np.mean([t["pnl"] for t in losses])) if losses else 0.0
        pf        = abs(sum(t["pnl"] for t in wins) / sum(t["pnl"] for t in losses)) \
                    if losses else float("inf")
        avg_rr    = float(np.mean([t["rr"] for t in self.trade_history])) \
                    if self.trade_history else 0.0
        calmar    = (total_pnl / self.initial_balance) / max(self.max_drawdown, 1e-6)

        return {
            "total_pnl":     round(total_pnl, 2),
            "n_trades":      len(self.trade_history),
            "win_rate":      round(win_rate, 3),
            "avg_win":       round(avg_win, 2),
            "avg_loss":      round(avg_loss, 2),
            "profit_factor": round(pf, 2),
            "avg_rr":        round(avg_rr, 3),
            "max_drawdown":  round(self.max_drawdown, 4),
            "calmar_ratio":  round(calmar, 3),
            "final_equity":  round(self.equity, 2),
        }
