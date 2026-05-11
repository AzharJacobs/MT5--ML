"""
rl/environment.py — Gym-compatible trading environment for XAUUSD 15min RL training.

Completely isolated from the existing ML system — reads the same feature data
(built by build_features()) but the RL agent learns entry/exit policy from scratch
with no hand-labelled winners/losers.

State:   current bar's feature vector + position context [position, unrealized_pnl, bars_held]
Actions: 0=hold/stay flat, 1=buy, 2=sell
Reward:  step PnL (normalised) − heavy drawdown penalty past 5% − small timestep cost

SL/TP use the same zone-based logic as live_trader.py so RL trades are directly
comparable to the ML system in backtesting.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Optional


class XAUUSDTradingEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(
        self,
        feature_df: pd.DataFrame,
        feature_columns: list,
        initial_balance: float = 5_000.0,
        lot_size: float = 0.01,
        contract_size: float = 10.0,   # XAUUSDm: 10 oz per 0.01 lot
        sl_atr_buffer: float = 0.5,
        min_rr: float = 1.5,
        drawdown_penalty: float = 2.0,
        episode_length: int = 500,     # bars per training episode
    ):
        super().__init__()

        self.df               = feature_df.reset_index(drop=True)
        self.feature_columns  = feature_columns
        self.initial_balance  = initial_balance
        self.lot_size         = lot_size
        self.contract_size    = contract_size
        self.sl_atr_buffer    = sl_atr_buffer
        self.min_rr           = min_rr
        self.drawdown_penalty = drawdown_penalty
        self.episode_length   = episode_length

        # Observation: feature vector + [position, unrealised_pnl_norm, bars_held_norm]
        n_features = len(feature_columns)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(n_features + 3,),
            dtype=np.float32,
        )
        # 0=hold, 1=buy, 2=sell
        self.action_space = spaces.Discrete(3)

        self._reset_state(start_step=0)

    # ─────────────────────────────────────────────────────────────────────────
    #  Internal state helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _reset_state(self, start_step: int = 0):
        self.current_step  = start_step
        self.episode_start = start_step
        self.balance       = self.initial_balance
        self.equity        = self.initial_balance
        self.peak_equity   = self.initial_balance
        self.position      = 0      # -1=short, 0=flat, 1=long
        self.entry_price   = 0.0
        self.sl            = 0.0
        self.tp            = 0.0
        self.bars_in_trade = 0
        self.trade_history = []
        self.max_drawdown  = 0.0

    def _get_obs(self) -> np.ndarray:
        row      = self.df.iloc[self.current_step]
        features = row[self.feature_columns].fillna(0).values.astype(np.float32)
        unreal   = self._unrealized_pnl(float(row["close"]))
        ctx      = np.array([
            float(self.position),
            np.clip(unreal / self.initial_balance, -1.0, 1.0),
            min(self.bars_in_trade / 100.0, 1.0),
        ], dtype=np.float32)
        return np.concatenate([features, ctx])

    def _unrealized_pnl(self, price: float) -> float:
        if self.position == 0:
            return 0.0
        return (price - self.entry_price) * self.position * self.lot_size * self.contract_size

    # ─────────────────────────────────────────────────────────────────────────
    #  SL / TP — mirrors live_trader.py compute_sl_tp exactly
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_sl_tp(self, row: pd.Series, direction: int):
        """Returns (sl, tp, rr) or (None, None, None) if levels unavailable."""
        close = float(row["close"])
        atr   = float(row.get("atr_14", 0) or 0)
        if atr <= 0:
            return None, None, None

        def _f(key):
            v = row.get(key, np.nan)
            try:
                v = float(v)
            except (TypeError, ValueError):
                v = np.nan
            return v

        if direction == 1:  # buy
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

        else:  # sell
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
    #  SL / TP hit detection
    # ─────────────────────────────────────────────────────────────────────────

    def _check_sl_tp(self, row: pd.Series):
        """Returns (sl_hit, tp_hit) based on bar high/low."""
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
        # Random start so the agent sees different market regimes each episode
        max_start = max(0, len(self.df) - self.episode_length - 1)
        start = int(self.np_random.integers(0, max_start + 1)) if max_start > 0 else 0
        self._reset_state(start_step=start)
        return self._get_obs(), {}

    def step(self, action: int):
        row   = self.df.iloc[self.current_step]
        close = float(row["close"])
        reward = 0.0

        # ── 1. Resolve any open SL/TP hit on this bar ────────────────────────
        sl_hit, tp_hit = self._check_sl_tp(row)
        if sl_hit or tp_hit:
            exit_price = self.sl if sl_hit else self.tp
            pnl = (exit_price - self.entry_price) * self.position * self.lot_size * self.contract_size
            self.balance += pnl
            # Scale reward to roughly [-1, +1] range for a $50 account
            reward += pnl / self.initial_balance * 100
            self.trade_history.append({
                "pnl": pnl,
                "exit": "sl" if sl_hit else "tp",
                "step": self.current_step,
            })
            self.position      = 0
            self.bars_in_trade = 0

        # ── 2. Agent action (only acts when flat) ────────────────────────────
        if self.position == 0 and action in (1, 2):
            direction = 1 if action == 1 else -1
            sl, tp, rr = self._compute_sl_tp(row, direction)
            if sl is not None:
                self.position    = direction
                self.entry_price = close
                self.sl          = sl
                self.tp          = tp
                self.bars_in_trade = 0

        # ── 3. Update position age ───────────────────────────────────────────
        if self.position != 0:
            self.bars_in_trade += 1

        # ── 4. Update equity and drawdown ────────────────────────────────────
        self.equity = self.balance + self._unrealized_pnl(close)
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        current_dd = (self.peak_equity - self.equity) / max(self.peak_equity, 1e-9)
        self.max_drawdown = max(self.max_drawdown, current_dd)

        # Heavy penalty for drawdown beyond 5%
        if current_dd > 0.05:
            reward -= self.drawdown_penalty * (current_dd - 0.05) * 100

        # Small per-step cost — prevents the agent sitting flat forever
        reward -= 0.001

        # ── 5. Advance ───────────────────────────────────────────────────────
        self.current_step += 1
        steps_taken  = self.current_step - self.episode_start
        terminated   = self.current_step >= len(self.df) - 1
        truncated    = (
            steps_taken >= self.episode_length
            or self.equity < self.initial_balance * 0.5   # blew 50% → end episode
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
        row = self.df.iloc[self.current_step]
        print(
            f"step={self.current_step} | pos={self.position:+d} | "
            f"equity={self.equity:.2f} | dd={self.max_drawdown:.2%} | "
            f"trades={len(self.trade_history)} | close={row['close']:.2f}"
        )

    def summary(self) -> dict:
        """Call after an episode to get performance stats."""
        wins   = [t for t in self.trade_history if t["pnl"] > 0]
        losses = [t for t in self.trade_history if t["pnl"] <= 0]
        total_pnl = sum(t["pnl"] for t in self.trade_history)
        win_rate  = len(wins) / max(len(self.trade_history), 1)
        avg_win   = np.mean([t["pnl"] for t in wins])   if wins   else 0.0
        avg_loss  = np.mean([t["pnl"] for t in losses]) if losses else 0.0
        pf = abs(sum(t["pnl"] for t in wins) / sum(t["pnl"] for t in losses)) \
             if losses else float("inf")
        return {
            "total_pnl":    round(total_pnl, 2),
            "n_trades":     len(self.trade_history),
            "win_rate":     round(win_rate, 3),
            "avg_win":      round(avg_win,  2),
            "avg_loss":     round(avg_loss, 2),
            "profit_factor": round(pf, 2),
            "max_drawdown": round(self.max_drawdown, 4),
            "final_equity": round(self.equity, 2),
        }
