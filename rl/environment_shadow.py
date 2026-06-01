"""
rl/environment_shadow.py — Trade Optimizer shadow environment for XAUUSD.

The agent watches ML bot signals and learns HOW to trade them better:

  action[0]  TAKE/SKIP   — 0=skip this signal,  1=take it
  action[1]  SL_MODE     — 0=ML_SL (passthrough), 1=TIGHT (0.65×), 2=WIDE (1.4×)
  action[2]  TP_MODE     — 0=ML_TP (passthrough), 1=CONSERVATIVE (1.5× RR), 2=EXTENDED (2.5× RR)

18 combinations. SL/TP modes only matter when action[0]=TAKE.

Obs (339-dim):
  4 TFs × 81 RL_FEATURE_COLUMNS = 324
  ML context block (8)           =   8
  Position context block (7)     =   7

Rewards teach the RL to beat the ML's raw PnL:
  - TIGHT SL when the zone is clean → tighter risk, better RR
  - WIDE SL when the zone is noisy  → avoids whipsaw SL hits
  - EXTENDED TP when momentum is strong → let winners run
  - CONSERVATIVE TP when momentum is weak → lock in early
  - Skip off-session / counter-HTF signals
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, List, Dict, Any


# ── Constants ──────────────────────────────────────────────────────────────────
STALL_BARS         = 8
SKIP_EVAL_TIMEOUT  = 60
MAX_TRADES_PER_EP  = 20

# SL ATR-buffer multipliers per session
_SESSION_SL_BUFFER = {
    "Asian":   0.80,
    "Off":     0.80,
    "London":  0.50,
    "NY":      0.50,
    "Overlap": 0.45,
}

# action[1] SL scale factors applied to the base ATR buffer
_SL_SCALE = {0: 1.00, 1: 0.65, 2: 1.40}

# action[2] TP RR multipliers (CONSERVATIVE / EXTENDED override ML_TP entirely)
_TP_RR = {1: 1.5, 2: 2.5}


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


class XAUUSDShadowEnv(gym.Env):
    """Trade optimizer shadow environment."""

    metadata = {"render_modes": []}
    N_ML_CTX  = 8
    N_POS_CTX = 7

    def __init__(
        self,
        primary_df:      pd.DataFrame,
        secondary_dfs:   Dict[str, pd.DataFrame],
        feature_columns: List[str],
        initial_balance: float = 5_000.0,
        lot_size:        float = 0.01,
        contract_size:   float = 10.0,
        min_rr:          float = 1.5,
        episode_length:  int   = 2000,
        # Reward weights
        skip_loser_reward:   float = 2.0,
        skip_winner_penalty: float = 1.5,
        stall_penalty:       float = 0.08,
        htf_penalty:         float = 1.0,
        rr_bonus_factor:     float = 0.3,
        drawdown_penalty:    float = 2.0,
    ):
        super().__init__()

        self.primary_df      = primary_df.reset_index(drop=True)
        self.secondary_dfs   = {tf: df.reset_index(drop=True) for tf, df in secondary_dfs.items()}
        self.feature_columns = feature_columns
        self.secondary_tfs   = sorted(secondary_dfs.keys())
        self.initial_balance = initial_balance
        self.lot_size        = lot_size
        self.contract_size   = contract_size
        self.min_rr          = min_rr
        self.episode_length  = episode_length

        self.skip_loser_reward   = skip_loser_reward
        self.skip_winner_penalty = skip_winner_penalty
        self.stall_penalty       = stall_penalty
        self.htf_penalty         = htf_penalty
        self.rr_bonus_factor     = rr_bonus_factor
        self.drawdown_penalty    = drawdown_penalty

        self._aligned = self._precompute_alignment()

        n_obs = (1 + len(self.secondary_tfs)) * len(feature_columns) + self.N_ML_CTX + self.N_POS_CTX
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32
        )
        # action[0]=take/skip, action[1]=sl_mode, action[2]=tp_mode
        self.action_space = spaces.MultiDiscrete([2, 3, 3])

        self._reset_state(0)

    # ── Alignment ─────────────────────────────────────────────────────────────

    def _precompute_alignment(self) -> Dict[str, np.ndarray]:
        primary_ts = pd.to_datetime(self.primary_df["timestamp"]).values.astype(np.int64)
        aligned: Dict[str, np.ndarray] = {}
        for tf, df in self.secondary_dfs.items():
            sec_ts  = pd.to_datetime(df["timestamp"]).values.astype(np.int64)
            indices = np.searchsorted(sec_ts, primary_ts, side="right") - 1
            aligned[tf] = np.clip(indices, 0, len(df) - 1)
        return aligned

    # ── State ─────────────────────────────────────────────────────────────────

    def _reset_state(self, start: int) -> None:
        self.current_step  = start
        self.episode_start = start
        self.balance       = self.initial_balance
        self.equity        = self.initial_balance
        self.peak_equity   = self.initial_balance
        self.max_drawdown  = 0.0

        # RL virtual position
        self.position      = 0
        self.entry_price   = 0.0
        self.sl            = 0.0
        self.tp            = 0.0
        self.entry_rr      = 0.0
        self.bars_in_trade = 0
        self.stall_count   = 0
        self.max_favorable = 0.0
        self.max_adverse   = 0.0
        self._hi_in_trade  = 0.0
        self._lo_in_trade  = float("inf")
        self.trade_quality = 0.0
        self._entry_sl_mode = 0
        self._entry_tp_mode = 0

        # ML shadow position (what ML would do with its own levels)
        self._ml_pos   = 0
        self._ml_entry = 0.0
        self._ml_sl    = 0.0
        self._ml_tp    = 0.0

        # Pending skip evaluations (for SKIP reward signal)
        self._pending_skips: List[Dict[str, Any]] = []

        self.trade_history: List[Dict[str, Any]] = []

    # ── Observation ───────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        fc  = self.feature_columns
        row = self.primary_df.iloc[self.current_step]

        parts: List[np.ndarray] = [row[fc].fillna(0).values.astype(np.float32)]
        for tf in self.secondary_tfs:
            idx = self._aligned[tf][self.current_step]
            parts.append(self.secondary_dfs[tf].iloc[idx][fc].fillna(0).values.astype(np.float32))

        close = float(row.get("close", 0) or 0)

        # ML context (8 values)
        ml_vpnl = self._ml_virtual_pnl(close) / max(self.initial_balance, 1)
        ml_ctx = np.array([
            np.clip(float(row.get("ml_signal", 0) or 0),        -1, 1),
            np.clip(float(row.get("ml_prob",   0) or 0),         0, 1),
            {"A": 1.0, "B": 0.67, "C": 0.33}.get(
                str(row.get("ml_grade", "") or "").strip().upper(), 0.0),
            np.clip(ml_vpnl,                                     -1, 1),
            np.clip(float(row.get("htf_4h_bias",       0) or 0), -1, 1),
            np.clip(float(row.get("macro_atr_ratio", 1.0) or 1.0) - 1.0, -1, 2),
            np.clip(float(row.get("active_zone_quality", 0) or 0) / 5.0,  0, 1),
            np.clip(float(row.get("volume_ratio", 1.0) or 1.0) - 1.0,    -1, 2),
        ], dtype=np.float32)
        parts.append(ml_ctx)

        # Position context (7 values)
        upnl  = self._unrealized_pnl(close) / max(self.initial_balance, 1)
        risk  = abs(close - self.sl) if (self.position != 0 and self.sl != 0) else 1.0
        mfe_r = self.max_favorable / max(risk, 1e-6) if self.position != 0 else 0.0
        mom5  = float(row.get("momentum_5", 0) or 0)
        mom_n = float(np.clip(mom5 * self.position, -1, 1)) if self.position != 0 else 0.0
        pos_ctx = np.array([
            float(self.position),
            np.clip(upnl,    -1, 1),
            min(self.bars_in_trade / 50.0, 1.0),
            mom_n,
            float(self.stall_count < STALL_BARS and self.position != 0),
            np.clip(mfe_r, 0, 3),
            np.clip(self.trade_quality / 5.0, 0, 1),
        ], dtype=np.float32)
        parts.append(pos_ctx)

        return np.concatenate(parts)

    # ── PnL helpers ───────────────────────────────────────────────────────────

    def _unrealized_pnl(self, price: float) -> float:
        if self.position == 0:
            return 0.0
        return (price - self.entry_price) * self.position * self.lot_size * self.contract_size

    def _ml_virtual_pnl(self, price: float) -> float:
        if self._ml_pos == 0:
            return 0.0
        return (price - self._ml_entry) * self._ml_pos * self.lot_size * self.contract_size

    # ── SL/TP computation ─────────────────────────────────────────────────────

    def _compute_levels(
        self, row: pd.Series, direction: int, sl_mode: int, tp_mode: int
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Return (sl, tp, rr, risk) or (None, None, None, None) if not viable."""
        close = float(row.get("close", 0) or 0)
        atr   = float(row.get("atr_14", 0) or 0)
        if atr <= 0:
            return None, None, None, None

        sess      = _session(row)
        base_buf  = _SESSION_SL_BUFFER.get(sess, 0.50)
        sl_buf    = base_buf * _SL_SCALE[sl_mode]

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

    # ── SL/TP hit detection ───────────────────────────────────────────────────

    def _check_hit(self, pos: int, sl: float, tp: float, row: pd.Series) -> Tuple[bool, bool]:
        if pos == 0:
            return False, False
        hi = float(row.get("high", 0) or 0)
        lo = float(row.get("low",  0) or 0)
        return (lo <= sl, hi >= tp) if pos == 1 else (hi >= sl, lo <= tp)

    # ── Trade progress ────────────────────────────────────────────────────────

    def _update_trade_progress(self, row: pd.Series) -> None:
        if self.position == 0:
            return
        hi = float(row.get("high", 0) or 0)
        lo = float(row.get("low",  0) or 0)
        if self.position == 1:
            fav = hi - self.entry_price
            adv = self.entry_price - lo
            if hi > self._hi_in_trade:
                self._hi_in_trade = hi
                self.stall_count  = 0
            else:
                self.stall_count += 1
        else:
            fav = self.entry_price - lo
            adv = hi - self.entry_price
            if lo < self._lo_in_trade:
                self._lo_in_trade = lo
                self.stall_count  = 0
            else:
                self.stall_count += 1
        self.max_favorable = max(self.max_favorable, max(float(fav), 0.0))
        self.max_adverse   = max(self.max_adverse,   max(float(adv), 0.0))

    # ── Position open / close ─────────────────────────────────────────────────

    def _open_rl(
        self, row: pd.Series, direction: int,
        sl: float, tp: float, rr: float, sl_mode: int, tp_mode: int,
    ) -> None:
        close = float(row.get("close", 0) or 0)
        self.position       = direction
        self.entry_price    = close
        self.sl             = sl
        self.tp             = tp
        self.entry_rr       = rr
        self.bars_in_trade  = 0
        self.stall_count    = 0
        self.max_favorable  = 0.0
        self.max_adverse    = 0.0
        self._entry_sl_mode = sl_mode
        self._entry_tp_mode = tp_mode
        self._hi_in_trade   = float(row.get("high", close) or close)
        self._lo_in_trade   = float(row.get("low",  close) or close)
        self.trade_quality  = float(row.get("active_zone_quality", 0) or 0)

    def _close_rl(self, exit_price: float) -> float:
        pnl = (exit_price - self.entry_price) * self.position \
              * self.lot_size * self.contract_size
        self.balance      += pnl
        self.position      = 0
        self.bars_in_trade = 0
        self.entry_rr      = 0.0
        self.max_favorable = 0.0
        self.max_adverse   = 0.0
        self.stall_count   = 0
        return pnl

    # ── Pending skip evaluation ───────────────────────────────────────────────

    def _evaluate_pending_skips(self, row: pd.Series) -> float:
        reward        = 0.0
        still_pending = []
        for p in self._pending_skips:
            p["age"] += 1
            sl_hit, tp_hit = self._check_hit(p["pos"], p["sl"], p["tp"], row)
            if tp_hit:
                reward -= self.skip_winner_penalty
            elif sl_hit:
                reward += self.skip_loser_reward
            elif p["age"] < SKIP_EVAL_TIMEOUT:
                still_pending.append(p)
        self._pending_skips = still_pending
        return reward

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        max_start = max(0, len(self.primary_df) - self.episode_length - 1)
        start     = int(self.np_random.integers(0, max_start + 1)) if max_start > 0 else 0
        self._reset_state(start)
        return self._get_obs(), {}

    def step(self, action):  # noqa: C901
        take    = int(action[0])
        sl_mode = int(action[1])
        tp_mode = int(action[2])

        row   = self.primary_df.iloc[self.current_step]
        close = float(row.get("close", 0) or 0)
        reward = 0.0

        # ── 1. Evaluate pending skips ─────────────────────────────────────────
        reward += self._evaluate_pending_skips(row)

        # ── 2. Resolve RL position ────────────────────────────────────────────
        if self.position != 0:
            sl_hit, tp_hit = self._check_hit(self.position, self.sl, self.tp, row)
            if sl_hit or tp_hit:
                exit_price = self.sl if sl_hit else self.tp
                pnl        = self._close_rl(exit_price)
                pnl_norm   = pnl / self.initial_balance * 100

                if tp_hit:
                    rr_bonus = self.rr_bonus_factor * max(self.entry_rr - self.min_rr, 0.0)
                    reward  += pnl_norm * (1.0 + rr_bonus)
                    if self._entry_tp_mode == 2:   # EXTENDED TP hit — extra bonus
                        reward += 0.5
                    # Differential: RL vs ML baseline
                    ml_pnl = (exit_price - self._ml_entry) * self._ml_pos \
                             * self.lot_size * self.contract_size if self._ml_pos != 0 else 0.0
                    if pnl > ml_pnl:
                        reward += 0.5
                    # Clean TIGHT_SL win (MAE stayed well inside SL)
                    sl_dist = abs(self.entry_price - self.sl)
                    if self._entry_sl_mode == 1 and self.max_adverse < 0.4 * sl_dist:
                        reward += 0.4
                else:
                    reward += pnl_norm
                    # Penalise if TIGHT_SL was to blame (MAE exceeded SL distance)
                    sl_dist = abs(self.entry_price - self.sl)
                    if self._entry_sl_mode == 1 and self.max_adverse > sl_dist:
                        reward -= 0.3

                self.trade_history.append({"pnl": pnl, "exit": "tp" if tp_hit else "sl"})

        # ── 3. Update ML shadow ───────────────────────────────────────────────
        if self._ml_pos != 0:
            ml_sl_h, ml_tp_h = self._check_hit(self._ml_pos, self._ml_sl, self._ml_tp, row)
            if ml_sl_h or ml_tp_h:
                self._ml_pos = 0

        # ── 4. Update trade progress + stall penalty ──────────────────────────
        self._update_trade_progress(row)
        if self.position != 0 and self.stall_count >= STALL_BARS:
            reward -= self.stall_penalty

        # ── 5. Read bar ML signal ─────────────────────────────────────────────
        ml_signal = int(float(row.get("ml_signal", 0) or 0))
        htf4h     = float(row.get("htf_4h_bias", 0) or 0)
        zone_q    = float(row.get("active_zone_quality", 0) or 0)
        counter_htf = (ml_signal == 1 and htf4h < -0.5) or (ml_signal == -1 and htf4h > 0.5)

        # ── 6. Apply action ───────────────────────────────────────────────────
        if ml_signal != 0:
            if take == 1:  # TAKE
                sl, tp, rr, _ = self._compute_levels(row, ml_signal, sl_mode, tp_mode)
                if sl is not None:
                    # Implicit rotate: if holding the opposite direction, close first
                    if self.position != 0 and self.position != ml_signal:
                        if self.stall_count >= STALL_BARS:
                            rot_pnl  = self._close_rl(close)
                            reward  += rot_pnl / self.initial_balance * 100
                        # else: can't flip without stall confirmation — skip
                    if self.position == 0:
                        self._open_rl(row, ml_signal, sl, tp, rr, sl_mode, tp_mode)
                        if counter_htf:
                            reward -= self.htf_penalty
                        # Open ML shadow with its own levels (sl_mode=0, tp_mode=0)
                        if self._ml_pos == 0:
                            ml_sl0, ml_tp0, _, _ = self._compute_levels(row, ml_signal, 0, 0)
                            if ml_sl0 is not None:
                                self._ml_pos   = ml_signal
                                self._ml_entry = close
                                self._ml_sl    = ml_sl0
                                self._ml_tp    = ml_tp0

            else:  # SKIP
                ml_sl0, ml_tp0, _, _ = self._compute_levels(row, ml_signal, 0, 0)
                if ml_sl0 is not None:
                    self._pending_skips.append({
                        "pos": ml_signal, "entry": close,
                        "sl": ml_sl0, "tp": ml_tp0, "age": 0,
                    })

        # ── 7. Bar counters ───────────────────────────────────────────────────
        if self.position != 0:
            self.bars_in_trade += 1

        # ── 8. Equity / drawdown ──────────────────────────────────────────────
        self.equity = self.balance + self._unrealized_pnl(close)
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        dd = (self.peak_equity - self.equity) / max(self.peak_equity, 1e-9)
        self.max_drawdown = max(self.max_drawdown, dd)
        if dd > 0.05:
            reward -= self.drawdown_penalty * (dd - 0.05) * 100

        # ── 9. Step ───────────────────────────────────────────────────────────
        self.current_step += 1
        steps      = self.current_step - self.episode_start
        terminated = self.current_step >= len(self.primary_df) - 1
        truncated  = steps >= self.episode_length or self.equity < self.initial_balance * 0.5

        if terminated or truncated:
            n = len(self.trade_history)
            if n > MAX_TRADES_PER_EP:
                reward -= 0.05 * (n - MAX_TRADES_PER_EP)
            pnl_pct = (self.equity - self.initial_balance) / self.initial_balance
            if self.max_drawdown > 0 and pnl_pct > 0:
                reward += min(pnl_pct / self.max_drawdown * 0.5, 5.0)

        info = {
            "balance":      self.balance,
            "equity":       self.equity,
            "position":     self.position,
            "stall_count":  self.stall_count,
            "max_drawdown": self.max_drawdown,
            "n_trades":     len(self.trade_history),
            "ml_signal":    ml_signal,
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self) -> None:
        print(
            f"step={self.current_step} pos={self.position:+d} "
            f"eq={self.equity:.2f} dd={self.max_drawdown:.2%} "
            f"stall={self.stall_count}"
        )

    def summary(self) -> dict:
        wins   = [t for t in self.trade_history if t["pnl"] > 0]
        losses = [t for t in self.trade_history if t["pnl"] <= 0]
        total  = sum(t["pnl"] for t in self.trade_history)
        n      = len(self.trade_history)
        wr     = len(wins) / max(n, 1)
        pf     = abs(sum(t["pnl"] for t in wins) / sum(t["pnl"] for t in losses)) \
                 if losses else float("inf")
        by_exit: Dict[str, int] = {}
        for t in self.trade_history:
            k = t.get("exit", "?")
            by_exit[k] = by_exit.get(k, 0) + 1
        return {
            "total_pnl":     round(total, 2),
            "n_trades":      n,
            "win_rate":      round(wr, 3),
            "profit_factor": round(pf, 2),
            "max_drawdown":  round(self.max_drawdown, 4),
            "final_equity":  round(self.equity, 2),
            "exits_by_type": by_exit,
        }
