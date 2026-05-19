"""
rl/environment_shadow.py — Shadow-mode RL environment for XAUUSD.

Trains an agent that observes the ML bot's signals and learns WHEN to:
  AGREE  (0): Take the ML signal; stay in current position if already open
  SKIP   (1): Filter the ML signal; decline entry — or exit early if in trade
              on low-momentum bars (conservative TP)
  ROTATE (2): Exit a stalling current trade and immediately enter the new ML
              signal setup

The ML bot's simulated signals (ml_signal, ml_prob, ml_grade pre-computed from
feature data) are part of the observation, so the agent learns context-aware
filtering rather than trading independently from scratch.

Reward signals:
  - Counter-factual skip evaluation (pending queue):
      • skip a loser  → +skip_loser_reward  (delivered when virtual ML position hits SL)
      • skip a winner → -skip_winner_penalty (delivered when virtual ML position hits TP)
  - HTF bias penalty:  immediate penalty for entering against 4H direction
  - Stall penalty:     accumulating cost per bar while holding a stalling trade
  - Rotation bonus:    credited when rotating out of a confirmed stall into a new setup
  - Conservative TP:   partial credit for early exit on low-momentum bars
  - RR bonus:          extra reward on high-RR winners
  - Calmar episode bonus: rewards consistent equity growth vs drawdown

Observation space (339 dims):
  4 TFs × 81 RL_FEATURE_COLUMNS  = 324   (66 REQUIRED + 15 macro)
  ML context block (8)            = 8
  Position context block (7)      = 7
  ─────────────────────────────────
  Total                           = 339

Action space:  Discrete(3) — 0=AGREE, 1=SKIP, 2=ROTATE
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, List, Dict, Any


# ── Tuning constants ──────────────────────────────────────────────────────────
STALL_BARS             = 8      # bars without new high/low → trade is stalling
LOW_MOMENTUM_ATR_RATIO = 0.75   # below this = low-momentum market
MAX_TRADES_PER_EP      = 20     # overtrading penalty threshold
SKIP_EVAL_TIMEOUT      = 60     # bars before a pending skip evaluation expires


class XAUUSDShadowEnv(gym.Env):
    """
    Shadow training environment.

    primary_df must contain, in addition to RL_FEATURE_COLUMNS and raw OHLCV:
        ml_signal  : int    (-1 / 0 / 1)   — pre-computed ML bot direction
        ml_prob    : float                  — ML model win-probability
        ml_grade   : str                    — "A" / "B" / "C" / ""
    """

    metadata = {"render_modes": []}

    N_ML_CTX  = 8   # size of ML-context observation block
    N_POS_CTX = 7   # size of position-context observation block

    def __init__(
        self,
        primary_df: pd.DataFrame,
        secondary_dfs: Dict[str, pd.DataFrame],   # {"5min": df, "1H": df, "4H": df}
        feature_columns: List[str],
        initial_balance: float   = 5_000.0,
        lot_size: float          = 0.01,
        contract_size: float     = 10.0,
        sl_atr_buffer: float     = 0.5,
        min_rr: float            = 1.5,
        episode_length: int      = 2000,
        # Reward weights
        skip_loser_reward: float   = 2.0,
        skip_winner_penalty: float = 1.5,
        rotation_bonus: float      = 1.5,
        stall_penalty: float       = 0.08,
        htf_penalty: float         = 1.0,
        drawdown_penalty: float    = 2.0,
        rr_bonus_factor: float     = 0.3,
    ):
        super().__init__()

        self.primary_df      = primary_df.reset_index(drop=True)
        self.secondary_dfs   = {tf: df.reset_index(drop=True) for tf, df in secondary_dfs.items()}
        self.feature_columns = feature_columns
        self.secondary_tfs   = sorted(secondary_dfs.keys())

        self.initial_balance  = initial_balance
        self.lot_size         = lot_size
        self.contract_size    = contract_size
        self.sl_atr_buffer    = sl_atr_buffer
        self.min_rr           = min_rr
        self.episode_length   = episode_length

        self.skip_loser_reward   = skip_loser_reward
        self.skip_winner_penalty = skip_winner_penalty
        self.rotation_bonus      = rotation_bonus
        self.stall_penalty       = stall_penalty
        self.htf_penalty         = htf_penalty
        self.drawdown_penalty    = drawdown_penalty
        self.rr_bonus_factor     = rr_bonus_factor

        self._aligned = self._precompute_alignment()

        n_tfs = 1 + len(self.secondary_tfs)
        n_obs = n_tfs * len(feature_columns) + self.N_ML_CTX + self.N_POS_CTX
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)   # 0=AGREE, 1=SKIP, 2=ROTATE

        self._reset_state(0)

    # ── Timestamp alignment ───────────────────────────────────────────────────

    def _precompute_alignment(self) -> Dict[str, np.ndarray]:
        primary_ts = pd.to_datetime(self.primary_df["timestamp"]).values.astype(np.int64)
        aligned: Dict[str, np.ndarray] = {}
        for tf, df in self.secondary_dfs.items():
            sec_ts  = pd.to_datetime(df["timestamp"]).values.astype(np.int64)
            indices = np.searchsorted(sec_ts, primary_ts, side="right") - 1
            aligned[tf] = np.clip(indices, 0, len(df) - 1)
        return aligned

    # ── State reset ──────────────────────────────────────────────────────────

    def _reset_state(self, start: int) -> None:
        self.current_step  = start
        self.episode_start = start
        self.balance       = self.initial_balance
        self.equity        = self.initial_balance
        self.peak_equity   = self.initial_balance
        self.max_drawdown  = 0.0

        # RL virtual position
        self.position      = 0       # -1 / 0 / +1
        self.entry_price   = 0.0
        self.entry_rr      = 0.0
        self.sl            = 0.0
        self.tp            = 0.0
        self.bars_in_trade = 0
        self.stall_count   = 0
        self.max_favorable = 0.0
        self._hi_in_trade  = 0.0
        self._lo_in_trade  = float("inf")
        self.trade_quality = 0.0

        # ML shadow position (counter-factual — always tracks what ML would do)
        self._ml_pos   = 0
        self._ml_entry = 0.0
        self._ml_sl    = 0.0
        self._ml_tp    = 0.0
        self._ml_rr    = 0.0
        self._ml_bars  = 0

        # Skip evaluation queue: each entry is a dict with pos/entry/sl/tp/rr/age
        self._pending_skips: List[Dict[str, Any]] = []

        self.trade_history: List[Dict[str, Any]] = []

    # ── Observation builder ──────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        fc    = self.feature_columns
        row   = self.primary_df.iloc[self.current_step]
        close = float(row.get("close", 0) or 0)

        # TF feature blocks
        parts: List[np.ndarray] = [row[fc].fillna(0).values.astype(np.float32)]
        for tf in self.secondary_tfs:
            idx     = self._aligned[tf][self.current_step]
            sec_row = self.secondary_dfs[tf].iloc[idx]
            parts.append(sec_row[fc].fillna(0).values.astype(np.float32))

        # ML context block (N_ML_CTX = 8)
        ml_sig  = float(row.get("ml_signal", 0) or 0)
        ml_prob = float(row.get("ml_prob",   0) or 0)
        ml_gr   = {"A": 1.0, "B": 0.67, "C": 0.33}.get(
            str(row.get("ml_grade", "") or "").strip().upper(), 0.0
        )
        ml_vpnl = self._ml_vpnl(close) / max(self.initial_balance, 1)
        htf4h   = float(row.get("htf_4h_bias",       0) or 0)
        atr_r   = float(row.get("macro_atr_ratio", 1.0) or 1.0)
        zone_q  = float(row.get("active_zone_quality", 0) or 0) / 5.0
        vol_r   = float(row.get("volume_ratio", 1.0) or 1.0)

        ml_ctx = np.array([
            np.clip(ml_sig,          -1,  1),
            np.clip(ml_prob,          0,  1),
            ml_gr,
            np.clip(ml_vpnl,         -1,  1),
            np.clip(htf4h,           -1,  1),
            np.clip(atr_r - 1.0,     -1,  2),   # centred around 0
            np.clip(zone_q,           0,  1),
            np.clip(vol_r - 1.0,     -1,  2),
        ], dtype=np.float32)
        parts.append(ml_ctx)

        # Position context block (N_POS_CTX = 7)
        upnl  = self._unrealized_pnl(close) / max(self.initial_balance, 1)
        risk  = abs(close - self.sl) if (self.position != 0 and self.sl != 0) else 1.0
        mfe_r = self.max_favorable / max(risk, 1e-6) if self.position != 0 else 0.0
        mom5  = float(row.get("momentum_5", 0) or 0)
        mom_n = float(np.clip(mom5 * self.position, -1, 1)) if self.position != 0 else 0.0
        still = float(self.stall_count < STALL_BARS and self.position != 0)

        pos_ctx = np.array([
            float(self.position),
            np.clip(upnl,    -1, 1),
            min(self.bars_in_trade / 50.0, 1.0),
            mom_n,
            still,
            np.clip(mfe_r,    0, 3),
            np.clip(self.trade_quality / 5.0, 0, 1),
        ], dtype=np.float32)
        parts.append(pos_ctx)

        return np.concatenate(parts)

    # ── PnL helpers ──────────────────────────────────────────────────────────

    def _unrealized_pnl(self, price: float) -> float:
        if self.position == 0:
            return 0.0
        return (price - self.entry_price) * self.position * self.lot_size * self.contract_size

    def _ml_vpnl(self, price: float) -> float:
        if self._ml_pos == 0:
            return 0.0
        return (price - self._ml_entry) * self._ml_pos * self.lot_size * self.contract_size

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

    # ── SL/TP hit detection ──────────────────────────────────────────────────

    def _check_hit(self, pos: int, sl: float, tp: float, row: pd.Series) -> Tuple[bool, bool]:
        if pos == 0:
            return False, False
        hi = float(row.get("high", 0) or 0)
        lo = float(row.get("low",  0) or 0)
        return (lo <= sl, hi >= tp) if pos == 1 else (hi >= sl, lo <= tp)

    # ── Stall / MFE tracking ─────────────────────────────────────────────────

    def _update_trade_progress(self, row: pd.Series) -> None:
        if self.position == 0:
            return
        hi = float(row.get("high", 0) or 0)
        lo = float(row.get("low",  0) or 0)

        if self.position == 1:
            fav = hi - self.entry_price
            if hi > self._hi_in_trade:
                self._hi_in_trade = hi
                self.stall_count  = 0
            else:
                self.stall_count += 1
        else:
            fav = self.entry_price - lo
            if lo < self._lo_in_trade:
                self._lo_in_trade = lo
                self.stall_count  = 0
            else:
                self.stall_count += 1

        self.max_favorable = max(self.max_favorable, max(float(fav), 0.0))

    # ── Counter-factual skip evaluation ─────────────────────────────────────

    def _evaluate_pending_skips(self, row: pd.Series) -> float:
        reward        = 0.0
        still_pending = []
        for skip in self._pending_skips:
            skip["age"] += 1
            sl_hit, tp_hit = self._check_hit(skip["pos"], skip["sl"], skip["tp"], row)

            if tp_hit:
                # Skipped a winner — penalise proportional to missed profit
                missed = abs(skip["tp"] - skip["entry"]) \
                         * abs(skip["pos"]) * self.lot_size * self.contract_size
                reward -= self.skip_winner_penalty * (missed / self.initial_balance * 100)
            elif sl_hit:
                # Skipped a loser — reward proportional to avoided loss
                saved = abs(skip["sl"] - skip["entry"]) \
                        * abs(skip["pos"]) * self.lot_size * self.contract_size
                reward += self.skip_loser_reward * (saved / self.initial_balance * 100)
            elif skip["age"] < SKIP_EVAL_TIMEOUT:
                still_pending.append(skip)  # still live, keep tracking

        self._pending_skips = still_pending
        return reward

    # ── Position open/close helpers ──────────────────────────────────────────

    def _open_rl(self, row: pd.Series, direction: int) -> bool:
        close = float(row.get("close", 0) or 0)
        sl, tp, rr = self._compute_sl_tp(row, direction)
        if sl is None:
            return False
        self.position      = direction
        self.entry_price   = close
        self.sl            = sl
        self.tp            = tp
        self.entry_rr      = rr
        self.bars_in_trade = 0
        self.max_favorable = 0.0
        self.stall_count   = 0
        self._hi_in_trade  = float(row.get("high", close) or close)
        self._lo_in_trade  = float(row.get("low",  close) or close)
        self.trade_quality = float(row.get("active_zone_quality", 0) or 0)
        return True

    def _close_rl(self, exit_price: float) -> float:
        pnl = (exit_price - self.entry_price) * self.position \
              * self.lot_size * self.contract_size
        self.balance       += pnl
        self.position       = 0
        self.bars_in_trade  = 0
        self.entry_rr       = 0.0
        self.max_favorable  = 0.0
        self.stall_count    = 0
        return pnl

    def _open_ml_shadow(self, direction: int, row: pd.Series) -> None:
        if self._ml_pos != 0:
            return
        close = float(row.get("close", 0) or 0)
        sl, tp, rr = self._compute_sl_tp(row, direction)
        if sl is None:
            return
        self._ml_pos   = direction
        self._ml_entry = close
        self._ml_sl    = sl
        self._ml_tp    = tp
        self._ml_rr    = rr or 0.0
        self._ml_bars  = 0

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        max_start = max(0, len(self.primary_df) - self.episode_length - 1)
        start     = int(self.np_random.integers(0, max_start + 1)) if max_start > 0 else 0
        self._reset_state(start)
        return self._get_obs(), {}

    def step(self, action: int):  # noqa: C901  (intentionally wide)
        row   = self.primary_df.iloc[self.current_step]
        close = float(row.get("close", 0) or 0)
        reward = 0.0

        # ── 1. Counter-factual skip evaluations ───────────────────────────────
        reward += self._evaluate_pending_skips(row)

        # ── 2. Resolve RL SL/TP ───────────────────────────────────────────────
        sl_hit, tp_hit = self._check_hit(self.position, self.sl, self.tp, row)
        if sl_hit or tp_hit:
            exit_price = self.sl if sl_hit else self.tp
            pnl        = self._close_rl(exit_price)
            pnl_norm   = pnl / self.initial_balance * 100
            if tp_hit:
                rr_bonus = self.rr_bonus_factor * max(self.entry_rr - self.min_rr, 0.0)
                reward  += pnl_norm * (1.0 + rr_bonus)
            else:
                reward  += pnl_norm
            self.trade_history.append({"pnl": pnl, "exit": "sl" if sl_hit else "tp"})

        # ── 3. Update stall counter / MFE ────────────────────────────────────
        self._update_trade_progress(row)

        # ── 4. Update ML shadow position ──────────────────────────────────────
        if self._ml_pos != 0:
            self._ml_bars += 1
            ml_sl_h, ml_tp_h = self._check_hit(self._ml_pos, self._ml_sl, self._ml_tp, row)
            if ml_sl_h or ml_tp_h:
                self._ml_pos  = 0
                self._ml_bars = 0

        # ── 5. Read bar ML signal ─────────────────────────────────────────────
        ml_signal = int(float(row.get("ml_signal", 0) or 0))
        ml_grade  = str(row.get("ml_grade", "") or "")
        htf4h     = float(row.get("htf_4h_bias", 0) or 0)
        atr_ratio = float(row.get("macro_atr_ratio", 1.0) or 1.0)
        zone_q    = float(row.get("active_zone_quality", 0) or 0)

        # Pre-compute SL/TP for the current bar ML signal (used in skip queue)
        ml_sl_bar = ml_tp_bar = ml_rr_bar = None
        if ml_signal != 0:
            ml_sl_bar, ml_tp_bar, ml_rr_bar = self._compute_sl_tp(row, ml_signal)

        counter_htf = (ml_signal == 1 and htf4h < -0.5) or (ml_signal == -1 and htf4h > 0.5)
        stalling    = self.stall_count >= STALL_BARS or self.bars_in_trade >= 25

        # ── 6. Agent action ───────────────────────────────────────────────────
        if ml_signal != 0 and ml_sl_bar is not None:

            if action == 0:   # AGREE — open if flat, stay if already in same direction
                if self.position == 0:
                    if self._open_rl(row, ml_signal):
                        self._open_ml_shadow(ml_signal, row)
                        if counter_htf:
                            reward -= self.htf_penalty

                elif self.position == ml_signal:
                    pass   # already in same direction — hold

                # Opposite position open: skip (can't flip without rotation)

            elif action == 1:   # SKIP — queue counter-factual evaluation
                self._pending_skips.append({
                    "pos":   ml_signal,
                    "entry": close,
                    "sl":    ml_sl_bar,
                    "tp":    ml_tp_bar,
                    "rr":    ml_rr_bar or 0.0,
                    "age":   0,
                })
                self._open_ml_shadow(ml_signal, row)  # always track ML shadow

                # Conservative TP: if already in a profitable trade and momentum is low,
                # exiting now (SKIP) is treated as a conservative partial exit
                if self.position != 0 and self.position == ml_signal:
                    upnl = self._unrealized_pnl(close)
                    low_momentum = atr_ratio < LOW_MOMENTUM_ATR_RATIO
                    if upnl > 0 and low_momentum:
                        pnl = self._close_rl(close)
                        pnl_norm = pnl / self.initial_balance * 100
                        reward += pnl_norm * 0.7   # conservative exit: 70% credit
                        self.trade_history.append({"pnl": pnl, "exit": "conservative_tp"})

            elif action == 2:   # ROTATE — exit stalling trade, enter new setup
                if self.position != 0:
                    if stalling and self.position != ml_signal:
                        pnl = self._close_rl(close)
                        pnl_norm = pnl / self.initial_balance * 100
                        reward += pnl_norm
                        self.trade_history.append({"pnl": pnl, "exit": "rotate"})
                        # Enter new setup
                        if self._open_rl(row, ml_signal):
                            self._open_ml_shadow(ml_signal, row)
                            reward += self.rotation_bonus * 0.5   # immediate partial bonus
                        if counter_htf:
                            reward -= self.htf_penalty
                    else:
                        reward -= 0.3   # premature rotation — punish
                else:
                    # Flat + ROTATE = same as AGREE
                    if self._open_rl(row, ml_signal):
                        self._open_ml_shadow(ml_signal, row)
                        if counter_htf:
                            reward -= self.htf_penalty

        else:
            # No ML signal this bar
            if action == 1 and self.position != 0:
                # SKIP with no new signal = early exit; only reward if low momentum + profit
                upnl = self._unrealized_pnl(close)
                low_momentum = atr_ratio < LOW_MOMENTUM_ATR_RATIO
                if upnl > 0 and low_momentum:
                    pnl = self._close_rl(close)
                    pnl_norm = pnl / self.initial_balance * 100
                    reward += pnl_norm * 0.7
                    self.trade_history.append({"pnl": pnl, "exit": "conservative_tp"})
                elif upnl <= 0:
                    reward -= 0.1  # exiting a losing trade with no replacement

            if action == 2 and self.position != 0 and not stalling:
                reward -= 0.15   # trying to rotate with no new signal and trade healthy

        # ── 7. Stall accumulating penalty ────────────────────────────────────
        if self.position != 0 and self.stall_count >= STALL_BARS:
            reward -= self.stall_penalty

        # ── 8. Advance bar counters ───────────────────────────────────────────
        if self.position != 0:
            self.bars_in_trade += 1
        if self._ml_pos != 0:
            self._ml_bars += 1

        # ── 9. Equity and drawdown tracking ──────────────────────────────────
        self.equity = self.balance + self._unrealized_pnl(close)
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        dd = (self.peak_equity - self.equity) / max(self.peak_equity, 1e-9)
        self.max_drawdown = max(self.max_drawdown, dd)

        if dd > 0.05:
            reward -= self.drawdown_penalty * (dd - 0.05) * 100

        # ── 10. Advance step ──────────────────────────────────────────────────
        self.current_step += 1
        steps      = self.current_step - self.episode_start
        terminated = self.current_step >= len(self.primary_df) - 1
        truncated  = steps >= self.episode_length or self.equity < self.initial_balance * 0.5

        # ── 11. Episode-end bonuses/penalties ────────────────────────────────
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
            "ml_position":  self._ml_pos,
            "stall_count":  self.stall_count,
            "max_drawdown": self.max_drawdown,
            "n_trades":     len(self.trade_history),
            "action":       action,
            "ml_signal":    ml_signal,
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self) -> None:
        row = self.primary_df.iloc[self.current_step]
        print(
            f"step={self.current_step} pos={self.position:+d} "
            f"eq={self.equity:.2f} dd={self.max_drawdown:.2%} "
            f"stall={self.stall_count} ml_pos={self._ml_pos:+d}"
        )

    # ── Summary ───────────────────────────────────────────────────────────────

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
            by_exit[t.get("exit", "?")] = by_exit.get(t.get("exit", "?"), 0) + 1
        return {
            "total_pnl":       round(total, 2),
            "n_trades":        n,
            "win_rate":        round(wr, 3),
            "profit_factor":   round(pf, 2),
            "max_drawdown":    round(self.max_drawdown, 4),
            "final_equity":    round(self.equity, 2),
            "exits_by_type":   by_exit,
        }
