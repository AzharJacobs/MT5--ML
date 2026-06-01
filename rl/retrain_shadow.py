"""
rl/retrain_shadow.py — Fine-tune the shadow model on real live outcomes.

The ML bot trades live. The shadow agent watches and logs decisions + outcomes.
This script reads those logs and makes the shadow agent smarter by fine-tuning
it on what the market *actually did* — not just what the simulator predicted.

Flow:
  1. Load all logs/rl_outcomes_*.csv   (what market actually did)
  2. Load all logs/rl_shadow_*.csv     (what the agent decided + context)
  3. Join → labeled dataset of (timestamp, action, real_outcome)
  4. Load feature cache (rl/data/features_*.csv)
  5. Build LiveReplayEnv — same as shadow env but real outcomes override
     the simulated skip evaluation for bars we have live data for
  6. Fine-tune the existing shadow model for --steps (default 200k)
  7. Save updated model; back up previous version

Run weekly or once 300+ resolved outcomes accumulate:
  python rl/retrain_shadow.py
  python rl/retrain_shadow.py --steps 100000
  python rl/retrain_shadow.py --min-outcomes 200   # skip if not enough data

The agent gets smarter with every week of live shadow data.
"""

from __future__ import annotations

import os
import sys
import shutil
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.pipeline_config import RL_FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("rl.retrain")

# ── Paths ──────────────────────────────────────────────────────────────────────
RL_DIR      = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(RL_DIR, "models")
DATA_DIR    = os.path.join(RL_DIR, "data")
SHADOW_DIR  = os.path.join(MODELS_DIR, "shadow_model")
FINAL_MODEL = os.path.join(MODELS_DIR, "final_shadow_model")

# ── Config ─────────────────────────────────────────────────────────────────────
TIMEFRAMES   = ["5min", "15min", "1H", "4H"]
PRIMARY_TF   = "15min"
TRAIN_SPLIT  = 0.80
EPISODE_LEN  = 2000
INIT_BALANCE = 5_000.0
DEFAULT_STEPS      = 200_000
MIN_OUTCOMES_DEFAULT = 50


# ─────────────────────────────────────────────────────────────────────────────
#  1. Load shadow + outcome data from rl_shadow DB table
# ─────────────────────────────────────────────────────────────────────────────

def _get_rl_engine():
    """SQLAlchemy engine for the RL/ML database (creator bypasses URL encoding of '/')."""
    import os, psycopg2
    from sqlalchemy import create_engine
    from dotenv import load_dotenv
    load_dotenv()
    host     = os.getenv("RL_DB_HOST",     os.getenv("DB_HOST",     "localhost"))
    port     = int(os.getenv("RL_DB_PORT", os.getenv("DB_PORT",     "5432")))
    database = os.getenv("RL_DB_NAME",     "RL/ML")
    user     = os.getenv("RL_DB_USER",     os.getenv("DB_USER",     "postgres"))
    password = os.getenv("RL_DB_PASSWORD", os.getenv("DB_PASSWORD", ""))

    def _creator():
        return psycopg2.connect(host=host, port=port, dbname=database,
                                user=user, password=password)

    return create_engine("postgresql+psycopg2://", creator=_creator, pool_pre_ping=True)


def load_shadow_data() -> pd.DataFrame:
    """
    Load all rows from rl_shadow table.
    Outcomes are resolved rows where rl_was_correct IS NOT NULL.
    """
    engine = _get_rl_engine()
    with engine.connect() as conn:
        df = pd.read_sql(
            "SELECT * FROM rl_shadow ORDER BY timestamp",
            conn,
        )
    logger.info("Loaded %d rl_shadow rows from DB", len(df))
    if df.empty:
        raise FileNotFoundError(
            "No rows in rl_shadow table.\n"
            "Run the live bot with --rl-shadow first to accumulate shadow data."
        )
    return df


def build_labeled_dataset(shadow: pd.DataFrame) -> pd.DataFrame:
    """
    Derive correct MultiDiscrete action [take, sl_mode, tp_mode] from resolved outcomes.

    action[0] take:    1 if trade was profitable, 0 if not
    action[1] sl_mode: 1=TIGHT if MAE < 40% of SL distance (clean zone),
                       2=WIDE  if MAE > 80% of SL distance (noisy zone),
                       0=ML_SL otherwise
    action[2] tp_mode: 2=EXTENDED if MFE > 1.8× TP distance (market ran hard),
                       1=CONSERVATIVE if MFE < 1.0× TP distance (market stalled early),
                       0=ML_TP otherwise
    """
    resolved = shadow[shadow["rl_was_correct"].notna()].copy()
    if resolved.empty:
        logger.warning("No resolved outcome rows found in rl_shadow.")
        return resolved

    resolved["rl_action_name"] = resolved["rl_decision"].map(
        {0: "skip", 1: "take"}
    ).fillna("skip")

    def _correct_actions(row) -> tuple:
        pnl     = float(row.get("rl_hypothetical_pnl") or 0.0)
        mfe     = float(row.get("max_favourable")       or 0.0)
        mae     = float(row.get("max_adverse")          or 0.0)
        ml_sl   = row.get("ml_sl_price")
        ml_tp   = row.get("ml_tp_price")
        entry   = row.get("ml_entry_price") or 0.0

        # action[0]: should we have taken this trade?
        take = 1 if pnl > 0 else 0

        # max_favourable / max_adverse in DB are in USD (lot_size=0.01, contract=10 → $0.10/pt)
        # ml_sl_price and ml_entry_price are raw price levels — convert to USD for comparison
        lot_usd = 0.01 * 10   # $0.10 per price point for XAUUSD 0.01 lots

        # action[1]: SL mode
        sl_dist_usd = abs(float(entry) - float(ml_sl)) * lot_usd if ml_sl else 0.0
        if sl_dist_usd > 0:
            mae_ratio = mae / sl_dist_usd
            if mae_ratio < 0.40:
                sl_mode = 1   # TIGHT — zone was clean, tight SL would have held
            elif mae_ratio > 0.80:
                sl_mode = 2   # WIDE — zone was noisy, wider SL needed
            else:
                sl_mode = 0
        else:
            sl_mode = 0

        # action[2]: TP mode
        tp_dist_usd = abs(float(ml_tp) - float(entry)) * lot_usd if ml_tp else 0.0
        if tp_dist_usd > 0:
            mfe_ratio = mfe / tp_dist_usd
            if mfe_ratio > 1.8:
                tp_mode = 2   # EXTENDED — market ran well past TP, should have aimed higher
            elif mfe_ratio < 1.0:
                tp_mode = 1   # CONSERVATIVE — market stalled before TP, take earlier
            else:
                tp_mode = 0
        else:
            tp_mode = 0

        return take, sl_mode, tp_mode

    results = resolved.apply(_correct_actions, axis=1, result_type="expand")
    resolved["correct_take"]    = results[0].astype(int)
    resolved["correct_sl_mode"] = results[1].astype(int)
    resolved["correct_tp_mode"] = results[2].astype(int)

    # was_correct: did the agent match on take/skip at minimum?
    resolved["was_correct"] = (resolved["rl_decision"] == resolved["correct_take"])

    total   = len(resolved)
    correct = resolved["was_correct"].sum()
    took    = (resolved["correct_take"] == 1).sum()
    skip    = (resolved["correct_take"] == 0).sum()
    tight   = (resolved["correct_sl_mode"] == 1).sum()
    wide    = (resolved["correct_sl_mode"] == 2).sum()
    cons    = (resolved["correct_tp_mode"] == 1).sum()
    ext     = (resolved["correct_tp_mode"] == 2).sum()

    logger.info(
        "Labeled dataset: %d decisions | take/skip accuracy=%.1f%% | "
        "correct_take=%d  correct_skip=%d | "
        "sl: tight=%d wide=%d ml=%d | tp: cons=%d ext=%d ml=%d",
        total, 100 * correct / max(total, 1),
        took, skip,
        tight, wide, total - tight - wide,
        cons, ext, total - cons - ext,
    )
    return resolved


# ─────────────────────────────────────────────────────────────────────────────
#  2. Feature cache
# ─────────────────────────────────────────────────────────────────────────────

def load_feature_cache() -> Dict[str, pd.DataFrame]:
    """Load rl/data/features_*.csv — must exist (built by train_shadow.py)."""
    tf_dfs = {}
    for tf in TIMEFRAMES:
        path = os.path.join(DATA_DIR, f"features_{tf}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Feature cache missing for {tf}: {path}\n"
                "Run: python rl/train_shadow.py --steps 50000  (builds cache as side-effect)"
            )
        df = pd.read_csv(path, parse_dates=["timestamp"])
        tf_dfs[tf] = df
        logger.info("Feature cache %-5s: %d bars", tf, len(df))
    return tf_dfs


# ─────────────────────────────────────────────────────────────────────────────
#  3. LiveReplayEnv — shadow env with real outcomes injected
# ─────────────────────────────────────────────────────────────────────────────

class LiveReplayEnv:
    """
    Wraps XAUUSDShadowEnv and injects real live outcomes at the bars where
    we have shadow log data.  For all other bars the normal simulator runs.

    Real outcomes override the pending-skip evaluation so the reward signal
    at those bars reflects what the market actually did — not what the
    simulator predicted from price levels.
    """

    def __init__(
        self,
        primary_df: pd.DataFrame,
        secondary_dfs: Dict[str, pd.DataFrame],
        feature_columns: list,
        real_outcomes: pd.DataFrame,          # labeled dataset (merged outcomes+shadow)
        initial_balance: float = INIT_BALANCE,
        episode_length: int    = EPISODE_LEN,
        # Boost weights so real-outcome signal is louder than simulated signal
        skip_loser_reward:   float = 3.0,     # higher than base 2.0
        skip_winner_penalty: float = 2.5,     # higher than base 1.5
    ):
        from rl.environment_shadow import XAUUSDShadowEnv

        self._env = XAUUSDShadowEnv(
            primary_df      = primary_df,
            secondary_dfs   = secondary_dfs,
            feature_columns = feature_columns,
            initial_balance = initial_balance,
            episode_length  = episode_length,
            skip_loser_reward   = skip_loser_reward,
            skip_winner_penalty = skip_winner_penalty,
        )

        # Index real outcomes by timestamp (15min bar) for fast lookup
        self._real_idx: Dict[pd.Timestamp, Dict[str, Any]] = {}
        primary_ts = pd.to_datetime(primary_df["timestamp"])

        for _, row in real_outcomes.iterrows():
            # rl_shadow table uses "timestamp" (not "decision_timestamp")
            ts = pd.to_datetime(row["timestamp"])
            # snap to nearest 15min bar in the feature cache
            diffs = (primary_ts - ts).abs()
            nearest_pos = diffs.idxmin()
            if diffs[nearest_pos].total_seconds() < 3600:  # within 1hr
                # Derive market_did from rl_hypothetical_pnl (no "market_did" column in DB)
                hyp_pnl = float(row.get("rl_hypothetical_pnl") or 0.0)
                if hyp_pnl > 0:
                    market_did = "won"
                elif hyp_pnl < 0:
                    market_did = "lost"
                else:
                    market_did = "stalled"
                self._real_idx[primary_ts[nearest_pos]] = {
                    "market_did":      market_did,
                    "pnl_usd":         hyp_pnl,
                    "bars_to_outcome": int(row.get("trade_duration") or 0),
                    # 3-element correct action for new MultiDiscrete space
                    "correct_take":    int(row.get("correct_take",    1 if hyp_pnl > 0 else 0)),
                    "correct_sl_mode": int(row.get("correct_sl_mode", 0)),
                    "correct_tp_mode": int(row.get("correct_tp_mode", 0)),
                }

        total_real = len(self._real_idx)
        logger.info("LiveReplayEnv: %d real outcome anchors injected into feature cache",
                    total_real)

        # Patch skip evaluator to use real outcomes where available
        self._patch_env()

        self.observation_space = self._env.observation_space
        self.action_space      = self._env.action_space

    def _patch_env(self):
        """Override pending-skip evaluator to use real outcomes at known bars."""
        real_idx = self._real_idx
        inner    = self._env

        original_evaluate = inner._evaluate_pending_skips.__func__

        def patched_evaluate(env_self, row: "pd.Series") -> float:
            current_ts = pd.to_datetime(
                env_self.primary_df.iloc[env_self.current_step].get("timestamp", "")
            )
            real = real_idx.get(current_ts)
            if real is None:
                return original_evaluate(env_self, row)

            reward = 0.0
            market_did = real["market_did"]
            for p in list(env_self._pending_skips):
                p["age"] = p.get("age", 0) + 1
                if market_did == "won":
                    reward -= env_self.skip_winner_penalty
                elif market_did == "lost":
                    reward += env_self.skip_loser_reward
                elif market_did == "stalled":
                    reward += env_self.skip_loser_reward * 0.4
            env_self._pending_skips = []   # all resolved
            return reward

        import types
        inner._evaluate_pending_skips = types.MethodType(patched_evaluate, inner)

    def reset(self, **kwargs):
        return self._env.reset(**kwargs)

    def step(self, action):
        return self._env.step(action)

    def render(self, *a, **kw):
        return self._env.render(*a, **kw)

    def close(self):
        return self._env.close()

    def summary(self):
        return self._env.summary()


# ─────────────────────────────────────────────────────────────────────────────
#  4. Load existing shadow model
# ─────────────────────────────────────────────────────────────────────────────

def load_model(env):
    """Load the best available shadow model for fine-tuning."""
    try:
        from sb3_contrib import RecurrentPPO
    except ImportError:
        raise ImportError("pip install sb3-contrib")

    candidates = [
        os.path.join(SHADOW_DIR, "best_model.zip"),
        FINAL_MODEL + ".zip",
    ]
    for path in candidates:
        if os.path.exists(path):
            logger.info("Loading model for fine-tuning: %s", path)
            model = RecurrentPPO.load(path, env=env)
            return model, path

    raise FileNotFoundError(
        "No shadow model found. Train one first:\n"
        "  python rl/train_shadow.py --steps 50000"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  5. Fine-tune and save
# ─────────────────────────────────────────────────────────────────────────────

def backup_model(source_path: str):
    """Rename existing model to model_backup_<date>.zip before overwriting."""
    if not os.path.exists(source_path):
        return
    stamp  = datetime.now().strftime("%Y%m%d_%H%M")
    backup = source_path.replace(".zip", f"_backup_{stamp}.zip")
    shutil.copy2(source_path, backup)
    logger.info("Backed up previous model → %s", os.path.basename(backup))


def retrain(
    train_dfs: Dict[str, pd.DataFrame],
    test_dfs:  Dict[str, pd.DataFrame],
    labeled:   pd.DataFrame,
    total_steps: int,
    no_eval: bool = False,
):
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.monitor import Monitor

    def _make_replay_env(tf_dfs, outcomes):
        def _factory():
            primary = tf_dfs[PRIMARY_TF]
            secondary = {tf: tf_dfs[tf] for tf in TIMEFRAMES if tf != PRIMARY_TF}
            env = LiveReplayEnv(
                primary_df      = primary,
                secondary_dfs   = secondary,
                feature_columns = RL_FEATURE_COLUMNS,
                real_outcomes   = outcomes,
                episode_length  = EPISODE_LEN,
            )
            return Monitor(env)
        return _factory

    train_env = DummyVecEnv([_make_replay_env(train_dfs, labeled)])

    model, source_path = load_model(train_env)

    callbacks = []
    if not no_eval:
        from rl.environment_shadow import XAUUSDShadowEnv
        from stable_baselines3.common.monitor import Monitor as M

        def _make_eval():
            primary   = test_dfs[PRIMARY_TF]
            secondary = {tf: test_dfs[tf] for tf in TIMEFRAMES if tf != PRIMARY_TF}
            env = XAUUSDShadowEnv(
                primary_df      = primary,
                secondary_dfs   = secondary,
                feature_columns = RL_FEATURE_COLUMNS,
                episode_length  = len(primary),
            )
            return M(env)

        eval_env = DummyVecEnv([_make_eval])
        eval_cb  = EvalCallback(
            eval_env,
            best_model_save_path = SHADOW_DIR,
            log_path             = os.path.join(MODELS_DIR, "eval_logs_retrain"),
            eval_freq            = max(total_steps // 10, 5_000),
            n_eval_episodes      = 3,
            deterministic        = True,
            verbose              = 1,
        )
        callbacks.append(eval_cb)

    logger.info("Fine-tuning for %d steps on live outcome data...", total_steps)
    start = datetime.now()
    model.learn(
        total_timesteps     = total_steps,
        callback            = callbacks if callbacks else None,
        reset_num_timesteps = False,   # continue from existing step count
        progress_bar        = True,
    )
    elapsed = (datetime.now() - start).total_seconds()
    logger.info("Fine-tuning complete in %.1f min", elapsed / 60)

    backup_model(FINAL_MODEL + ".zip")
    model.save(FINAL_MODEL)
    logger.info("Updated model saved → %s.zip", FINAL_MODEL)
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  6. Summary report
# ─────────────────────────────────────────────────────────────────────────────

def print_report(labeled: pd.DataFrame):
    if labeled.empty:
        print("\nNo resolved outcome data yet — run more live shadow sessions first.")
        return
    total   = len(labeled)
    correct = labeled["was_correct"].sum()
    pnl_col = labeled["rl_hypothetical_pnl"] if "rl_hypothetical_pnl" in labeled.columns \
              else pd.Series(dtype=float)
    won     = (pnl_col > 0).sum()
    lost    = (pnl_col < 0).sum()
    stalled = ((pnl_col == 0) | pnl_col.isna()).sum()

    skipped_losers  = ((labeled["rl_action_name"] == "skip") & (pnl_col < 0)).sum()
    skipped_winners = ((labeled["rl_action_name"] == "skip") & (pnl_col > 0)).sum()
    took_winners    = ((labeled["rl_action_name"] == "take") & (pnl_col > 0)).sum()
    took_losers     = ((labeled["rl_action_name"] == "take") & (pnl_col < 0)).sum()

    tight = (labeled.get("correct_sl_mode", pd.Series(dtype=int)) == 1).sum() \
            if "correct_sl_mode" in labeled.columns else 0
    wide  = (labeled.get("correct_sl_mode", pd.Series(dtype=int)) == 2).sum() \
            if "correct_sl_mode" in labeled.columns else 0
    cons  = (labeled.get("correct_tp_mode", pd.Series(dtype=int)) == 1).sum() \
            if "correct_tp_mode" in labeled.columns else 0
    ext   = (labeled.get("correct_tp_mode", pd.Series(dtype=int)) == 2).sum() \
            if "correct_tp_mode" in labeled.columns else 0

    print("\n" + "=" * 60)
    print("TRADE OPTIMIZER — LIVE OUTCOME ANALYSIS")
    print("=" * 60)
    print(f"  Total resolved decisions : {total}")
    print(f"  Market won               : {won}  ({100*won/max(total,1):.1f}%)")
    print(f"  Market lost              : {lost}  ({100*lost/max(total,1):.1f}%)")
    print(f"  Market stalled           : {stalled}")
    print(f"  Take/Skip accuracy       : {100*correct/max(total,1):.1f}%")
    print("─" * 60)
    print("  Take/Skip decisions:")
    print(f"    Took winners           : {took_winners}")
    print(f"    Took losers            : {took_losers}")
    print(f"    Skipped losers         : {skipped_losers}")
    print(f"    Skipped winners        : {skipped_winners}")
    print("─" * 60)
    print("  Optimal level labels (for training):")
    print(f"    SL: TIGHT={tight}  WIDE={wide}  ML={total-tight-wide}")
    print(f"    TP: CONS={cons}   EXT={ext}   ML={total-cons-ext}")
    print("=" * 60)

    if total < 50:
        print(f"\n  NOTE: Only {total} outcomes — need 50+ for retraining.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune shadow RL agent on real live outcome data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard weekly retrain (200k steps)
  python rl/retrain_shadow.py

  # Quick fine-tune after 2-3 days of shadow data
  python rl/retrain_shadow.py --steps 50000

  # Skip if not enough data yet
  python rl/retrain_shadow.py --min-outcomes 300

  # Just show the outcome analysis, don't retrain
  python rl/retrain_shadow.py --report-only
        """,
    )
    parser.add_argument("--steps",        type=int,  default=DEFAULT_STEPS)
    parser.add_argument("--min-outcomes", type=int,  default=MIN_OUTCOMES_DEFAULT,
                        help="Abort if fewer resolved outcomes than this")
    parser.add_argument("--no-eval",      action="store_true",
                        help="Skip EvalCallback (faster, no best-model tracking)")
    parser.add_argument("--report-only",  action="store_true",
                        help="Print outcome report then exit — no training")
    args = parser.parse_args()

    # ── 1. Load shadow data from DB ───────────────────────────────────────────
    shadow = load_shadow_data()

    # ── 2. Label from resolved outcomes ──────────────────────────────────────
    labeled = build_labeled_dataset(shadow)
    print_report(labeled)

    if len(labeled) < args.min_outcomes:
        logger.warning(
            "Only %d labeled outcomes (need %d). Run more live shadow sessions first.",
            len(labeled), args.min_outcomes,
        )
        return

    if args.report_only:
        return

    # ── 3. Feature cache ──────────────────────────────────────────────────────
    tf_dfs   = load_feature_cache()
    primary  = tf_dfs[PRIMARY_TF]
    split_idx = int(len(primary) * TRAIN_SPLIT)
    split_ts  = primary.iloc[split_idx]["timestamp"]

    train_dfs, test_dfs = {}, {}
    for tf, df in tf_dfs.items():
        train_dfs[tf] = df[df["timestamp"] < split_ts].reset_index(drop=True)
        test_dfs[tf]  = df[df["timestamp"] >= split_ts].reset_index(drop=True)
        logger.info("%-5s  train=%d  test=%d", tf, len(train_dfs[tf]), len(test_dfs[tf]))

    # ── 4. Fine-tune ──────────────────────────────────────────────────────────
    retrain(train_dfs, test_dfs, labeled, args.steps, args.no_eval)

    logger.info(
        "Done. Next step: restart live_trader.py with --rl-shadow "
        "to deploy the updated model."
    )


if __name__ == "__main__":
    main()
