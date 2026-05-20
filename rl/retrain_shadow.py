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
    Derive correct_action from resolved outcome data already in rl_shadow rows.

    rl_was_correct (bool from DB) tells us if the agent was right.
    We also re-derive correct_action for replay training:
      rl_hypothetical_pnl > 0  → market went in RL direction (agree was right = 0)
      rl_hypothetical_pnl < 0  → market went against RL     (skip was right  = 1)
      stalled (pnl ≈ 0, long duration) → rotate was right   = 2
    """
    # Only rows with resolved outcomes
    resolved = shadow[shadow["rl_was_correct"].notna()].copy()
    if resolved.empty:
        logger.warning("No resolved outcome rows found in rl_shadow.")
        return resolved

    # Map rl_decision int (0=agree,1=skip,2=rotate) back to name for compatibility
    action_map = {0: "agree", 1: "skip", 2: "rotate"}
    resolved["rl_action_name"] = resolved["rl_decision"].map(action_map).fillna("agree")

    def _correct_action(row) -> int:
        pnl  = row.get("rl_hypothetical_pnl") or 0.0
        dur  = row.get("trade_duration") or 0
        if pnl > 0:
            return 0   # agree
        if pnl < 0:
            return 1   # skip
        if dur >= 20:
            return 2   # rotate
        return 1       # stalled short → skip

    resolved["correct_action"] = resolved.apply(_correct_action, axis=1)
    resolved["was_correct"]    = (
        resolved["rl_decision"] == resolved["correct_action"]
    )

    total   = len(resolved)
    correct = resolved["was_correct"].sum()
    logger.info(
        "Labeled dataset: %d decisions | accuracy=%.1f%% | "
        "correct_skip=%d  correct_agree=%d  correct_rotate=%d",
        total, 100 * correct / max(total, 1),
        (resolved["correct_action"] == 1).sum(),
        (resolved["correct_action"] == 0).sum(),
        (resolved["correct_action"] == 2).sum(),
    )

    wrong = resolved[~resolved["was_correct"]]
    if len(wrong) > 0:
        logger.info(
            "Wrong decisions breakdown: %s",
            wrong.groupby(["rl_action_name",
                           resolved.loc[wrong.index, "rl_hypothetical_pnl"].apply(
                               lambda x: "lost" if (x or 0) < 0 else "stalled"
                           )]).size().to_dict(),
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
                    "correct_action":  int(row["correct_action"]),
                }

        total_real = len(self._real_idx)
        logger.info("LiveReplayEnv: %d real outcome anchors injected into feature cache",
                    total_real)

        # Monkey-patch _evaluate_pending_skips to use real outcomes
        self._patch_env()

        # Expose gym interface
        self.observation_space = self._env.observation_space
        self.action_space      = self._env.action_space

    def _patch_env(self):
        """Override the pending-skip evaluator to use real outcomes at known bars."""
        real_idx = self._real_idx
        inner    = self._env

        original_evaluate = inner._evaluate_pending_skips.__func__  # unbound

        def patched_evaluate(env_self, row: "pd.Series") -> float:
            """Use real outcome if available; fall back to price simulation."""
            current_ts = pd.to_datetime(
                env_self.primary_df.iloc[env_self.current_step].get("timestamp", "")
            )
            real = real_idx.get(current_ts)
            if real is None:
                return original_evaluate(env_self, row)

            # Inject real outcome into each pending skip
            reward = 0.0
            still_pending = []
            for p in env_self._pending_skips:
                p["age"] = p.get("age", 0) + 1
                market_did = real["market_did"]
                if market_did == "won":
                    # We skipped a winner — penalise
                    reward -= env_self.skip_winner_penalty
                elif market_did == "lost":
                    # We skipped a loser — reward
                    reward += env_self.skip_loser_reward
                elif market_did == "stalled":
                    # Mild reward for avoiding a dead trade
                    reward += env_self.skip_loser_reward * 0.4
                # resolved — don't keep in queue
            env_self._pending_skips = still_pending
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

    # Derive market direction from rl_hypothetical_pnl
    pnl_col = labeled.get("rl_hypothetical_pnl", pd.Series(dtype=float)) \
        if "rl_hypothetical_pnl" not in labeled.columns else labeled["rl_hypothetical_pnl"]
    won     = (pnl_col > 0).sum()
    lost    = (pnl_col < 0).sum()
    stalled = ((pnl_col == 0) | pnl_col.isna()).sum()

    skipped_losers  = ((labeled["rl_action_name"] == "skip")  & (pnl_col < 0)).sum()
    skipped_winners = ((labeled["rl_action_name"] == "skip")  & (pnl_col > 0)).sum()
    agreed_winners  = ((labeled["rl_action_name"] == "agree") & (pnl_col > 0)).sum()
    agreed_losers   = ((labeled["rl_action_name"] == "agree") & (pnl_col < 0)).sum()

    print("\n" + "=" * 60)
    print("SHADOW AGENT — LIVE OUTCOME ANALYSIS")
    print("=" * 60)
    print(f"  Total resolved decisions : {total}")
    print(f"  Market won               : {won}  ({100*won/max(total,1):.1f}%)")
    print(f"  Market lost              : {lost}  ({100*lost/max(total,1):.1f}%)")
    print(f"  Market stalled           : {stalled}")
    print(f"  Overall accuracy         : {100*correct/max(total,1):.1f}%")
    print("─" * 60)
    print("  Good decisions:")
    print(f"    Skipped losers         : {skipped_losers}")
    print(f"    Agreed on winners      : {agreed_winners}")
    print("  Bad decisions:")
    print(f"    Skipped winners        : {skipped_winners}")
    print(f"    Agreed on losers       : {agreed_losers}")
    print("=" * 60)

    if total < 50:
        print(f"\n  NOTE: Only {total} outcomes — more live data = better fine-tuning.")
        print("  Consider running the live bot for more sessions before retraining.")
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
