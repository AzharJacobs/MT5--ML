"""
rl/train_rl.py — Train a PPO agent on historical XAUUSD 15min data.

Completely isolated from the existing ML system — no existing files are modified.

Usage:
  # First run: pulls data from DB, builds features, caches to CSV, trains
  python rl/train_rl.py --timeframe 15min --start-date 2024-01-01 --end-date 2026-02-28

  # Subsequent runs: use cached CSV (no DB needed)
  python rl/train_rl.py --from-cache

  # Resume training from a saved checkpoint
  python rl/train_rl.py --from-cache --resume

Output:
  rl/models/best_model/   — best checkpoint (by eval reward)
  rl/models/final_model   — model at end of training
  rl/data/features_15min.csv  — cached feature data
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.pipeline_config import REQUIRED_FEATURE_COLUMNS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("rl.train")

# ── Paths ─────────────────────────────────────────────────────────────────────
RL_DIR         = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR     = os.path.join(RL_DIR, "models")
DATA_DIR       = os.path.join(RL_DIR, "data")
CACHE_FILE     = os.path.join(DATA_DIR, "features_15min.csv")
BEST_MODEL_DIR = os.path.join(MODELS_DIR, "best_model")
FINAL_MODEL    = os.path.join(MODELS_DIR, "final_model")

# ── Training config ────────────────────────────────────────────────────────────
TRAIN_SPLIT    = 0.80          # 80% train, 20% test
EPISODE_LENGTH = 500           # bars per training episode
TOTAL_STEPS    = 500_000       # total PPO timesteps
EVAL_FREQ      = 10_000        # evaluate every N steps
N_EVAL_EPS     = 5             # episodes per evaluation
INITIAL_BALANCE = 5_000.0      # sim account size (comparable to backtest)


# ─────────────────────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_from_db(timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Pull raw OHLCV from PostgreSQL and run build_features()."""
    from data.loader import get_connection
    from data.feature_engineer import build_features

    logger.info("Connecting to DB...")
    db = get_connection()
    if not db.connect():
        raise ConnectionError("DB connection failed. Check .env credentials.")

    query = """
        SELECT timestamp, open, high, low, close, volume,
               hour, day_of_week, month, year, session,
               candle_size, body_size, wick_upper, wick_lower
        FROM xauusd_ohlcv
        WHERE symbol = 'XAUUSDm'
          AND timeframe = %s
          AND timestamp >= %s
          AND timestamp <= %s
        ORDER BY timestamp ASC
    """
    logger.info("Fetching %s bars from %s to %s...", timeframe, start_date, end_date)
    df = db.fetch_dataframe(query, (timeframe, start_date, end_date))
    db.disconnect()

    if df.empty:
        raise ValueError(f"No data returned for {timeframe} between {start_date} and {end_date}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    logger.info("Fetched %d raw bars", len(df))

    logger.info("Building features...")
    tf_params = {"15min": {"impulse_atr_multiplier": 0.5, "include_london_ny": False}}
    params = tf_params.get(timeframe, {})
    feat_df = build_features(
        df,
        impulse_atr_multiplier=params.get("impulse_atr_multiplier", 0.5),
        include_london_ny=params.get("include_london_ny", True),
    )
    logger.info("Features built — %d bars, %d columns", len(feat_df), len(feat_df.columns))
    return feat_df


def load_from_cache() -> pd.DataFrame:
    if not os.path.exists(CACHE_FILE):
        raise FileNotFoundError(
            f"Cache not found at {CACHE_FILE}. "
            "Run without --from-cache first to build it from the DB."
        )
    logger.info("Loading features from cache: %s", CACHE_FILE)
    df = pd.read_csv(CACHE_FILE, parse_dates=["timestamp"])
    logger.info("Loaded %d bars from cache", len(df))
    return df


def save_cache(df: pd.DataFrame):
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(CACHE_FILE, index=False)
    logger.info("Feature data cached to %s", CACHE_FILE)


# ─────────────────────────────────────────────────────────────────────────────
#  Train / eval split
# ─────────────────────────────────────────────────────────────────────────────

def split_data(df: pd.DataFrame):
    split_idx = int(len(df) * TRAIN_SPLIT)
    train_df  = df.iloc[:split_idx].reset_index(drop=True)
    test_df   = df.iloc[split_idx:].reset_index(drop=True)
    logger.info(
        "Split: train=%d bars (%s → %s), test=%d bars (%s → %s)",
        len(train_df), train_df["timestamp"].iloc[0].date(), train_df["timestamp"].iloc[-1].date(),
        len(test_df),  test_df["timestamp"].iloc[0].date(),  test_df["timestamp"].iloc[-1].date(),
    )
    return train_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
#  PPO training
# ─────────────────────────────────────────────────────────────────────────────

def train(train_df: pd.DataFrame, test_df: pd.DataFrame, resume: bool = False):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from rl.environment import XAUUSDTradingEnv

    os.makedirs(MODELS_DIR, exist_ok=True)

    # Validate feature columns present in data
    missing = [c for c in REQUIRED_FEATURE_COLUMNS if c not in train_df.columns]
    if missing:
        raise ValueError(f"Training data missing required columns: {missing}")

    # ── Environments ─────────────────────────────────────────────────────────
    def make_train_env():
        env = XAUUSDTradingEnv(
            feature_df      = train_df,
            feature_columns = REQUIRED_FEATURE_COLUMNS,
            initial_balance = INITIAL_BALANCE,
            episode_length  = EPISODE_LENGTH,
        )
        return Monitor(env)

    def make_eval_env():
        env = XAUUSDTradingEnv(
            feature_df      = test_df,
            feature_columns = REQUIRED_FEATURE_COLUMNS,
            initial_balance = INITIAL_BALANCE,
            episode_length  = EPISODE_LENGTH,
        )
        return Monitor(env)

    train_env = DummyVecEnv([make_train_env])
    eval_env  = DummyVecEnv([make_eval_env])

    # ── Callbacks ─────────────────────────────────────────────────────────────
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = BEST_MODEL_DIR,
        log_path             = os.path.join(MODELS_DIR, "eval_logs"),
        eval_freq            = EVAL_FREQ,
        n_eval_episodes      = N_EVAL_EPS,
        deterministic        = True,
        verbose              = 1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq  = 50_000,
        save_path  = os.path.join(MODELS_DIR, "checkpoints"),
        name_prefix= "ppo_xauusd",
        verbose    = 1,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    if resume and os.path.exists(FINAL_MODEL + ".zip"):
        logger.info("Resuming from %s", FINAL_MODEL)
        model = PPO.load(FINAL_MODEL, env=train_env)
    else:
        logger.info("Initialising new PPO agent")
        model = PPO(
            policy             = "MlpPolicy",
            env                = train_env,
            learning_rate      = 3e-4,
            n_steps            = 2048,
            batch_size         = 64,
            n_epochs           = 10,
            gamma              = 0.99,
            gae_lambda         = 0.95,
            clip_range         = 0.2,
            ent_coef           = 0.01,   # encourages exploration
            vf_coef            = 0.5,
            max_grad_norm      = 0.5,
            verbose            = 1,
            tensorboard_log    = os.path.join(MODELS_DIR, "tensorboard"),
        )

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("Training PPO for %d steps...", TOTAL_STEPS)
    start_time = datetime.now()

    model.learn(
        total_timesteps = TOTAL_STEPS,
        callback        = [eval_callback, checkpoint_callback],
        reset_num_timesteps = not resume,
        progress_bar    = True,
    )

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info("Training complete in %.1f minutes", elapsed / 60)

    model.save(FINAL_MODEL)
    logger.info("Final model saved to %s", FINAL_MODEL)

    return model


# ─────────────────────────────────────────────────────────────────────────────
#  Quick evaluation pass on held-out test data
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, test_df: pd.DataFrame):
    from rl.environment import XAUUSDTradingEnv

    logger.info("\n── Test set evaluation ──")
    env = XAUUSDTradingEnv(
        feature_df      = test_df,
        feature_columns = REQUIRED_FEATURE_COLUMNS,
        initial_balance = INITIAL_BALANCE,
        episode_length  = len(test_df),   # run full test period in one episode
    )

    obs, _ = env.reset(seed=42)
    done   = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated

    stats = env.summary()
    print("\n" + "=" * 50)
    print("RL AGENT — TEST SET RESULTS")
    print("=" * 50)
    for k, v in stats.items():
        print(f"  {k:<22} {v}")
    print("=" * 50)

    # Compare to reference ML backtest stats
    print("\nML reference (Jan–Apr 2026 backtest):")
    print("  win_rate               0.692")
    print("  profit_factor          2.49")
    print("  max_drawdown           ~0.088  ($44 on $5000 sim)")
    print("=" * 50)
    return stats


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train PPO RL agent on XAUUSD 15min")
    parser.add_argument("--timeframe",   default="15min")
    parser.add_argument("--start-date",  default="2024-01-01")
    parser.add_argument("--end-date",    default="2026-02-28")
    parser.add_argument("--from-cache",  action="store_true",
                        help="Skip DB fetch — use cached rl/data/features_15min.csv")
    parser.add_argument("--resume",      action="store_true",
                        help="Resume training from rl/models/final_model.zip")
    parser.add_argument("--steps",       type=int, default=TOTAL_STEPS,
                        help=f"PPO timesteps to train (default {TOTAL_STEPS})")
    args = parser.parse_args()

    global TOTAL_STEPS
    TOTAL_STEPS = args.steps

    # ── Load data ─────────────────────────────────────────────────────────────
    if args.from_cache:
        feat_df = load_from_cache()
    else:
        feat_df = load_from_db(args.timeframe, args.start_date, args.end_date)
        save_cache(feat_df)

    train_df, test_df = split_data(feat_df)

    # ── Train ─────────────────────────────────────────────────────────────────
    model = train(train_df, test_df, resume=args.resume)

    # ── Evaluate on held-out test set ─────────────────────────────────────────
    evaluate(model, test_df)


if __name__ == "__main__":
    main()
