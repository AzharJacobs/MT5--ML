"""
rl/train_rl_mtf.py — Train a RecurrentPPO (LSTM) agent on multi-timeframe XAUUSD data.

Architecture:
  - RecurrentPPO with LSTM policy (sb3-contrib)
  - Observation: 4 TFs × 66 features (51 ML + 15 macro) + 3 position context = 267 dims
  - The LSTM hidden state carries regime context across bars — this is what the
    ML bot lacks. It learns "we've been making lower highs for 2 weeks, don't buy."

Macro features added (data/macro_features.py):
  ADX, EMA regime, swing structure (HH/HL vs LH/LL), range position,
  volatility regime — all the context the ML bot bolted on as external filters.

Reward (fixed from original):
  - No per-step cost (was trapping the agent)
  - Symmetric SL/TP treatment (no double SL penalty)
  - Overtrading penalty at episode end
  - Calmar bonus for clean equity growth

Install sb3-contrib before running:
  pip install sb3-contrib

Usage:
  # First run — fetch from DB, build features, cache, train
  python rl/train_rl_mtf.py --start-date 2024-01-01 --end-date 2026-04-30

  # Subsequent runs — use cached CSVs
  python rl/train_rl_mtf.py --from-cache

  # Resume interrupted run
  python rl/train_rl_mtf.py --from-cache --resume

  # Quick smoke test
  python rl/train_rl_mtf.py --from-cache --steps 50000

Output:
  rl/models/best_model_mtf/    — best checkpoint (EvalCallback)
  rl/models/final_model_mtf    — final model at end of run
  rl/data/features_<TF>.csv    — cached feature CSVs
"""

import os
import sys
import argparse
import logging
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.pipeline_config import RL_FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("rl.train_mtf")

# ── Paths ──────────────────────────────────────────────────────────────────────
RL_DIR      = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(RL_DIR, "models")
DATA_DIR    = os.path.join(RL_DIR, "data")
BEST_DIR    = os.path.join(MODELS_DIR, "best_model_mtf")
FINAL_MODEL = os.path.join(MODELS_DIR, "final_model_mtf")

# ── Timeframes ─────────────────────────────────────────────────────────────────
TIMEFRAMES = ["5min", "15min", "1H", "4H"]
PRIMARY_TF = "15min"

TF_BUILD_PARAMS = {
    "5min":  {"impulse_atr_multiplier": 0.5, "include_london_ny": True},
    "15min": {"impulse_atr_multiplier": 0.3, "include_london_ny": False},
    "1H":    {"impulse_atr_multiplier": 0.5, "include_london_ny": True},
    "4H":    {"impulse_atr_multiplier": 0.5, "include_london_ny": True},
}

# ── Training hyperparameters ────────────────────────────────────────────────────
TRAIN_SPLIT     = 0.80
EPISODE_LENGTH  = 2000      # ~3 weeks at 15min — enough to see full trend cycles
TOTAL_STEPS     = 5_000_000
EVAL_FREQ       = 25_000
N_EVAL_EPS      = 5
INITIAL_BALANCE = 5_000.0


# ─────────────────────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_one_tf(timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
    from data.loader import get_connection
    from data.feature_engineer import build_features
    from data.macro_features import add_macro_features

    db = get_connection()
    if not db.connect():
        raise ConnectionError("PostgreSQL connection failed. Check .env credentials.")

    query = """
        SELECT timestamp, open, high, low, close, volume,
               hour, day_of_week, month, year, session,
               candle_size, body_size, wick_upper, wick_lower
        FROM xauusd_ohlcv
        WHERE symbol = 'XAUUSDm' AND timeframe = %s
          AND timestamp >= %s AND timestamp <= %s
        ORDER BY timestamp ASC
    """
    logger.info("Fetching %s from DB  (%s → %s)...", timeframe, start_date, end_date)
    df = db.fetch_dataframe(query, (timeframe, start_date, end_date))
    db.disconnect()

    if df.empty:
        raise ValueError(f"No data returned for {timeframe} between {start_date} and {end_date}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    params  = TF_BUILD_PARAMS.get(timeframe, {})
    feat_df = build_features(
        df,
        impulse_atr_multiplier=params.get("impulse_atr_multiplier", 0.5),
        include_london_ny=params.get("include_london_ny", True),
    )
    feat_df = add_macro_features(feat_df)

    missing = [c for c in RL_FEATURE_COLUMNS if c not in feat_df.columns]
    if missing:
        raise ValueError(f"{timeframe}: missing RL feature columns after build: {missing}")

    logger.info("%s: %d bars, %d features", timeframe, len(feat_df), feat_df.shape[1])
    return feat_df


def load_all_from_db(start_date: str, end_date: str) -> dict:
    return {tf: _load_one_tf(tf, start_date, end_date) for tf in TIMEFRAMES}


def save_cache(tf_dfs: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    for tf, df in tf_dfs.items():
        path = os.path.join(DATA_DIR, f"features_{tf}.csv")
        df.to_csv(path, index=False)
        logger.info("Cached %s → %s", tf, path)


def load_from_cache() -> dict:
    from data.macro_features import add_macro_features

    tf_dfs = {}
    for tf in TIMEFRAMES:
        path = os.path.join(DATA_DIR, f"features_{tf}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Cache missing for {tf}: {path}\n"
                "Run without --from-cache first to build it from the DB."
            )
        df = pd.read_csv(path, parse_dates=["timestamp"])

        # Add macro features if not already present (old cache from before this update)
        from config.pipeline_config import RL_MACRO_FEATURE_COLUMNS
        if any(c not in df.columns for c in RL_MACRO_FEATURE_COLUMNS):
            logger.info("%s cache is missing macro features — recomputing...", tf)
            df = add_macro_features(df)

        tf_dfs[tf] = df
        logger.info("Loaded %s from cache: %d bars", tf, len(tf_dfs[tf]))
    return tf_dfs


# ─────────────────────────────────────────────────────────────────────────────
#  Train / test split
# ─────────────────────────────────────────────────────────────────────────────

def split_data(tf_dfs: dict) -> tuple:
    primary   = tf_dfs[PRIMARY_TF]
    split_idx = int(len(primary) * TRAIN_SPLIT)
    split_ts  = primary.iloc[split_idx]["timestamp"]

    train_dfs, test_dfs = {}, {}
    for tf, df in tf_dfs.items():
        train_dfs[tf] = df[df["timestamp"] < split_ts].reset_index(drop=True)
        test_dfs[tf]  = df[df["timestamp"] >= split_ts].reset_index(drop=True)
        logger.info(
            "%-5s  train=%d bars  test=%d bars  split=%s",
            tf, len(train_dfs[tf]), len(test_dfs[tf]), split_ts.date(),
        )
    return train_dfs, test_dfs


# ─────────────────────────────────────────────────────────────────────────────
#  RecurrentPPO training
# ─────────────────────────────────────────────────────────────────────────────

def train(train_dfs: dict, test_dfs: dict, resume: bool = False) -> object:
    try:
        from sb3_contrib import RecurrentPPO
    except ImportError:
        raise ImportError(
            "sb3-contrib is required for RecurrentPPO.\n"
            "Install with: pip install sb3-contrib"
        )

    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from rl.environment_mtf import XAUUSDMultiTFEnv

    os.makedirs(MODELS_DIR, exist_ok=True)

    missing = [c for c in RL_FEATURE_COLUMNS if c not in train_dfs[PRIMARY_TF].columns]
    if missing:
        raise ValueError(f"Primary TF training data missing columns: {missing}")

    n_features = len(RL_FEATURE_COLUMNS)
    n_tfs      = len(TIMEFRAMES)
    obs_dim    = n_tfs * n_features + 3
    logger.info(
        "Obs space: %d TFs × %d features + 3 context = %d dims",
        n_tfs, n_features, obs_dim,
    )

    def _make_env(tf_dfs):
        def _factory():
            env = XAUUSDMultiTFEnv(
                tf_dfs          = tf_dfs,
                feature_columns = RL_FEATURE_COLUMNS,
                primary_tf      = PRIMARY_TF,
                initial_balance = INITIAL_BALANCE,
                episode_length  = EPISODE_LENGTH,
            )
            return Monitor(env)
        return _factory

    train_env = DummyVecEnv([_make_env(train_dfs)])
    eval_env  = DummyVecEnv([_make_env(test_dfs)])

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = BEST_DIR,
        log_path             = os.path.join(MODELS_DIR, "eval_logs_mtf"),
        eval_freq            = EVAL_FREQ,
        n_eval_episodes      = N_EVAL_EPS,
        deterministic        = True,
        verbose              = 1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq   = 250_000,
        save_path   = os.path.join(MODELS_DIR, "checkpoints_mtf"),
        name_prefix = "rppo_mtf_xauusd",
        verbose     = 1,
    )

    if resume and os.path.exists(FINAL_MODEL + ".zip"):
        logger.info("Resuming from %s", FINAL_MODEL)
        model = RecurrentPPO.load(FINAL_MODEL, env=train_env)
    else:
        logger.info("New RecurrentPPO agent | obs_dim=%d | tfs=%s", obs_dim, TIMEFRAMES)
        model = RecurrentPPO(
            policy        = "MlpLstmPolicy",
            env           = train_env,
            learning_rate = 5e-5,        # stable: was 2e-4 which caused KL=0.17
            n_steps       = 2048,
            batch_size    = 128,
            n_epochs      = 10,
            gamma         = 0.995,       # longer horizon: was 0.99 (too short for 2000-bar eps)
            gae_lambda    = 0.95,
            clip_range    = 0.1,         # tighter: was 0.2 which caused clip_fraction=0.39
            ent_coef      = 0.02,        # more exploration: was 0.005
            vf_coef       = 0.5,
            max_grad_norm = 0.5,
            policy_kwargs = {
                "lstm_hidden_size": 256,
                "n_lstm_layers":    2,
                "net_arch":         [256, 128],  # layers before/after LSTM
            },
            verbose         = 1,
            tensorboard_log = os.path.join(MODELS_DIR, "tensorboard_mtf"),
        )

    logger.info("Training RecurrentPPO for %d steps...", TOTAL_STEPS)
    start_time = datetime.now()
    model.learn(
        total_timesteps     = TOTAL_STEPS,
        callback            = [eval_callback, checkpoint_callback],
        reset_num_timesteps = not resume,
        progress_bar        = True,
    )
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info("Training complete in %.1f min", elapsed / 60)

    model.save(FINAL_MODEL)
    logger.info("Final model saved to %s.zip", FINAL_MODEL)
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation on held-out test data
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, test_dfs: dict):
    from rl.environment_mtf import XAUUSDMultiTFEnv

    logger.info("Running full test-set evaluation...")
    env = XAUUSDMultiTFEnv(
        tf_dfs          = test_dfs,
        feature_columns = RL_FEATURE_COLUMNS,
        primary_tf      = PRIMARY_TF,
        initial_balance = INITIAL_BALANCE,
        episode_length  = len(test_dfs[PRIMARY_TF]),
    )

    obs, _     = env.reset(seed=42)
    lstm_states = None
    done        = False
    episode_start = np.ones((1,), dtype=bool)

    while not done:
        action, lstm_states = model.predict(
            obs,
            state          = lstm_states,
            episode_start  = episode_start,
            deterministic  = True,
        )
        obs, _, terminated, truncated, _ = env.step(int(action))
        done          = terminated or truncated
        episode_start = np.array([done])

    stats = env.summary()
    print("\n" + "=" * 58)
    print("RECURRENT PPO — TEST SET RESULTS")
    print("=" * 58)
    for k, v in stats.items():
        print(f"  {k:<26} {v}")
    print("─" * 58)
    print("ML bot reference (Jan–Apr 2026 backtest):")
    print("  win_rate                   0.692")
    print("  profit_factor              2.49")
    print("  max_drawdown               ~0.088")
    print("=" * 58)
    return stats


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global TOTAL_STEPS

    parser = argparse.ArgumentParser(
        description="Train RecurrentPPO RL agent on multi-timeframe XAUUSD data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run from DB (first time)
  python rl/train_rl_mtf.py --start-date 2024-01-01 --end-date 2026-04-30

  # Cached run (subsequent)
  python rl/train_rl_mtf.py --from-cache

  # Resume interrupted training
  python rl/train_rl_mtf.py --from-cache --resume

  # Quick smoke test (50k steps)
  python rl/train_rl_mtf.py --from-cache --steps 50000
        """,
    )
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date",   default="2026-04-30")
    parser.add_argument("--from-cache", action="store_true")
    parser.add_argument("--resume",     action="store_true")
    parser.add_argument("--steps",      type=int, default=TOTAL_STEPS)
    args = parser.parse_args()

    TOTAL_STEPS = args.steps

    if args.from_cache:
        tf_dfs = load_from_cache()
    else:
        tf_dfs = load_all_from_db(args.start_date, args.end_date)
        save_cache(tf_dfs)

    train_dfs, test_dfs = split_data(tf_dfs)
    model = train(train_dfs, test_dfs, resume=args.resume)
    evaluate(model, test_dfs)


if __name__ == "__main__":
    main()
