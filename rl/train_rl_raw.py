"""
rl/train_rl_raw.py — Train a self-learning PPO agent on raw OHLCV price windows.

No feature engineering. No zones. No rule columns. The agent learns everything
from normalised OHLCV windows across 5min, 15min, 1H, 4H timeframes.

SL/TP are ATR-based (sl_atr=1.5 × ATR, tp_atr=2.5 × ATR → RR ≈ 1.67).

Observation: 4 TFs × 50 candles × 5 OHLCV features + time + position = 1006 dims
Network:     MLP [512, 256] (tapers from large obs to action space)
Training:    1M steps, ~2–3 h on GPU

Usage:
  # First run — pull raw OHLCV from DB, cache, train
  python rl/train_rl_raw.py --start-date 2024-01-01 --end-date 2026-02-28

  # Cached run
  python rl/train_rl_raw.py --from-cache

  # Resume
  python rl/train_rl_raw.py --from-cache --resume

  # Quick smoke test
  python rl/train_rl_raw.py --from-cache --steps 50000

Output:
  rl/models/best_model_raw/   — best checkpoint (EvalCallback)
  rl/models/final_model_raw   — final model
  rl/data/raw_<TF>.csv        — cached raw OHLCV (no feature building)
"""

import os
import sys
import argparse
import logging
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("rl.train_raw")

# ── Paths ──────────────────────────────────────────────────────────────────────
RL_DIR      = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(RL_DIR, "models")
DATA_DIR    = os.path.join(RL_DIR, "data")
BEST_DIR    = os.path.join(MODELS_DIR, "best_model_raw")
FINAL_MODEL = os.path.join(MODELS_DIR, "final_model_raw")

# ── Timeframes ─────────────────────────────────────────────────────────────────
TIMEFRAMES = ["15min", "1H", "4H"]
PRIMARY_TF = "15min"

# ── Hyperparameters ────────────────────────────────────────────────────────────
TRAIN_SPLIT     = 0.80
WINDOW_SIZE     = 50      # candles of lookback per TF in observation
SL_ATR          = 1.5
TP_ATR          = 2.5
EPISODE_LENGTH  = 500
TOTAL_STEPS     = 3_000_000
EVAL_FREQ       = 20_000
N_EVAL_EPS      = 5
INITIAL_BALANCE = 5_000.0


# ─────────────────────────────────────────────────────────────────────────────
#  Session ID helper (session_id column for time context in obs)
# ─────────────────────────────────────────────────────────────────────────────

_DAY_NAME_MAP = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
    "Friday": 4, "Saturday": 5, "Sunday": 6,
}

def _add_session_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute session_id from hour (broker time GMT+3):
      1 = London open  (10, 11, 12)
      2 = NY open      (13, 14)
      3 = overlap      (15, 16)
      0 = off-hours
    Also normalises day_of_week to int in case the DB stores day names.
    """
    df = df.copy()

    # Normalise day_of_week: "Thursday" → 3 (DB stores day names as strings)
    df["day_of_week"] = df["day_of_week"].map(
        lambda x: _DAY_NAME_MAP.get(str(x), x) if isinstance(x, str) else x
    ).fillna(0).astype(int)

    h = df["hour"].astype(int)
    session = pd.Series(0, index=df.index)
    session[h.isin([10, 11, 12])] = 1
    session[h.isin([13, 14])]     = 2
    session[h.isin([15, 16])]     = 3
    df["session_id"] = session.values
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Data loading — raw OHLCV only (no feature_engineer.py)
# ─────────────────────────────────────────────────────────────────────────────

def _load_raw_tf(timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
    from data.loader import get_connection

    db = get_connection()
    if not db.connect():
        raise ConnectionError("PostgreSQL connection failed. Check .env credentials.")

    query = """
        SELECT timestamp, open, high, low, close, volume,
               hour, day_of_week, month
        FROM xauusd_ohlcv
        WHERE symbol = 'XAUUSDm' AND timeframe = %s
          AND timestamp >= %s AND timestamp <= %s
        ORDER BY timestamp ASC
    """
    logger.info("Fetching raw %s from DB  (%s → %s)...", timeframe, start_date, end_date)
    df = db.fetch_dataframe(query, (timeframe, start_date, end_date))
    db.disconnect()

    if df.empty:
        raise ValueError(f"No data for {timeframe} between {start_date} and {end_date}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = _add_session_id(df)
    logger.info("%s: %d raw bars loaded", timeframe, len(df))
    return df.reset_index(drop=True)


def load_all_raw(start_date: str, end_date: str) -> dict:
    return {tf: _load_raw_tf(tf, start_date, end_date) for tf in TIMEFRAMES}


def save_cache(tf_dfs: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    for tf, df in tf_dfs.items():
        path = os.path.join(DATA_DIR, f"raw_{tf}.csv")
        df.to_csv(path, index=False)
        logger.info("Cached raw %s → %s", tf, path)


def load_from_cache() -> dict:
    tf_dfs = {}
    for tf in TIMEFRAMES:
        path = os.path.join(DATA_DIR, f"raw_{tf}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Raw cache missing for {tf}: {path}\n"
                "Run without --from-cache first."
            )
        tf_dfs[tf] = pd.read_csv(path, parse_dates=["timestamp"])
        logger.info("Loaded raw %s from cache: %d bars", tf, len(tf_dfs[tf]))
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
#  Training
# ─────────────────────────────────────────────────────────────────────────────

def train(train_dfs: dict, test_dfs: dict, resume: bool = False) -> object:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from rl.environment_raw import XAUUSDRawEnv

    os.makedirs(MODELS_DIR, exist_ok=True)

    obs_dim = len(TIMEFRAMES) * WINDOW_SIZE * 5 + 3 + 3

    def _make_env(tf_dfs):
        def _factory():
            env = XAUUSDRawEnv(
                tf_dfs          = tf_dfs,
                primary_tf      = PRIMARY_TF,
                window_size     = WINDOW_SIZE,
                sl_atr          = SL_ATR,
                tp_atr          = TP_ATR,
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
        log_path             = os.path.join(MODELS_DIR, "eval_logs_raw"),
        eval_freq            = EVAL_FREQ,
        n_eval_episodes      = N_EVAL_EPS,
        deterministic        = True,
        verbose              = 1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq   = 100_000,
        save_path   = os.path.join(MODELS_DIR, "checkpoints_raw"),
        name_prefix = "ppo_raw_xauusd",
        verbose     = 1,
    )

    if resume and os.path.exists(FINAL_MODEL + ".zip"):
        logger.info("Resuming from %s", FINAL_MODEL)
        model = PPO.load(FINAL_MODEL, env=train_env)
    else:
        logger.info("New raw PPO agent | obs_dim=%d | tfs=%s", obs_dim, TIMEFRAMES)
        model = PPO(
            policy        = "MlpPolicy",
            env           = train_env,
            # Tuned for 1006-dim input + GPU
            learning_rate = 3e-4,
            n_steps       = 4096,     # larger rollout catches more rare trades
            batch_size    = 256,
            n_epochs      = 10,
            gamma         = 0.99,
            gae_lambda    = 0.95,
            clip_range    = 0.2,
            ent_coef      = 0.01,     # encourage exploration early on
            vf_coef       = 0.5,
            max_grad_norm = 0.5,
            # Two hidden layers that taper down from the large obs
            policy_kwargs = {"net_arch": [512, 256]},
            verbose       = 1,
            tensorboard_log = os.path.join(MODELS_DIR, "tensorboard_raw"),
        )

    logger.info("Training raw PPO for %d steps (~2–3h on GPU)...", TOTAL_STEPS)
    logger.info(
        "Agent learns from scratch: no zones, no rules, no feature engineering."
    )
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
    from rl.environment_raw import XAUUSDRawEnv

    logger.info("Running full test-set evaluation (raw agent, no feature engineering)...")
    env = XAUUSDRawEnv(
        tf_dfs          = test_dfs,
        primary_tf      = PRIMARY_TF,
        window_size     = WINDOW_SIZE,
        sl_atr          = SL_ATR,
        tp_atr          = TP_ATR,
        initial_balance = INITIAL_BALANCE,
        episode_length  = len(test_dfs[PRIMARY_TF]),
    )

    obs, _ = env.reset(seed=42)
    done   = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated

    stats = env.summary()
    print("\n" + "=" * 56)
    print("RAW RL AGENT — TEST SET RESULTS (no feature engineering)")
    print("=" * 56)
    for k, v in stats.items():
        print(f"  {k:<28} {v}")
    print("─" * 56)
    print("ML bot reference (Jan–Apr 2026 backtest):")
    print("  win_rate                       0.692")
    print("  profit_factor                  2.49")
    print("  max_drawdown                   ~0.088")
    print("=" * 56)
    return stats


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global TOTAL_STEPS, SL_ATR, TP_ATR, WINDOW_SIZE

    parser = argparse.ArgumentParser(
        description="Train self-learning PPO agent on raw XAUUSD OHLCV windows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This agent learns its OWN strategy from raw price action alone.
It receives no zone labels, no rule features, no HTF bias columns.

Examples:
  # Full run from DB (first time)
  python rl/train_rl_raw.py --start-date 2024-01-01 --end-date 2026-02-28

  # Cached run (subsequent, much faster)
  python rl/train_rl_raw.py --from-cache

  # Resume interrupted training
  python rl/train_rl_raw.py --from-cache --resume

  # Quick 50k-step smoke test
  python rl/train_rl_raw.py --from-cache --steps 50000

After training, run in shadow mode alongside your live bot:
  python live_trader.py --timeframe 15min --mode paper --rl-shadow
        """,
    )
    parser.add_argument("--start-date",  default="2024-01-01")
    parser.add_argument("--end-date",    default="2026-02-28")
    parser.add_argument("--from-cache",  action="store_true",  help="Use rl/data/raw_*.csv")
    parser.add_argument("--resume",      action="store_true",  help="Resume from final_model_raw.zip")
    parser.add_argument("--steps",       type=int, default=TOTAL_STEPS)
    parser.add_argument("--sl-atr",      type=float, default=SL_ATR,  help="SL distance in ATR")
    parser.add_argument("--tp-atr",      type=float, default=TP_ATR,  help="TP distance in ATR")
    parser.add_argument("--window",      type=int,   default=WINDOW_SIZE, help="Candle lookback per TF")
    args = parser.parse_args()

    TOTAL_STEPS = args.steps
    SL_ATR      = args.sl_atr
    TP_ATR      = args.tp_atr
    WINDOW_SIZE = args.window

    logger.info(
        "Config: window=%d  sl_atr=%.1f  tp_atr=%.1f  rr=%.2f  steps=%d",
        WINDOW_SIZE, SL_ATR, TP_ATR, TP_ATR / SL_ATR, TOTAL_STEPS,
    )

    if args.from_cache:
        tf_dfs = load_from_cache()
    else:
        tf_dfs = load_all_raw(args.start_date, args.end_date)
        save_cache(tf_dfs)

    train_dfs, test_dfs = split_data(tf_dfs)
    model = train(train_dfs, test_dfs, resume=args.resume)
    evaluate(model, test_dfs)


if __name__ == "__main__":
    main()
