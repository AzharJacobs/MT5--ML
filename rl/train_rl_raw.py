"""
rl/train_rl_raw.py — Train a self-learning PPO agent on raw OHLCV price windows.

No feature engineering. No zones. No rule columns.
The agent learns everything from normalised OHLCV windows across 15min, 1H, 4H.

SL/TP: ATR-based (sl_atr=1.5, tp_atr=2.5 → RR ≈ 1.67).
Obs:   3 TFs × 50 candles × 5 OHLCV features + time + position = 756 dims
Net:   MLP [512, 256]
Steps: 3M (~25-35 min on RTX 3060)

Usage:
  python rl/train_rl_raw.py --start-date 2022-01-01 --end-date 2026-05-08
  python rl/train_rl_raw.py --from-cache
  python rl/train_rl_raw.py --from-cache --resume
  python rl/train_rl_raw.py --from-cache --steps 50000
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

RL_DIR      = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(RL_DIR, "models")
DATA_DIR    = os.path.join(RL_DIR, "data")
BEST_DIR    = os.path.join(MODELS_DIR, "best_model_raw")
FINAL_MODEL = os.path.join(MODELS_DIR, "final_model_raw")

TIMEFRAMES      = ["15min", "1H", "4H"]
PRIMARY_TF      = "15min"
TRAIN_SPLIT     = 0.80
WINDOW_SIZE     = 50
SL_ATR          = 1.5
TP_ATR          = 2.5
EPISODE_LENGTH  = 500
TOTAL_STEPS     = 3_000_000
EVAL_FREQ       = 20_000
N_EVAL_EPS      = 5
INITIAL_BALANCE = 5_000.0

_DAY_NAME_MAP = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
    "Friday": 4, "Saturday": 5, "Sunday": 6,
}


def _add_session_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
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


def _load_raw_tf(timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
    from data.loader import get_connection
    db = get_connection()
    if not db.connect():
        raise ConnectionError("PostgreSQL connection failed.")
    query = """
        SELECT timestamp, open, high, low, close, volume, hour, day_of_week, month
        FROM xauusd_ohlcv
        WHERE symbol = 'XAUUSDm' AND timeframe = %s
          AND timestamp >= %s AND timestamp <= %s
        ORDER BY timestamp ASC
    """
    logger.info("Fetching raw %s from DB  (%s → %s)...", timeframe, start_date, end_date)
    df = db.fetch_dataframe(query, (timeframe, start_date, end_date))
    db.disconnect()
    if df.empty:
        raise ValueError(f"No data for {timeframe}")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = _add_session_id(df)
    logger.info("%s: %d raw bars", timeframe, len(df))
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
            raise FileNotFoundError(f"Cache missing: {path}. Run without --from-cache first.")
        tf_dfs[tf] = pd.read_csv(path, parse_dates=["timestamp"])
        logger.info("Loaded raw %s from cache: %d bars", tf, len(tf_dfs[tf]))
    return tf_dfs


def split_data(tf_dfs: dict) -> tuple:
    primary   = tf_dfs[PRIMARY_TF]
    split_idx = int(len(primary) * TRAIN_SPLIT)
    split_ts  = primary.iloc[split_idx]["timestamp"]
    train_dfs, test_dfs = {}, {}
    for tf, df in tf_dfs.items():
        train_dfs[tf] = df[df["timestamp"] < split_ts].reset_index(drop=True)
        test_dfs[tf]  = df[df["timestamp"] >= split_ts].reset_index(drop=True)
        logger.info("%-5s  train=%d  test=%d  split=%s",
                    tf, len(train_dfs[tf]), len(test_dfs[tf]), split_ts.date())
    return train_dfs, test_dfs


def train(train_dfs: dict, test_dfs: dict, resume: bool = False):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from rl.environment_raw import XAUUSDRawEnv

    os.makedirs(MODELS_DIR, exist_ok=True)
    obs_dim = len(TIMEFRAMES) * WINDOW_SIZE * 5 + 3 + 3

    def _make_env(tf_dfs):
        def _factory():
            return Monitor(XAUUSDRawEnv(
                tf_dfs=tf_dfs, primary_tf=PRIMARY_TF, window_size=WINDOW_SIZE,
                sl_atr=SL_ATR, tp_atr=TP_ATR, initial_balance=INITIAL_BALANCE,
                episode_length=EPISODE_LENGTH,
            ))
        return _factory

    train_env = DummyVecEnv([_make_env(train_dfs)])
    eval_env  = DummyVecEnv([_make_env(test_dfs)])

    eval_cb = EvalCallback(
        eval_env, best_model_save_path=BEST_DIR,
        log_path=os.path.join(MODELS_DIR, "eval_logs_raw"),
        eval_freq=EVAL_FREQ, n_eval_episodes=N_EVAL_EPS,
        deterministic=True, verbose=1,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=100_000,
        save_path=os.path.join(MODELS_DIR, "checkpoints_raw"),
        name_prefix="ppo_raw_xauusd", verbose=1,
    )

    if resume and os.path.exists(FINAL_MODEL + ".zip"):
        logger.info("Resuming from %s", FINAL_MODEL)
        model = PPO.load(FINAL_MODEL, env=train_env)
    else:
        logger.info("New raw PPO agent | obs_dim=%d | tfs=%s", obs_dim, TIMEFRAMES)
        model = PPO(
            policy="MlpPolicy", env=train_env,
            learning_rate=3e-4, n_steps=4096, batch_size=256,
            n_epochs=10, gamma=0.99, gae_lambda=0.95,
            clip_range=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
            policy_kwargs={"net_arch": [512, 256]},
            device="cpu",   # MLP is faster on CPU — GPU overhead dominates for small nets
            verbose=1,
            tensorboard_log=os.path.join(MODELS_DIR, "tensorboard_raw"),
        )

    logger.info("Training for %d steps...", TOTAL_STEPS)
    t0 = datetime.now()
    model.learn(total_timesteps=TOTAL_STEPS, callback=[eval_cb, ckpt_cb],
                reset_num_timesteps=not resume, progress_bar=True)
    logger.info("Done in %.1f min", (datetime.now() - t0).total_seconds() / 60)
    model.save(FINAL_MODEL)
    logger.info("Saved to %s.zip", FINAL_MODEL)
    return model


def evaluate(model, test_dfs: dict):
    from rl.environment_raw import XAUUSDRawEnv
    env = XAUUSDRawEnv(
        tf_dfs=test_dfs, primary_tf=PRIMARY_TF, window_size=WINDOW_SIZE,
        sl_atr=SL_ATR, tp_atr=TP_ATR, initial_balance=INITIAL_BALANCE,
        episode_length=len(test_dfs[PRIMARY_TF]),
    )
    obs, _ = env.reset(seed=42)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated
    stats = env.summary()
    print("\n" + "=" * 56)
    print("RAW RL AGENT — TEST SET RESULTS")
    print("=" * 56)
    for k, v in stats.items():
        print(f"  {k:<26} {v}")
    print("─" * 56)
    print("ML bot reference:  win_rate=0.692  PF=2.49  dd=~8.8%")
    print("=" * 56)
    return stats


def main():
    global TOTAL_STEPS, SL_ATR, TP_ATR, WINDOW_SIZE

    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date",  default="2022-01-01")
    parser.add_argument("--end-date",    default="2026-05-08")
    parser.add_argument("--from-cache",  action="store_true")
    parser.add_argument("--resume",      action="store_true")
    parser.add_argument("--steps",       type=int,   default=TOTAL_STEPS)
    parser.add_argument("--sl-atr",      type=float, default=SL_ATR)
    parser.add_argument("--tp-atr",      type=float, default=TP_ATR)
    parser.add_argument("--window",      type=int,   default=WINDOW_SIZE)
    args = parser.parse_args()

    TOTAL_STEPS = args.steps
    SL_ATR      = args.sl_atr
    TP_ATR      = args.tp_atr
    WINDOW_SIZE = args.window

    logger.info("Config: window=%d  sl_atr=%.1f  tp_atr=%.1f  rr=%.2f  steps=%d",
                WINDOW_SIZE, SL_ATR, TP_ATR, TP_ATR / SL_ATR, TOTAL_STEPS)

    tf_dfs = load_from_cache() if args.from_cache else load_all_raw(args.start_date, args.end_date)
    if not args.from_cache:
        save_cache(tf_dfs)

    train_dfs, test_dfs = split_data(tf_dfs)
    model = train(train_dfs, test_dfs, resume=args.resume)
    evaluate(model, test_dfs)


if __name__ == "__main__":
    main()
