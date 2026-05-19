"""
rl/test_env_mtf.py — Quick sanity checks for XAUUSDMultiTFEnv.

Tests 3 things without touching the DB:
  1. Env instantiates correctly with synthetic data
  2. Obs shape matches declared observation_space
  3. 500 random-action steps complete without error

Usage:
  python rl/test_env_mtf.py
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.pipeline_config import RL_FEATURE_COLUMNS
from data.macro_features import add_macro_features
from rl.environment_mtf import XAUUSDMultiTFEnv


def _make_synthetic_df(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Build a minimal synthetic OHLCV + feature DataFrame for testing."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2024-01-01", periods=n, freq="15min")

    close = 2000.0 + np.cumsum(rng.normal(0, 2, n))
    high  = close + rng.uniform(0.5, 5.0, n)
    low   = close - rng.uniform(0.5, 5.0, n)
    open_ = close + rng.normal(0, 1, n)
    vol   = rng.uniform(100, 1000, n)

    df = pd.DataFrame({
        "timestamp":   timestamps,
        "open":        open_,
        "high":        high,
        "low":         low,
        "close":       close,
        "volume":      vol,
        "hour":        timestamps.hour.astype(float),
        "day_of_week": timestamps.dayofweek.astype(float),
        "month":       timestamps.month.astype(float),
        "candle_size": (high - low),
        "body_size":   np.abs(close - open_),
        "wick_upper":  (high - np.maximum(close, open_)),
        "wick_lower":  (np.minimum(close, open_) - low),
    })

    # Add ATR so macro_features can use it
    hl = df["high"] - df["low"]
    df["atr_14"] = hl.ewm(span=14, adjust=False).mean()

    # Add macro features (requires atr_14, close, high, low, open)
    df = add_macro_features(df)

    # Fill all RL_FEATURE_COLUMNS that are missing with 0
    for col in RL_FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0

    return df.reset_index(drop=True)


def test_env():
    print("Building synthetic DataFrames...")
    primary_df = _make_synthetic_df(n=3000, seed=1)

    # Build secondary TFs by downsampling (realistic enough for shape tests)
    h1_df  = _make_synthetic_df(n=500,  seed=2)
    h4_df  = _make_synthetic_df(n=200,  seed=3)
    m5_df  = _make_synthetic_df(n=8000, seed=4)

    tf_dfs = {
        "5min":  m5_df,
        "15min": primary_df,
        "1H":    h1_df,
        "4H":    h4_df,
    }

    # ── Test 1: instantiation ────────────────────────────────────────────────
    print("Test 1: Instantiation...")
    env = XAUUSDMultiTFEnv(
        tf_dfs=tf_dfs,
        feature_columns=RL_FEATURE_COLUMNS,
        primary_tf="15min",
        initial_balance=5_000.0,
        episode_length=500,
    )
    print(f"  obs_space shape: {env.observation_space.shape}")
    expected_dim = len(tf_dfs) * len(RL_FEATURE_COLUMNS) + 3
    assert env.observation_space.shape == (expected_dim,), \
        f"Expected ({expected_dim},) got {env.observation_space.shape}"
    print(f"  PASS — obs_dim={expected_dim} ({len(tf_dfs)} TFs × {len(RL_FEATURE_COLUMNS)} features + 3 ctx)")

    # ── Test 2: reset ────────────────────────────────────────────────────────
    print("Test 2: Reset...")
    obs, info = env.reset(seed=99)
    assert obs.shape == (expected_dim,), f"Bad obs shape: {obs.shape}"
    assert obs.dtype == np.float32, f"Bad obs dtype: {obs.dtype}"
    assert not np.any(np.isnan(obs)), "NaN in initial obs"
    print(f"  PASS — obs shape={obs.shape}, no NaNs")

    # ── Test 3: 500 random steps ─────────────────────────────────────────────
    print("Test 3: 500 random steps...")
    rng = np.random.default_rng(0)
    n_trades = 0
    for step in range(500):
        action = int(rng.integers(0, 3))
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (expected_dim,), f"Bad obs shape at step {step}"
        assert not np.any(np.isnan(obs)), f"NaN in obs at step {step}"
        assert isinstance(reward, float), f"Bad reward type at step {step}"
        n_trades = info["n_trades"]
        if terminated or truncated:
            obs, _ = env.reset()
            break
    print(f"  PASS — completed, trades_so_far={n_trades}, final_equity={info['equity']:.2f}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("ALL TESTS PASSED")
    print(f"  obs_dim:         {expected_dim}")
    print(f"  n_features/TF:   {len(RL_FEATURE_COLUMNS)}")
    print(f"  n_timeframes:    {len(tf_dfs)}")
    print("=" * 50)
    print("\nReady to train. Next step:")
    print("  python rl/train_rl_mtf.py --start-date 2024-01-01 --end-date 2026-04-30 --steps 50000")
    print("  (smoke test — ~10 min, no GPU needed)")
    print("\n  python rl/train_rl_mtf.py --start-date 2022-01-01 --end-date 2026-04-30")
    print("  (full 5M step training — ~3-4 hours)")


if __name__ == "__main__":
    test_env()
