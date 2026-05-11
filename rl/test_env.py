"""Quick sanity check — runs the environment with random actions for 200 steps."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from rl.environment import XAUUSDTradingEnv
from config.pipeline_config import REQUIRED_FEATURE_COLUMNS

def make_dummy_df(n=600):
    """Synthetic OHLCV + feature data to test the env without needing the DB."""
    np.random.seed(42)
    price = 3000.0 + np.cumsum(np.random.randn(n) * 2)
    df = pd.DataFrame({
        "close": price,
        "open":  price - np.random.rand(n),
        "high":  price + np.random.rand(n) * 3,
        "low":   price - np.random.rand(n) * 3,
        "volume": np.random.randint(100, 1000, n).astype(float),
    })
    atr = df["high"] - df["low"]
    df["atr_14"] = atr.ewm(span=14).mean()
    df["demand_zone_bottom"] = price - 5.0
    df["demand_zone_top"]    = price - 2.0
    df["supply_zone_bottom"] = price + 2.0
    df["supply_zone_top"]    = price + 5.0
    # Fill all required feature columns with zeros if missing
    for col in REQUIRED_FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    return df

if __name__ == "__main__":
    df      = make_dummy_df(600)
    env     = XAUUSDTradingEnv(df, feature_columns=REQUIRED_FEATURE_COLUMNS, episode_length=200)
    obs, _  = env.reset(seed=0)

    print(f"Observation shape : {obs.shape}")
    print(f"Action space      : {env.action_space}")
    print(f"Running 200 random steps...")

    done = False
    step = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1

    print(f"\nEpisode finished after {step} steps")
    stats = env.summary()
    for k, v in stats.items():
        print(f"  {k:<20} {v}")
    print("\nEnvironment sanity check PASSED")
