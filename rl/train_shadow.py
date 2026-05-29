"""
rl/train_shadow.py — Train the shadow RL agent that filters/augments ML signals.

Pipeline:
  1. Load multi-TF feature cache (reuse rl/data/features_*.csv from train_rl_mtf.py)
  2. Pre-compute ML signals for the 15min primary TF:
       • Load saved ML model bundle (experiments/runs/)
       • Run inference on each bar — add ml_signal / ml_prob / ml_grade columns
       • Falls back to rule-based proxy if no model exists yet
  3. Build XAUUSDShadowEnv for train + test splits
  4. Train RecurrentPPO (LSTM) for TOTAL_STEPS
  5. Save to rl/models/shadow_model/  (best checkpoint)
              rl/models/final_shadow_model.zip

Requirements:
  pip install sb3-contrib stable-baselines3

Usage:
  # Prerequisites: run train_rl_mtf.py first to build the feature cache
  python rl/train_rl_mtf.py --from-cache          # builds rl/data/features_*.csv

  # Train shadow agent (uses cached features)
  python rl/train_shadow.py

  # Quick smoke test
  python rl/train_shadow.py --steps 50000

  # Resume interrupted run
  python rl/train_shadow.py --resume

Output:
  rl/models/shadow_model/best_model.zip  — best checkpoint (EvalCallback)
  rl/models/final_shadow_model.zip       — final weights at end of run
  rl/models/tensorboard_shadow/          — TensorBoard logs
"""

import os
import sys
import argparse
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.pipeline_config import (
    RL_FEATURE_COLUMNS, RL_FEATURE_COLUMNS_CYCLICAL,
    REQUIRED_FEATURE_COLUMNS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("rl.train_shadow")

# ── Paths ──────────────────────────────────────────────────────────────────────
RL_DIR      = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(RL_DIR, "models")
DATA_DIR    = os.path.join(RL_DIR, "data")
SHADOW_DIR  = os.path.join(MODELS_DIR, "shadow_model")
FINAL_MODEL = os.path.join(MODELS_DIR, "final_shadow_model")

# ── Configuration ──────────────────────────────────────────────────────────────
TIMEFRAMES   = ["5min", "15min", "1H", "4H"]
PRIMARY_TF   = "15min"
TRAIN_SPLIT  = 0.80
TOTAL_STEPS  = 2_000_000
EVAL_FREQ    = 20_000
N_EVAL_EPS   = 5
EPISODE_LEN  = 2000
INIT_BALANCE = 5_000.0

# Grade thresholds that mirror live_trader.py exactly
GRADE_A_ZONE   = 3.5
GRADE_A_CONF   = 0.42
GRADE_B_ZONE   = 3.0
GRADE_B_CONF   = 0.40
SKIP_GRADE_B   = True   # mirrors GRADE_MULTIPLIERS = {"B": 0}


# ─────────────────────────────────────────────────────────────────────────────
#  ML signal pre-computation
# ─────────────────────────────────────────────────────────────────────────────

def _grade(zone_q: float, prob: float) -> str:
    if zone_q >= GRADE_A_ZONE and prob >= GRADE_A_CONF:
        return "A"
    if zone_q >= GRADE_B_ZONE and prob >= GRADE_B_CONF:
        return "B"
    return "C"


def _eval_row(row: pd.Series, model, scaler, feature_cols: list,
              threshold: float, classes: list) -> dict:
    """Run the ML model on a single feature row. Returns signal dict."""
    no_trade = {"signal": 0, "prob": 0.0, "grade": ""}

    in_demand = float(row.get("in_demand_zone", 0) or 0) == 1.0
    in_supply = float(row.get("in_supply_zone", 0) or 0) == 1.0
    if not in_demand and not in_supply:
        return no_trade
    if float(row.get("between_zones", 0) or 0) == 1.0:
        return no_trade

    X = pd.DataFrame([{c: float(row.get(c, 0) or 0) for c in feature_cols}])
    if scaler is not None:
        X = pd.DataFrame(scaler.transform(X), columns=feature_cols)

    proba  = model.predict_proba(X)[0]
    prob_w = float(proba[classes.index(1)]) if 1 in classes else float(proba[-1])

    if prob_w < threshold:
        return no_trade

    direction = 1 if in_demand else -1
    zone_q    = float(row.get("active_zone_quality", 0) or 0)
    gr        = _grade(zone_q, prob_w)

    if SKIP_GRADE_B and gr == "B":
        return no_trade

    return {"signal": direction, "prob": prob_w, "grade": gr}


def compute_ml_signals_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ml_signal / ml_prob / ml_grade to df using the saved ML model bundle.
    Loads from experiments/runs/ (same path as live_trader.py).
    """
    from models.trainer import MODEL_DIR, MODEL_FILE, METADATA_FILE

    model_path = os.path.join(MODEL_DIR, MODEL_FILE)
    meta_path  = os.path.join(MODEL_DIR, METADATA_FILE)

    if not os.path.exists(model_path):
        logger.warning(
            "No ML model at %s — falling back to rule-based signal proxy.", model_path
        )
        return compute_ml_signals_rules(df)

    logger.info("Loading ML model from %s ...", model_path)
    model = joblib.load(model_path)
    meta  = joblib.load(meta_path) if os.path.exists(meta_path) else {}

    feature_cols = meta.get("feature_columns", REQUIRED_FEATURE_COLUMNS)
    scaler       = meta.get("scaler")
    threshold    = float(meta.get("optimal_threshold", 0.5))
    classes      = list(model.classes_) if hasattr(model, "classes_") else [0, 1]

    logger.info(
        "Pre-computing ML signals for %d bars  (threshold=%.3f)...", len(df), threshold
    )

    out = df.copy()
    out["ml_signal"] = 0
    out["ml_prob"]   = 0.0
    out["ml_grade"]  = ""

    signals  = [_eval_row(row, model, scaler, feature_cols, threshold, classes)
                for _, row in df.iterrows()]

    out["ml_signal"] = [s["signal"] for s in signals]
    out["ml_prob"]   = [s["prob"]   for s in signals]
    out["ml_grade"]  = [s["grade"]  for s in signals]

    buys  = (out["ml_signal"] == 1).sum()
    sells = (out["ml_signal"] == -1).sum()
    logger.info(
        "ML signals: %d buy  %d sell  %d flat  (active=%.1f%%)",
        buys, sells, len(out) - buys - sells,
        (buys + sells) / max(len(out), 1) * 100,
    )
    return out


def compute_ml_signals_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rule-based proxy for ML signals when no trained model is available.
    Uses rule_buy_score / rule_sell_score from feature_engineer.py.
    Produces signal only when score >= 4 and zone quality >= 2.0.
    """
    out = df.copy()
    out["ml_signal"] = 0
    out["ml_prob"]   = 0.0
    out["ml_grade"]  = ""

    buy_score  = out.get("rule_buy_score",  pd.Series(0, index=out.index)).fillna(0)
    sell_score = out.get("rule_sell_score", pd.Series(0, index=out.index)).fillna(0)
    zone_q     = out.get("active_zone_quality", pd.Series(0, index=out.index)).fillna(0)
    in_demand  = out.get("in_demand_zone", pd.Series(0, index=out.index)).fillna(0).astype(float) == 1.0
    in_supply  = out.get("in_supply_zone", pd.Series(0, index=out.index)).fillna(0).astype(float) == 1.0

    buy_mask  = in_demand & (buy_score  >= 4) & (zone_q >= 2.0)
    sell_mask = in_supply & (sell_score >= 4) & (zone_q >= 2.0)

    # Buy signals
    buy_prob  = (0.40 + buy_score[buy_mask] * 0.05).clip(upper=0.85)
    buy_grade = np.where(
        (zone_q[buy_mask] >= GRADE_A_ZONE) & (buy_score[buy_mask] >= 6), "A", "C"
    )
    out.loc[buy_mask, "ml_signal"] = 1
    out.loc[buy_mask, "ml_prob"]   = buy_prob
    out.loc[buy_mask, "ml_grade"]  = buy_grade

    # Sell signals (only where not already buy)
    sell_only = sell_mask & ~buy_mask
    sell_prob  = (0.40 + sell_score[sell_only] * 0.05).clip(upper=0.85)
    sell_grade = np.where(
        (zone_q[sell_only] >= GRADE_A_ZONE) & (sell_score[sell_only] >= 6), "A", "C"
    )
    out.loc[sell_only, "ml_signal"] = -1
    out.loc[sell_only, "ml_prob"]   = sell_prob
    out.loc[sell_only, "ml_grade"]  = sell_grade

    buys  = (out["ml_signal"] == 1).sum()
    sells = (out["ml_signal"] == -1).sum()
    logger.info(
        "Rule proxy signals: %d buy  %d sell  (active=%.1f%%)",
        buys, sells, (buys + sells) / max(len(out), 1) * 100,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Cache loading (reuses train_rl_mtf.py cache)
# ─────────────────────────────────────────────────────────────────────────────

TF_BUILD_PARAMS = {
    "5min":  {"impulse_atr_multiplier": 0.5, "include_london_ny": True},
    "15min": {"impulse_atr_multiplier": 0.3, "include_london_ny": False},
    "1H":    {"impulse_atr_multiplier": 0.5, "include_london_ny": True},
    "4H":    {"impulse_atr_multiplier": 0.5, "include_london_ny": True},
}


def _build_from_raw(tf: str) -> pd.DataFrame:
    """Build features_{tf}.csv from raw_{tf}.csv when features cache is missing."""
    from data.feature_engineer import build_features
    from data.macro_features import add_macro_features

    raw_path = os.path.join(DATA_DIR, f"raw_{tf}.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Neither features_{tf}.csv nor raw_{tf}.csv found in {DATA_DIR}.\n"
            "Run: python rl/train_rl_mtf.py --start-date 2024-01-01 --end-date 2026-04-30"
        )

    logger.info("%s: features cache missing — building from %s ...", tf, raw_path)
    df = pd.read_csv(raw_path, parse_dates=["timestamp"])

    # Derive columns required by build_features() that raw exports omit
    if "year" not in df.columns:
        df["year"] = df["timestamp"].dt.year
    if "candle_size" not in df.columns:
        df["candle_size"] = df["high"] - df["low"]
    if "body_size" not in df.columns:
        df["body_size"] = (df["close"] - df["open"]).abs()
    if "wick_upper" not in df.columns:
        df["wick_upper"] = df["high"] - df[["open", "close"]].max(axis=1)
    if "wick_lower" not in df.columns:
        df["wick_lower"] = df[["open", "close"]].min(axis=1) - df["low"]

    params   = TF_BUILD_PARAMS.get(tf, {})
    feat_df  = build_features(
        df,
        impulse_atr_multiplier=params.get("impulse_atr_multiplier", 0.5),
        include_london_ny=params.get("include_london_ny", True),
    )
    feat_df  = add_macro_features(feat_df)

    out_path = os.path.join(DATA_DIR, f"features_{tf}.csv")
    feat_df.to_csv(out_path, index=False)
    logger.info("%s: saved %d bars → %s", tf, len(feat_df), out_path)
    return feat_df


def load_from_cache() -> dict:
    from data.macro_features import add_macro_features
    from config.pipeline_config import RL_MACRO_FEATURE_COLUMNS

    tf_dfs = {}
    for tf in TIMEFRAMES:
        path = os.path.join(DATA_DIR, f"features_{tf}.csv")
        if not os.path.exists(path):
            df = _build_from_raw(tf)
        else:
            df = pd.read_csv(path, parse_dates=["timestamp"])
            if any(c not in df.columns for c in RL_MACRO_FEATURE_COLUMNS):
                logger.info("%s: macro features missing from cache — recomputing...", tf)
                df = add_macro_features(df)
        tf_dfs[tf] = df
        logger.info("Loaded %-5s: %d bars", tf, len(df))
    return tf_dfs


def split_data(tf_dfs: dict) -> tuple:
    primary   = tf_dfs[PRIMARY_TF]
    split_idx = int(len(primary) * TRAIN_SPLIT)
    split_ts  = primary.iloc[split_idx]["timestamp"]

    train_dfs, test_dfs = {}, {}
    for tf, df in tf_dfs.items():
        train_dfs[tf] = df[df["timestamp"] < split_ts].reset_index(drop=True)
        test_dfs[tf]  = df[df["timestamp"] >= split_ts].reset_index(drop=True)
        logger.info(
            "%-5s  train=%d  test=%d  split=%s",
            tf, len(train_dfs[tf]), len(test_dfs[tf]), split_ts.date(),
        )
    return train_dfs, test_dfs


# ─────────────────────────────────────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────────────────────────────────────

def train(train_dfs: dict, test_dfs: dict, resume: bool = False,
          session_encoding: str = "binary"):
    try:
        from sb3_contrib import RecurrentPPO
    except ImportError:
        raise ImportError(
            "sb3-contrib is required.\n"
            "Install with: pip install sb3-contrib"
        )

    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from rl.environment_shadow import XAUUSDShadowEnv

    feat_cols = (
        RL_FEATURE_COLUMNS_CYCLICAL if session_encoding == "cyclical"
        else RL_FEATURE_COLUMNS
    )
    suffix = f"_{session_encoding}" if session_encoding != "binary" else ""
    shadow_dir  = os.path.join(MODELS_DIR, f"shadow_model{suffix}")
    final_model = os.path.join(MODELS_DIR, f"final_shadow_model{suffix}")

    os.makedirs(MODELS_DIR, exist_ok=True)

    logger.info(
        "Session encoding: %s  |  feature_cols=%d per TF",
        session_encoding, len(feat_cols),
    )

    # Pre-compute ML signals for both splits
    logger.info("Pre-computing ML signals for training set (%d bars)...", len(train_dfs[PRIMARY_TF]))
    train_primary = compute_ml_signals_model(train_dfs[PRIMARY_TF])

    logger.info("Pre-computing ML signals for test set (%d bars)...", len(test_dfs[PRIMARY_TF]))
    test_primary  = compute_ml_signals_model(test_dfs[PRIMARY_TF])

    secondary_tfs = [tf for tf in TIMEFRAMES if tf != PRIMARY_TF]

    def _make_env(primary_df, sec_dfs):
        def _factory():
            env = XAUUSDShadowEnv(
                primary_df      = primary_df,
                secondary_dfs   = {tf: sec_dfs[tf] for tf in secondary_tfs},
                feature_columns = feat_cols,
                initial_balance = INIT_BALANCE,
                episode_length  = EPISODE_LEN,
            )
            return Monitor(env)
        return _factory

    train_sec = {tf: train_dfs[tf] for tf in secondary_tfs}
    test_sec  = {tf: test_dfs[tf]  for tf in secondary_tfs}

    train_env = DummyVecEnv([_make_env(train_primary, train_sec)])
    eval_env  = DummyVecEnv([_make_env(test_primary,  test_sec)])

    obs_dim = train_env.observation_space.shape[0]
    logger.info("Shadow env  obs_dim=%d  actions=3 (agree/skip/rotate)", obs_dim)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = shadow_dir,
        log_path             = os.path.join(MODELS_DIR, f"eval_logs_shadow{suffix}"),
        eval_freq            = EVAL_FREQ,
        n_eval_episodes      = N_EVAL_EPS,
        deterministic        = True,
        verbose              = 1,
    )
    checkpoint_cb = CheckpointCallback(
        save_freq   = 200_000,
        save_path   = os.path.join(MODELS_DIR, f"checkpoints_shadow{suffix}"),
        name_prefix = f"rppo_shadow{suffix}",
        verbose     = 1,
    )

    if resume and os.path.exists(final_model + ".zip"):
        logger.info("Resuming from %s.zip", final_model)
        model = RecurrentPPO.load(final_model, env=train_env)
    else:
        logger.info("New RecurrentPPO shadow agent | obs=%d", obs_dim)
        model = RecurrentPPO(
            policy        = "MlpLstmPolicy",
            env           = train_env,
            learning_rate = 3e-5,    # conservative: shadow needs precise credit assignment
            n_steps       = 2048,
            batch_size    = 128,
            n_epochs      = 10,
            gamma         = 0.995,   # long horizon: skip rewards come with delay
            gae_lambda    = 0.95,
            clip_range    = 0.1,
            ent_coef      = 0.02,    # enough exploration to try skipping
            vf_coef       = 0.5,
            max_grad_norm = 0.5,
            policy_kwargs = {
                "lstm_hidden_size": 256,
                "n_lstm_layers":    2,
                "net_arch":         [256, 128],
            },
            verbose         = 1,
            tensorboard_log = os.path.join(MODELS_DIR, "tensorboard_shadow"),
        )

    logger.info("Training shadow agent for %d steps...", TOTAL_STEPS)
    t0 = datetime.now()
    model.learn(
        total_timesteps     = TOTAL_STEPS,
        callback            = [eval_cb, checkpoint_cb],
        reset_num_timesteps = not resume,
        progress_bar        = True,
    )
    elapsed = (datetime.now() - t0).total_seconds()
    logger.info("Training complete in %.1f min", elapsed / 60)

    model.save(final_model)
    logger.info("Saved final model → %s.zip", final_model)
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  Test-set evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, test_dfs: dict, session_encoding: str = "binary") -> dict:
    from rl.environment_shadow import XAUUSDShadowEnv

    feat_cols = (
        RL_FEATURE_COLUMNS_CYCLICAL if session_encoding == "cyclical"
        else RL_FEATURE_COLUMNS
    )

    test_primary  = compute_ml_signals_model(test_dfs[PRIMARY_TF])
    secondary_tfs = [tf for tf in TIMEFRAMES if tf != PRIMARY_TF]

    env = XAUUSDShadowEnv(
        primary_df      = test_primary,
        secondary_dfs   = {tf: test_dfs[tf] for tf in secondary_tfs},
        feature_columns = feat_cols,
        initial_balance = INIT_BALANCE,
        episode_length  = len(test_primary),
    )

    obs, _        = env.reset(seed=42)
    lstm_states   = None
    done          = False
    episode_start = np.ones((1,), dtype=bool)

    while not done:
        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=episode_start, deterministic=True
        )
        obs, _, terminated, truncated, _ = env.step(int(action))
        done          = terminated or truncated
        episode_start = np.array([done])

    stats = env.summary()
    print("\n" + "=" * 60)
    print("SHADOW AGENT — TEST-SET RESULTS")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k:<26} {v}")
    print("─" * 60)
    print("Reference (ML bot Jan–Apr 2026 backtest):")
    print("  win_rate                   0.692")
    print("  profit_factor              2.49")
    print("  max_drawdown               ~0.088")
    print("=" * 60)
    return stats


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global TOTAL_STEPS

    parser = argparse.ArgumentParser(
        description="Train RecurrentPPO shadow agent for XAUUSD ML bot filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full training run (uses feature cache from train_rl_mtf.py)
  python rl/train_shadow.py

  # Quick smoke test
  python rl/train_shadow.py --steps 50000

  # Resume after interruption
  python rl/train_shadow.py --resume

  # Skip evaluation after training
  python rl/train_shadow.py --no-eval
        """,
    )
    parser.add_argument("--steps",            type=int, default=TOTAL_STEPS, help="Total training steps")
    parser.add_argument("--resume",           action="store_true",           help="Resume from final_shadow_model.zip")
    parser.add_argument("--no-eval",          action="store_true",           help="Skip test-set evaluation")
    parser.add_argument("--session-encoding", default="binary",
                        choices=["binary", "cyclical"],
                        help="binary (default): in_session+session_id; "
                             "cyclical: hour_sin+hour_cos (A/B experiment)")
    args = parser.parse_args()

    TOTAL_STEPS = args.steps

    tf_dfs = load_from_cache()
    train_dfs, test_dfs = split_data(tf_dfs)
    model = train(train_dfs, test_dfs, resume=args.resume,
                  session_encoding=args.session_encoding)

    if not args.no_eval:
        evaluate(model, test_dfs, session_encoding=args.session_encoding)


if __name__ == "__main__":
    main()
