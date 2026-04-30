# CHANGES

## Summary of all files touched in this refactor (2026-05-01)

---

### NEW FILES

#### `config/__init__.py`
Empty file. Makes `config/` a Python package so `from config.pipeline_config import ...` works across the project.

#### `config/pipeline_config.py`
**Single source of truth for all shared pipeline constants.**

- `SESSION_HOURS_CORE`, `SESSION_HOUR_12_MAX_MINUTE`, `SESSION_HOUR_LONDON_NY` — session window definition, mirrors `signal_generator.py` exactly
- `MIN_ZONE_QUALITY = 1.5` — lowered from 3.0 in `engine.py`
- `HTF_EXTREME_THRESHOLD = 0.8` — threshold for the new soft HTF gate
- `DEFAULT_CONFIDENCE_THRESHOLD = 0.52`
- `MIN_RR = 1.5`, `MAX_TP_ATR = 6.0`, `MAX_SL_ATR = 3.0`, `MAX_BARS_FORWARD = 50`, `SMOTE_RATIO = 0.4`
- `REQUIRED_FEATURE_COLUMNS` — canonical feature column list (previously duplicated in `feature_engineer.py`)

#### `scripts/check_model_health.py`
**Standalone model health checker.**

- Loads `experiments/runs/model_metadata.joblib`
- Prints: `trained_at`, `recall_minority`, `f1_minority`, `optimal_threshold`, sell-wins fix status, `smote_used`, feature count
- Warns and prints "MODEL NEEDS RETRAINING — run: python -m models.trainer" if:
  - `recall_minority < 0.20`, OR
  - `trained_at` is before 2025-01-01, OR
  - `label_map` contains `-1` key (old sell-wins bug)
- Exit code 0 = healthy, 1 = needs retraining
- Run: `python scripts/check_model_health.py`

#### `scripts/run_pipeline.py`
**Ordered pipeline entry point.**

Runs in sequence:
1. `scripts/check_model_health.py`
2. `python -m tests.test_label_logic`
3. `python -m backtest.engine --timeframe 15min --cash 1000 --start-date 2024-11-08`
4. Prints final summary table

Each step catches exceptions and reports clearly. `--skip-checks` flag skips steps 1 and 2.
Run: `python scripts/run_pipeline.py`

---

### MODIFIED FILES

#### `config/pipeline_config.py` (new, see above)

#### `data/feature_engineer.py`
Three changes:

1. **Import from config** — `from config.pipeline_config import REQUIRED_FEATURE_COLUMNS` added at top
2. **`in_session` feature** — added in `build_features()` before the warmup slice. Binary column: 1 if bar is in London open / London-NY overlap session (hours 10, 11, 12<:30, 16). The model now learns session importance itself instead of having it enforced as a hard gate.
3. **Feature consistency assertion** — at the end of `build_features()`, raises `ValueError` naming any missing columns from `REQUIRED_FEATURE_COLUMNS` rather than silently returning NaN columns.
4. **`FEATURE_COLUMNS` alias** — the large hardcoded list at the bottom is replaced with `FEATURE_COLUMNS = REQUIRED_FEATURE_COLUMNS`. All downstream callers (`from data.feature_engineer import FEATURE_COLUMNS`) are unchanged.

#### `backtest/engine.py`
Four changes:

1. **Import from config** — `from config.pipeline_config import MIN_ZONE_QUALITY, HTF_EXTREME_THRESHOLD` replaces the hardcoded `MIN_ZONE_QUALITY = 3.0` module-level constant.
2. **Session gate removed** from `MLSignalStrategy.next()`. The `asia_london` / `london_ny` block and `self._diag["session"]` increment are gone. The `filtered_session` field in `BacktestResult` stays at 0 for backward compatibility. Session is now a model feature (`in_session`).
3. **Soft HTF gate** replaces the hard HTF gate. Old behaviour: block ALL sells when `htf_4h_bias > 0`, block ALL buys when `htf_4h_bias < 0`. New behaviour: only block if `|htf_4h_bias| > HTF_EXTREME_THRESHOLD` (0.8) **AND** `winner_proba < confidence`. High-confidence counter-trend trades are allowed through.
4. **Diagnostic % table** in `main()`. After the filter count table, each gate now shows a `% of total` column. If any gate rejects > 60% of bars, a `WARNING` line prints with the gate name and a suggested fix.

#### `strategy/base_strategy.py`
One change:

- **Session gate removed** from `apply_strategy()`. The `USE_TIME_FILTERS` / `low_activity_hours` block that returned `'neutral'` for off-hours and weekends is removed. The `USE_TIME_FILTERS` flag and constant remain defined for backward compatibility.

#### `risk/stop_loss.py`
Complete rewrite as thin wrapper:

- Old code: standalone ATR-based logic using wrong field names (`htf_demand_low`, `htf_supply_high` — these columns do not exist in the feature matrix).
- New code: calls `calculate_stop_loss()` from `strategy/base_strategy.py`, which is the canonical implementation. Falls back to `entry_price ± (atr * sl_buffer_atr)` only when `calculate_stop_loss()` returns `None`.
- All existing `test_risk.py` tests (`test_stop_loss_buy_below_entry`, `test_stop_loss_sell_above_entry`) continue to pass.

#### `tests/test_strategy.py`
Three changes:

1. **`test_generate_labels_returns_series`** renamed to **`test_generate_labels_returns_dataframe`** and fixed — `generate_labels()` returns a `pd.DataFrame`, not a `pd.Series`.
2. **`_make_featured_df()`** updated — added all columns that `generate_labels()` actually requires: `timestamp`, `hour`, `atr_14`, `volume_ratio`, `between_zones`, zone boundary columns.
3. **`test_sell_wins_nonzero()`** added — regression guard for the `sell_wins=0` bug. Creates synthetic data with a valid sell setup followed by declining price that hits TP. Asserts `sell_wins > 0`. If this test fails, binary label fix was reverted.
4. **`test_buy_wins_nonzero()`** added — mirror test for buy direction.

---

### UNCHANGED FILES

The following files were read but not modified:

- `strategy/signal_generator.py` — session filter stays here for label generation (training labels remain session-aware)
- `models/trainer.py`, `data/pipeline.py`, `data/loader.py`
- `tests/test_risk.py`, `tests/test_label_logic.py`, `tests/test_backtest.py`, `tests/test_features.py`
- All `execution/`, `monitoring/`, `risk/portfolio_manager.py`, `risk/position_sizer.py`
- `config/experiment.yaml`, `config/risk.yaml`
