"""
signal_generator.py — Label generation utilities.

Zone-to-Zone label generation (generate_labels, LTF zone entry logic,
HTF TP/SL derivation, session gating, worn-zone downgrade) has been
removed. It will be rebuilt from scratch in label_generator_Z&Z.py.
"""

import pandas as pd
import logging

logger = logging.getLogger("mt5_collector.labels")


def get_class_weights(df: pd.DataFrame) -> dict:
    from collections import Counter
    counts = Counter(df["label"])
    total  = len(df)
    return {cls: total / (len(counts) * cnt) for cls, cnt in counts.items()}


def _log_summary(df: pd.DataFrame, timeframe: str = None) -> None:
    signals   = (df["signal"] != 0).sum()
    buy_sigs  = (df["signal"] ==  1).sum()
    sell_sigs = (df["signal"] == -1).sum()
    winners   = (df["label"]  ==  1).sum()
    tp_hits   = (df["trade_outcome"] ==  1).sum()
    sl_hits   = (df["trade_outcome"] == -1).sum()

    buy_wins  = ((df["label"] == 1) & (df["signal_direction"] ==  1)).sum()
    sell_wins = ((df["label"] == 1) & (df["signal_direction"] == -1)).sum()

    win_rate = tp_hits / max(signals, 1) * 100
    tf_tag   = f"{timeframe} " if timeframe else ""
    logger.info(
        f"{tf_tag}Labels | signals={signals} "
        f"(buys={buy_sigs} sells={sell_sigs}) | "
        f"winners={winners} (buy_wins={buy_wins} sell_wins={sell_wins}) | "
        f"win_rate={win_rate:.1f}% TP={tp_hits} SL={sl_hits}"
    )
