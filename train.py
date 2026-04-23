#!/usr/bin/env python3
"""
train.py — Entry point: runs the full training pipeline.

Usage:
    python train.py --timeframes 5min 15min
    python train.py --timeframes 5min 15min --no-smote
    python train.py --timeframes 5min 15min --tune
"""

from models.trainer import main

if __name__ == "__main__":
    main()
