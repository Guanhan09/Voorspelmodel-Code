"""Entry point voor hertraining: `python -m backend.train`.

Roept de bestaande trainer aan. Respecteert REUSE_DATASET (0 = dataset opnieuw
bouwen omdat de features veranderd zijn).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from ml_trainer import train_and_evaluate

if __name__ == "__main__":
    train_and_evaluate()
