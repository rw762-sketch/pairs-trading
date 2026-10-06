"""Reproducible defaults for the clustered pairs-trading pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    start: str = "2021-09-17"
    end: str = "2026-09-18"  # Yahoo's end date is exclusive.
    train_fraction: float = 0.7
    clusters: int = 10
    seed: int = 42
    pvalue: float = 0.05
    correction: str = (
        "raw"  # Exploratory raw p-values; use 'holm' for family correction.
    )
    lookback: int = 60
    entry: float = 2.0
    exit: float = 0.5
    stop: float = 3.5
    max_hold: int = 0  # Zero disables the per-trade time limit.
    fee: float = 0.001
    borrow: float = 0.02
    capital: float = 100000.0
    half_life_min: float = 1.0
    half_life_max: float = 60.0
    min_crossings: float = 12.0  # Annualized historical mean crossings.
    max_pair_weight: float = 0.20
    formation: int = 252
    trading_window: int = 126
