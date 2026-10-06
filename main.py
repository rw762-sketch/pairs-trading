"""One entry point: K-means -> every within-cluster cointegration test -> backtest."""

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from cointegration import screen_pairs
from config import Config
from data_fetcher import load_prices, split_prices, reserve_testing_period
from evaluation import evaluate_pairs
from selection import apply_filters
from stock_clustering import cluster_stocks, within_cluster_pairs
from reporting import write_report

ROOT = Path(__file__).resolve().parent


def run_pipeline(prices, config, output, source=None):
    started = perf_counter()
    training, later, excluded = split_prices(
        prices, config.train_fraction, config.lookback
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    labels, clustering = cluster_stocks(training, config.clusters, config.seed)
    family = within_cluster_pairs(labels)
    print(
        f"{len(prices.columns)} requested stocks; {len(training.columns)} training-eligible; "
        f"{len(family):,} within-cluster pairs. Testing EVERY pair.",
        flush=True,
    )
    with threadpool_limits(limits=1):
        panel, diagnostics = screen_pairs(
            training, family, config.pvalue, config.correction, progress=True
        )
    panel = apply_filters(panel, training, config)
    labels.to_csv(output / "cluster-assignments.csv", index_label="Ticker")
    panel.to_csv(output / "cointegration-tests.csv", index=False)
    diagnostics.to_csv(output / "stock-diagnostics.csv", index=False)
    selected = panel[panel.Selected]
    selected.to_csv(output / "selected-pairs.csv", index=False)
    portfolio, signals, pair_metrics = evaluate_pairs(
        prices, later.index, selected, config, output
    )
    metrics = portfolio["metrics"]
    assert np.isclose(
        portfolio["trades"].Net_PnL.sum(),
        metrics["Final_Portfolio_Value"] - config.capital,
        atol=1e-7,
    )
    portfolio["equity_curve"].to_csv(
        output / "portfolio-account.csv", index_label="Date"
    )
    portfolio["trades"].to_csv(output / "portfolio-trades.csv", index=False)
    pd.DataFrame([metrics]).to_csv(output / "portfolio-metrics.csv", index=False)
    pair_metrics.to_csv(output / "pair-metrics.csv", index=False)
    total = len(training.columns) * (len(training.columns) - 1) // 2
    metadata = {
        "generated_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
        "config": asdict(config),
        "source": source or {},
        "requested_stocks": len(prices.columns),
        "training_eligible_stocks": len(training.columns),
        "excluded_training_tickers": excluded,
        "training_start": str(training.index[0].date()),
        "training_end": str(training.index[-1].date()),
        "evaluation_start": str(later.index[0].date()),
        "evaluation_end": str(later.index[-1].date()),
        "history_closes": len(prices),
        "training_closes": len(training),
        "evaluation_closes": len(later),
        "clustering": clustering,
        "all_possible_pairs": total,
        "tested_pairs": len(panel),
        "failed_tests": int(panel.Status.eq("failed").sum()),
        "selected_pairs": len(selected),
        "cash_sleeves": int(pair_metrics.Status.str.startswith("cash").sum()),
        "runtime_seconds": perf_counter() - started,
        "code_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in ROOT.glob("*.py")
        },
    }
    (output / "run.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    write_report(output, metadata, portfolio, pair_metrics, selected, signals)
    print(
        f"Net return {metrics['Total_Return']:.2%} | Sharpe {metrics['Sharpe_Ratio']:.3f} | "
        f"Max drawdown {metrics['Max_Drawdown']:.2%} | {metrics['Num_Trades']} trades",
        flush=True,
    )
    print(f"Results: {output / 'report.html'}", flush=True)
    return {
        "panel": panel,
        "labels": labels,
        "portfolio": portfolio,
        "metadata": metadata,
        "signals": signals,
    }


def main():
    from run_selected import load_selected_strategy, run_selected_strategy

    defaults = Config(**load_selected_strategy()["config"])
    parser = argparse.ArgumentParser(description=__doc__)
    for key, value in asdict(defaults).items():
        parser.add_argument(
            "--" + key.replace("_", "-"), type=type(value), default=value
        )
    parser.add_argument(
        "--reserve-test-closes", type=int, default=252,
        help="Reserve trailing closes from development (default 252). Previously inspected dates are not fresh validation.",
    )
    parser.add_argument(
        "--mode", choices=["selected", "walk-forward", "holdout"], default="selected"
    )
    parser.add_argument(
        "--screen-cache",
        type=Path,
        help="Reuse a prior walk-forward run with matching input and screening hashes.",
    )
    parser.add_argument("--universe", type=Path, default=ROOT / "data/universe.json")
    parser.add_argument("--cache", type=Path, default=ROOT / "data/cache")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--refresh", action="store_true", help="Redownload adjusted closes from Yahoo."
    )
    args = parser.parse_args()
    config = Config(**{k: getattr(args, k) for k in asdict(defaults)})
    if args.mode == "selected":
        if config != defaults or args.reserve_test_closes != 252 or args.refresh or args.screen_cache:
            parser.error("Selected mode pins the retained strategy and price snapshot. Use --mode walk-forward or --mode holdout for other experiments.")
        run_selected_strategy(args.cache, args.output)
        return
    if (
        config.clusters < 1
        or config.correction not in {"raw", "holm"}
        or config.capital <= 0
        or config.max_hold < 0
    ):
        parser.error(
            "Require positive clusters/capital, nonnegative max-hold (0 disables it); correction raw or holm."
        )
    if (
        not 0 <= config.exit < config.entry < config.stop
        or min(config.fee, config.borrow) < 0
    ):
        parser.error("Require 0 <= exit < entry < stop and nonnegative costs.")
    prices, source = load_prices(
        args.universe, args.cache, config.start, config.end, args.refresh
    )
    source["cache_sha256"] = hashlib.sha256(
        (args.cache / "adjusted-prices.pkl").read_bytes()
    ).hexdigest()
    source["universe_sha256"] = hashlib.sha256(args.universe.read_bytes()).hexdigest()
    try:
        prices, reservation = reserve_testing_period(
            prices, args.reserve_test_closes, config.formation
        )
    except ValueError as error:
        parser.error(str(error))
    source = dict(source, test_reservation=reservation)
    print(f"Reserved {reservation['reserved_closes']} closes: "
          f"{reservation['reserved_start']} to {reservation['reserved_end']}. "
          "Excluded from development; already-inspected history is not a fresh holdout.", flush=True)
    timestamp = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d_%H-%M-%S")
    destination = args.output / timestamp
    number = 1
    while destination.exists():
        destination = args.output / f"{timestamp}-{number:02d}"
        number += 1
    if (
        not 0 < config.half_life_min <= config.half_life_max
        or config.min_crossings < 0
        or not 0 < config.max_pair_weight <= 1
    ):
        parser.error("Invalid recovery bounds, crossings or allocation cap.")
    if config.formation < max(60, config.lookback + 1) or config.trading_window < 2:
        parser.error(
            "Need sufficient preceding formation history and at least two trading closes."
        )
    if args.mode == "walk-forward":
        from walk_forward import run_walk_forward

        run_walk_forward(prices, config, destination, source, args.screen_cache)
    else:
        run_pipeline(prices, config, destination, source)
    (args.output / "LATEST.txt").write_text(str(destination.resolve()) + "\n")
    (args.output / "index.html").write_text(
        f'<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url={destination.name}/report.html"><a href="{destination.name}/report.html">Open latest report</a>'
    )


if __name__ == "__main__":
    main()
