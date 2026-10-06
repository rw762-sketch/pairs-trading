"""Apply the frozen current rule to an earlier calendar year, without tuning.

This is a retrospective robustness check, not untouched final validation. Pair
selection and allocation are recomputed using only dates before the test year.
The current constituent list still introduces survivorship bias.
"""

from dataclasses import asdict, replace
from datetime import datetime
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo
import argparse
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from threadpoolctl import threadpool_limits

from allocation import persistence_weights
from cointegration import screen_pairs
from config import Config
from evaluation import evaluate_pairs
from run_optimization import plot_account
from run_persistence import persistent_mask
from selection import apply_filters
from stock_clustering import cluster_stocks, within_cluster_pairs
from walk_forward import schedule_windows


ROOT = Path(__file__).resolve().parent


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _window_record(number, dates):
    return {
        "window": number,
        "formation_start": str(dates[0].date()),
        "formation_end": str(dates[-1].date()),
        "formation_closes": len(dates),
    }


def _historical_pvalues(development, candidates, formations):
    columns = [f"P_Window_{number}" for number in range(1, 7)]
    values = pd.DataFrame(index=pd.Index(candidates.Pair, name="Pair"),
                          columns=columns, dtype=float)
    errors = []
    with threadpool_limits(limits=1):
        for number, dates in enumerate(formations, 1):
            training = development.loc[dates]
            missing = failed = 0
            for row in candidates.itertuples():
                observations = training[[row.Ticker1, row.Ticker2]].to_numpy(dtype=float)
                value = np.nan
                reason = ""
                if not np.isfinite(observations).all() or (observations <= 0).any():
                    missing += 1
                    reason = "missing, nonfinite or nonpositive training prices"
                else:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            value = float(coint(observations[:, 0], observations[:, 1],
                                                trend="c", autolag="aic")[1])
                        if not np.isfinite(value):
                            raise ValueError("Nonfinite cointegration p-value")
                    except (ValueError, np.linalg.LinAlgError) as exc:
                        value = np.nan
                        failed += 1
                        reason = str(exc)
                values.loc[row.Pair, columns[number - 1]] = value
                if reason:
                    errors.append({"Pair": row.Pair, "Window": number, "Error": reason})
            print(f"Formation {number}/6: {len(candidates)} candidate tests; "
                  f"{missing} unavailable and {failed} failed", flush=True)
    return values, pd.DataFrame(errors, columns=["Pair", "Window", "Error"])


def _write_report(output, plan, selected, metrics):
    gross = metrics["Net_PnL"] + metrics["Total_Costs"]
    summary = pd.DataFrame([
        ("Selected pairs", len(selected)),
        ("Completed trades", metrics["Num_Trades"]),
        ("Net return after modeled costs", f"{metrics['Total_Return']:.2%}"),
        ("Gross return before modeled costs", f"{gross / metrics['Initial_Capital']:.2%}"),
        ("Net profit", f"${metrics['Net_PnL']:,.2f}"),
        ("Gross profit", f"${gross:,.2f}"),
        ("Transaction costs", f"${metrics['Total_Transaction_Costs']:,.2f}"),
        ("Short borrowing costs", f"${metrics['Total_Borrow_Costs']:,.2f}"),
        ("Total modeled costs", f"${metrics['Total_Costs']:,.2f}"),
        ("Ending account value", f"${metrics['Final_Portfolio_Value']:,.2f}"),
        ("Maximum drawdown", f"{metrics['Max_Drawdown']:.2%}"),
        ("Annualized volatility", f"{metrics['Volatility']:.2%}"),
        ("Sharpe ratio (zero risk-free rate)", f"{metrics['Sharpe_Ratio']:.3f}"),
        ("Win rate", f"{metrics['Win_Rate']:.2%}"),
    ], columns=["Metric", "Value"])
    table = selected[["Pair", "Pass_Count", "Mean_P", "Weight", "Raw_P", "Half_Life"]]
    files = ["plan.json", "performance.csv", "selected-pairs.csv", "allocation.csv",
             "selection-audit.csv", "persistence-tests.csv", "persistence-errors.csv",
             "final-formation-tests.csv", "stock-diagnostics.csv", "cluster-assignments.csv",
             "eligibility-audit.csv", "pair-metrics.csv", "trades.csv", "equity.csv"]
    dates = plan["test_period"]
    rule = plan["config"]
    text = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<title>{plan["year"]} historical robustness test</title>'
        '<style>body{font:16px system-ui;max-width:1050px;margin:40px auto;padding:0 20px;'
        'line-height:1.55}table{border-collapse:collapse}td,th{padding:7px 10px;'
        'border:1px solid #ddd;text-align:left}img{width:100%}code{overflow-wrap:anywhere}</style>'
        '</head><body>'
        f'<h1>{plan["year"]} historical robustness test</h1>'
        f'<p>Trading {dates["start"]} through {dates["end"]}. '
        f'All fitting, pair selection and fixed weights use prices before {plan["year"]}-01-01. '
        'The current strategy settings were frozen before this run, with no optimization on '
        'this year. Pair identities were selected again from earlier training data.</p>'
        '<p>This is a retrospective robustness check, not untouched final validation. '
        'The rule was developed after inspecting later returns. The stock universe is the '
        'current constituent list applied retrospectively, so survivorship bias remains. '
        'Yahoo adjusted closes are the present downloaded historical snapshot. '
        'A selected sleeve with unavailable evaluation prices aborts the run; '
        'pairs and weights are never changed using test-year availability.</p>'
        f'<p>{plan["counts"]["within_cluster_tests"]:,} within-cluster cointegration tests; '
        f'{plan["counts"]["base_candidates"]} final-formation candidates; '
        f'{plan["counts"]["persistent_candidates"]} pass at least '
        f'{plan["required_passes"]}/6 earlier formations including the latest; '
        f'{len(selected)} remain after recovery and shared-stock filters.</p>'
        f'<p>Fixed rules: p &lt; {rule["pvalue"]}; entry |z| between {rule["entry"]} '
        f'and {rule["stop"]}; convergence exit at {rule["exit"]}; adverse stop {rule["stop"]}; '
        f'prior {rule["lookback"]} closes for z-score; fixed pre-test hedge ratios; '
        f'half-life {rule["half_life_min"]}–{rule["half_life_max"]} closes; '
        f'at least {rule["min_crossings"]} annualized mean crossings; no shared stocks. '
        'No holding timeout. Signals execute next close and remaining positions are '
        'liquidated at the last test close.</p>'
        '<p>Weights are proportional to passing periods divided by mean p-value across '
        'all six formations (missing p-values count as 1; denominator floor 0.000001). '
        'All selected sleeve budgets sum to 100%, without a pair allocation cap; inactive '
        'sleeves remain cash. No bank interest is credited to cash.</p>'
        f'<p>Modeled transaction cost: {rule["fee"]:.2%} of gross traded notional on '
        f'each entry and exit, across both legs. Annual short borrow rate: {rule["borrow"]:.2%}, '
        'accrued over calendar days. These are assumptions rather than observed broker charges.</p>'
        '<h2>Performance</h2>' + summary.to_html(index=False, escape=True) +
        '<h2>Selected pairs and allocations</h2>' + table.to_html(index=False, escape=True) +
        '<img src="cumulative-return.png" alt="Calendar-year cumulative portfolio return">'
        '<h2>Development formations</h2>' +
        pd.DataFrame(plan["formation_windows"]).to_html(index=False) +
        '<h2>Reproducible records</h2><ul>' +
        ''.join(f'<li><a href="{escape(name)}">{escape(name)}</a></li>' for name in files) +
        '</ul></body></html>'
    )
    (output / "report.html").write_text(text)


def run_historical_year(year=2021, cache=None, output_root=ROOT / "results",
                        strategy_path=ROOT / "selected_strategy.json"):
    """Recompute training-only selection; preserve the active strategy replay."""
    if not isinstance(year, int) or year < 1900 or year > 9998:
        raise ValueError("Require a valid calendar test year")
    cache = Path(cache) if cache else ROOT / "data" / f"cache-historical-{year}"
    price_file = cache / "adjusted-prices.pkl"
    strategy_path = Path(strategy_path)
    chosen = json.loads(strategy_path.read_text())
    config = replace(Config(**chosen["config"]), start=f"{year - 4}-01-01",
                     end=f"{year + 1}-01-01")
    if (chosen["required_passes"] != 3 or not chosen["require_latest"] or
            chosen["allocation_rule"]["method"] != "pass_count_over_mean_p"):
        raise ValueError("This runner requires the current frozen 3-of-6 allocation rule")
    prices = pd.read_pickle(price_file)
    if (not isinstance(prices.index, pd.DatetimeIndex) or prices.index.hasnans or
            not prices.index.is_unique or not prices.index.is_monotonic_increasing or
            prices.columns.duplicated().any()):
        raise ValueError("Require unique chronological price dates and ticker columns")
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year + 1, month=1, day=1)
    development = prices.loc[prices.index < start].copy()
    dates = prices.index[(prices.index >= start) & (prices.index < end)]
    if len(dates) < 2 or len(development) < config.formation:
        raise ValueError("Need pre-test formation history and at least two test closes")
    schedule = schedule_windows(development.index, config.formation, config.trading_window)
    if len(schedule) != 6:
        raise ValueError(f"Need exactly six earlier formations; found {len(schedule)}. "
                         f"Use approximately four calendar years before {year}.")
    formations = [formation for formation, _ in schedule]
    if any(formation[-1] >= start for formation in formations):
        raise AssertionError("Formation overlaps the test year")
    train = development.tail(config.formation)
    valid = np.isfinite(train).all() & (train > 0).all()
    eligibility = pd.DataFrame({"Ticker": train.columns, "Training_Eligible": valid.to_numpy()})
    train = train.loc[:, sorted(train.columns[valid])]
    if train.shape[1] < 2:
        raise ValueError("Fewer than two stocks have complete positive final training prices")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d_%H-%M-%S")
    output = output_root / f"{year}_{stamp}"
    output.mkdir()
    labels, clustering = cluster_stocks(train, config.clusters, config.seed)
    family = within_cluster_pairs(labels)
    print(f"Historical {year}: {len(train.columns)} training-eligible stocks; "
          f"{len(family):,} within-cluster cointegration tests", flush=True)
    with threadpool_limits(limits=1):
        panel, diagnostics = screen_pairs(train, family, config.pvalue, config.correction,
                                          progress=True)
    candidates = panel.loc[panel.Selected].copy()
    values, errors = _historical_pvalues(development, candidates, formations)
    keep = persistent_mask(values, config.pvalue, chosen["required_passes"],
                           chosen["require_latest"])
    persistence = values.copy()
    persistence["Pass_Count"] = values.lt(config.pvalue).sum(axis=1)
    persistence["Mean_P"] = values.fillna(1.0).mean(axis=1)
    persistence["Persistent"] = keep
    audit = panel.copy()
    audit["Final_Formation_Selected"] = audit.Selected
    audit["Persistence_Eligible"] = audit.Pair.isin(keep.index[keep])
    audit["Selected"] = audit.Selected & audit.Persistence_Eligible
    audit = apply_filters(audit, train, config)
    selected = audit.loc[audit.Selected].copy()
    if len(selected):
        weights = persistence_weights(values.loc[selected.Pair], config.pvalue,
                                      chosen["allocation_rule"]["pvalue_floor"])
        np.testing.assert_allclose(weights.Weight.sum(), 1.0)
    else:
        weights = pd.DataFrame(columns=["Pass_Count", "Mean_P", "Allocation_Score", "Weight"],
                               index=pd.Index([], name="Pair"))
    for column in weights.columns:
        selected[column] = selected.Pair.map(weights[column])
    audit["Weight"] = audit.Pair.map(weights.Weight).fillna(0.0)
    print(f"{int(keep.sum())} persistent candidates; {len(selected)} selected pairs", flush=True)
    info_file = cache / "price-cache-info.json"
    plan = {
        "recorded_before_simulation": datetime.now(ZoneInfo("America/New_York")).isoformat(),
        "year": year,
        "experiment": "historical robustness with frozen current rules",
        "fresh_holdout": False,
        "tuning": "None on this year; current selected settings held fixed before simulation",
        "config": asdict(config),
        "config_date_note": "Only acquisition start/end adapted to the historical date range; "
                            "all strategy parameters retained from the selected specification.",
        "original_selected_strategy": chosen,
        "strategy_path": str(strategy_path.resolve()),
        "strategy_sha256": _sha256(strategy_path),
        "required_passes": chosen["required_passes"],
        "require_latest": chosen["require_latest"],
        "allocation_rule": chosen["allocation_rule"],
        "data_adaptation": "Recompute all pairs, fits and weights from earlier data; "
                           "do not reuse pair names or returns from the later selected run.",
        "no_test_prices_in_selection_or_weights": True,
        "causality_note": "All selections and weights are saved before examining test prices. "
                          "After freezing, validate full retained-sleeve evaluation/warmup "
                          "coverage and abort on gaps. Do not remove or reweight pairs. "
                          "Signals use past rolling moments and next-close execution.",
        "price_cache": str(price_file.resolve()),
        "price_sha256": _sha256(price_file),
        "cache_metadata": json.loads(info_file.read_text()) if info_file.exists() else {},
        "development_period": {"start": str(development.index[0].date()),
                               "end": str(development.index[-1].date()),
                               "closes": len(development)},
        "final_formation": _window_record("final", train.index),
        "formation_windows": [_window_record(n, formation)
                              for n, formation in enumerate(formations, 1)],
        "test_period": {"start": str(dates[0].date()), "end": str(dates[-1].date()),
                        "closes": len(dates), "end_exclusive": str(end.date())},
        "clustering": clustering,
        "counts": {"training_eligible_stocks": len(train.columns),
                   "within_cluster_tests": len(family), "base_candidates": len(candidates),
                   "persistent_candidates": int(keep.sum()), "selected_pairs": len(selected)},
        "limitations": ["Rule developed after inspecting later-year returns; this is retrospective",
                        "Current constituent universe applied to the earlier period: survivorship bias",
                        "Present Yahoo adjusted historical prices, not point-in-time vintage data",
                        "Exploratory raw p-values and overlapping formation windows",
                        "Costs assumed; no interest credited to cash"],
        "code_sha256": {name: _sha256(ROOT / name) for name in [
            "run_historical_year.py", "cointegration.py", "stock_clustering.py", "selection.py",
            "allocation.py", "evaluation.py", "signal_generation.py", "backtester.py"]},
    }
    # Both the frozen rule and all selected parameters/weights are saved before evaluation.
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    panel.to_csv(output / "final-formation-tests.csv", index=False)
    diagnostics.to_csv(output / "stock-diagnostics.csv", index=False)
    labels.to_csv(output / "cluster-assignments.csv", index_label="Ticker")
    eligibility.to_csv(output / "eligibility-audit.csv", index=False)
    persistence.to_csv(output / "persistence-tests.csv", index_label="Pair")
    errors.to_csv(output / "persistence-errors.csv", index=False)
    selected.to_csv(output / "selected-pairs.csv", index=False)
    weights.to_csv(output / "allocation.csv", index_label="Pair")
    audit.to_csv(output / "selection-audit.csv", index=False)
    # Discard dates after the test year before passing data to the shared evaluator.
    history = prices.loc[prices.index < end]
    for row in selected.itertuples():
        coverage = history[[row.Ticker1, row.Ticker2]].tail(len(dates) + config.lookback)
        if (len(coverage) != len(dates) + config.lookback or
                not np.isfinite(coverage.to_numpy(dtype=float)).all() or
                (coverage <= 0).any().any()):
            raise ValueError(f"Frozen selected pair {row.Pair} has unavailable test/warmup "
                             "prices. Refusing to drop/reweight it using future availability. "
                             f"Pre-evaluation records remain at {output.resolve()}.")
    portfolio, _, pair_metrics = evaluate_pairs(history, dates, selected, config, output)
    metrics = portfolio["metrics"].copy()
    metrics["Net_PnL"] = metrics["Final_Portfolio_Value"] - config.capital
    metrics["Gross_PnL"] = metrics["Net_PnL"] + metrics["Total_Costs"]
    metrics["Gross_Return"] = metrics["Gross_PnL"] / config.capital
    metrics["Selected_Pairs"] = len(selected)
    np.testing.assert_allclose(portfolio["trades"].Gross_PnL.sum(), metrics["Gross_PnL"],
                               atol=1e-7)
    pd.DataFrame([metrics]).to_csv(output / "performance.csv", index=False)
    portfolio["equity_curve"].to_csv(output / "equity.csv", index_label="Date")
    portfolio["trades"].to_csv(output / "trades.csv", index=False)
    pair_metrics.to_csv(output / "pair-metrics.csv", index=False)
    plot_account(portfolio, output / "cumulative-return.png",
                 f"{year} historical robustness: frozen strategy rules")
    _write_report(output, plan, selected, metrics)
    print(f"Historical {year}: {metrics['Total_Return']:.2%} net; "
          f"{metrics['Gross_Return']:.2%} gross; {metrics['Num_Trades']} trades; "
          f"results: {output.resolve()}", flush=True)
    return {"portfolio": portfolio, "selected": selected, "metrics": metrics, "output": output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--cache", type=Path, help="Separate historical adjusted-price cache")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results")
    parser.add_argument("--strategy", type=Path, default=ROOT / "selected_strategy.json")
    args = parser.parse_args()
    run_historical_year(args.year, args.cache, args.output_root, args.strategy)


if __name__ == "__main__":
    main()
