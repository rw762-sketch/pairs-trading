"""Non-overlapping chronological trading windows with fresh training-only screens."""

from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib, json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from backtester import TRADE_COLUMNS, performance_metrics
from cointegration import screen_pairs
from evaluation import evaluate_pairs
from selection import apply_filters
from stock_clustering import cluster_stocks, within_cluster_pairs
from reporting import write_walk_forward_report


def schedule_windows(index, formation=252, trading_window=126):
    if (
        not isinstance(index, pd.DatetimeIndex)
        or not index.is_unique
        or not index.is_monotonic_increasing
    ):
        raise ValueError("Dates must be unique and chronological.")
    if formation < 3 or trading_window < 2 or len(index) <= formation:
        raise ValueError("Insufficient formation/trading history.")
    return [
        (
            index[start - formation : start],
            index[start : min(start + trading_window, len(index))],
        )
        for start in range(formation, len(index), trading_window)
    ]


def exclude_previous_losses(selected, previous_losses, max_weight):
    """Drop unordered pairs with negative net P&L in only the preceding window."""
    result = selected.copy()
    result["Previous_Window_Loser"] = [
        tuple(sorted((r.Ticker1, r.Ticker2))) in previous_losses
        for r in result.itertuples()
    ]
    eligible = result.loc[~result.Previous_Window_Loser].copy()
    eligible["Weight"] = min(1 / len(eligible), max_weight) if len(eligible) else 0.0
    return eligible, result


def previous_window_losers(trades):
    if trades.empty:
        return set()
    grouped = trades.groupby(["Ticker1", "Ticker2"]).Net_PnL.sum()
    # Net_PnL already includes fees and short borrowing.
    totals = {}
    for (a, b), value in grouped.items():
        key = tuple(sorted((a, b)))
        totals[key] = totals.get(key, 0.0) + value
    return {key for key, value in totals.items() if value < 0}


def run_walk_forward(prices, config, output, source=None, screen_cache=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    windows = schedule_windows(prices.index, config.formation, config.trading_window)
    plan = {
        "recorded_before_simulation": datetime.now(
            ZoneInfo("America/New_York")
        ).isoformat(),
        "config": asdict(config),
        "source": source or {},
        "fresh_holdout": False,
        "variants": ["baseline", "filtered", "previous_loss_filter"],
        "selection_tuning": "None; thresholds fixed before simulation.",
        "previous_loss_rule": "On top of filtered selection, exclude unordered pairs with aggregate net P&L < 0 in the immediately preceding window of this variant. New, untraded, skipped and zero-P&L pairs remain eligible. Reallocate equally subject to the existing cap. No permanent blacklist.",
        "window_count": len(windows),
        "code_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
        "windows": [
            {
                "formation_start": str(f[0].date()),
                "formation_end": str(f[-1].date()),
                "trading_start": str(t[0].date()),
                "trading_end": str(t[-1].date()),
            }
            for f, t in windows
        ],
    }
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    if screen_cache is not None:
        screen_cache = Path(screen_cache)
        previous = json.loads((screen_cache / "plan.json").read_text())
        if (
            previous["config"] != plan["config"]
            or previous["windows"] != plan["windows"]
        ):
            raise ValueError("Screen cache settings or chronological windows differ.")
        for key in ["cache_sha256", "universe_sha256"]:
            if (
                not plan["source"].get(key)
                or previous["source"].get(key) != plan["source"][key]
            ):
                raise ValueError("Screen cache input hashes differ.")
        for key in ["cointegration.py", "stock_clustering.py"]:
            if previous["code_sha256"][key] != plan["code_sha256"][key]:
                raise ValueError("Screening code changed; recompute pair tests.")
    storage = {
        name: {"capital": config.capital, "curves": [], "trades": []}
        for name in plan["variants"]
    }
    rows = []
    previous_losses = set()
    for number, (formation, dates) in enumerate(windows, 1):
        assert formation[-1] < dates[0]
        history = prices.loc[: dates[-1]]
        train = history.loc[formation]
        valid = np.isfinite(train).all() & (train > 0).all()
        train = train.loc[:, sorted(train.columns[valid])]
        labels, _ = cluster_stocks(train, config.clusters, config.seed)
        family = within_cluster_pairs(labels)
        print(
            f"Window {number}/{len(windows)}: {dates[0].date()} to {dates[-1].date()}; {len(family):,} cointegration tests",
            flush=True,
        )
        cached = screen_cache / "windows" / f"{number:02d}" if screen_cache else None
        if cached and (cached / "cointegration-tests.csv").exists():
            panel = pd.read_csv(cached / "cointegration-tests.csv")
            expected = {f"{a}-{b}" for a, b in family}
            if set(panel.Pair) != expected or len(panel) != len(family):
                raise ValueError("Cached pair family differs.")
            panel["Selected"] = panel.Base_Selected
            diagnostics = pd.read_csv(cached / "stock-diagnostics.csv")
            print("  Reusing verified formation-screen results.", flush=True)
        else:
            with threadpool_limits(limits=1):
                panel, diagnostics = screen_pairs(
                    train, family, config.pvalue, config.correction
                )
        panel = apply_filters(panel, train, config)
        directory = output / "windows" / f"{number:02d}"
        directory.mkdir(parents=True)
        panel.to_csv(directory / "cointegration-tests.csv", index=False)
        diagnostics.to_csv(directory / "stock-diagnostics.csv", index=False)
        labels.to_csv(directory / "cluster-assignments.csv", index_label="Ticker")
        for variant in storage:
            selected = panel[
                panel.Base_Selected if variant == "baseline" else panel.Selected
            ].copy()
            if variant == "baseline":
                selected["Weight"] = 1 / max(len(selected), 1)
            excluded_count = 0
            if variant == "previous_loss_filter":
                selected, audit = exclude_previous_losses(
                    selected, previous_losses, config.max_pair_weight
                )
                excluded_count = int(audit.Previous_Window_Loser.sum())
                audit.to_csv(directory / "previous-loss-audit.csv", index=False)
            selected.to_csv(directory / f"{variant}-selected.csv", index=False)
            capital = storage[variant]["capital"]
            if capital <= 0:
                raise ValueError("Account exhausted; stop before the next window.")
            settings = replace(config, capital=capital)
            portfolio, _, _ = evaluate_pairs(
                history,
                dates,
                selected,
                settings,
                directory if variant == "filtered" else None,
            )
            if variant == "previous_loss_filter":
                previous_losses = previous_window_losers(portfolio["trades"])
            curve = portfolio["equity_curve"]
            trades = portfolio["trades"].copy()
            metrics = portfolio["metrics"]
            pd.testing.assert_index_equal(curve.index, dates)
            trades["Window"] = number
            trades["Variant"] = variant
            trades.to_csv(directory / f"{variant}-trades.csv", index=False)
            curve.to_csv(directory / f"{variant}-account.csv", index_label="Date")
            if number < len(windows):
                trades.loc[trades.Exit_Reason.eq("end_of_data"), "Exit_Reason"] = (
                    "window_end"
                )
            storage[variant]["curves"].append(curve)
            if len(trades):
                storage[variant]["trades"].append(trades)
            storage[variant]["capital"] = float(curve.iloc[-1])
            rows.append(
                {
                    "Window": number,
                    "Variant": variant,
                    "Formation_Start": formation[0],
                    "Formation_End": formation[-1],
                    "Trading_Start": dates[0],
                    "Trading_End": dates[-1],
                    "Tested_Pairs": len(panel),
                    "Base_Candidates": int(panel.Base_Selected.sum()),
                    "Recovery_Candidates": int(panel.Recovery_Eligible.sum()),
                    "Selected_Pairs": len(selected),
                    "Previous_Losers_Excluded": excluded_count,
                    "Starting_Capital": capital,
                    "Ending_Capital": curve.iloc[-1],
                    "Cash_Weight": max(0.0, 1 - selected.Weight.sum()),
                    **metrics,
                }
            )
            print(
                f"  {variant}: {metrics['Total_Return']:.2%}; {len(selected)} pairs; {metrics['Num_Trades']} trades",
                flush=True,
            )
    metrics_rows = []
    portfolios = {}
    for name, saved in storage.items():
        curve = pd.concat(saved["curves"]).rename("Portfolio_Value")
        expected = prices.index[config.formation :]
        pd.testing.assert_index_equal(curve.index, expected)
        trades = (
            pd.concat(saved["trades"], ignore_index=True)
            if saved["trades"]
            else pd.DataFrame(columns=TRADE_COLUMNS + ["Window", "Variant"])
        )
        assert np.isclose(
            trades.Net_PnL.sum(), curve.iloc[-1] - config.capital, atol=1e-7
        )
        subset = pd.DataFrame(rows).query("Variant == @name")
        np.testing.assert_allclose(
            subset.Starting_Capital.iloc[1:], subset.Ending_Capital.iloc[:-1]
        )
        metrics = performance_metrics(curve, trades, config.capital)
        metrics_rows.append({"Variant": name, **metrics})
        curve.to_csv(output / f"{name}-account.csv", index_label="Date")
        trades.to_csv(output / f"{name}-trades.csv", index=False)
        portfolios[name] = {"equity_curve": curve, "trades": trades, "metrics": metrics}
    pd.DataFrame(metrics_rows).to_csv(output / "comparison.csv", index=False)
    pd.DataFrame(rows).to_csv(output / "window-results.csv", index=False)
    write_walk_forward_report(output, plan, pd.DataFrame(rows), portfolios)
    print(f"Results: {output / 'report.html'}", flush=True)
    return {"plan": plan, "windows": pd.DataFrame(rows), "portfolios": portfolios}
