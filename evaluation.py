"""A shared next-close evaluator for static and repeated chronological windows."""

from pathlib import Path
import numpy as np
import pandas as pd
from backtester import (
    PairBacktester,
    TRADE_COLUMNS,
    combine_portfolio_results,
    performance_metrics,
)
from signal_generation import PairSignalGenerator


def evaluate_pairs(history, dates, selected, config, output=None):
    results = {}
    signals = {}
    rows = []
    weights = {}
    if output is not None:
        files = Path(output) / "pairs"
        files.mkdir(parents=True, exist_ok=True)
    budget = config.capital
    for row in selected.itertuples():
        weight = float(getattr(row, "Weight", 1 / max(len(selected), 1)))
        prices = history.loc[: dates[-1], [row.Ticker1, row.Ticker2]].tail(
            len(dates) + config.lookback
        )
        if np.isfinite(prices.to_numpy()).all() and (prices > 0).all().all():
            gen = PairSignalGenerator(
                prices,
                row.Ticker1,
                row.Ticker2,
                hedge_ratio=row.Beta,
                lookback=config.lookback,
            )
            gen.calculate_zscore()
            for attr in (
                "price_data",
                "spread",
                "z_score",
                "moving_mean",
                "moving_std",
            ):
                setattr(gen, attr, getattr(gen, attr).loc[dates])
            sig = gen.get_signals(config.entry, config.exit, stop_zscore=config.stop)
            engine = PairBacktester(
                sig,
                initial_capital=budget,
                transaction_cost=config.fee,
                annual_borrow_rate=config.borrow,
                max_holding_period=config.max_hold or None,
            )
            curve = engine.run_backtest()
            trades = engine.get_trades_dataframe()
            metrics = engine.get_performance_metrics()
            signals[row.Pair] = sig
            status = "evaluated"
            if output is not None:
                sig.to_csv(files / f"{row.Pair}-signals.csv", index_label="Date")
                curve.to_csv(files / f"{row.Pair}-account.csv", index_label="Date")
                trades.to_csv(files / f"{row.Pair}-trades.csv", index=False)
        else:
            curve = pd.DataFrame({"Portfolio_Value": budget}, index=dates)
            trades = pd.DataFrame(columns=TRADE_COLUMNS)
            metrics = performance_metrics(curve.Portfolio_Value, trades, budget)
            status = "cash: unavailable evaluation prices"
        assert np.isclose(
            trades.Net_PnL.sum(), curve.Portfolio_Value.iloc[-1] - budget, atol=1e-7
        )
        results[row.Pair] = {
            "equity_curve": curve,
            "trades": trades,
            "metrics": metrics,
            "initial_capital": budget,
        }
        weights[row.Pair] = weight
        rows.append(
            {
                "Pair": row.Pair,
                "Status": status,
                "Weight": weight,
                "Allocated_Profit": metrics["Total_Return"] * weight * budget,
                **metrics,
            }
        )
    cash = max(0.0, 1 - sum(weights.values()))
    if cash > 1e-12 or not results:
        curve = pd.DataFrame({"Portfolio_Value": budget}, index=dates)
        trades = pd.DataFrame(columns=TRADE_COLUMNS)
        results["Cash"] = {
            "equity_curve": curve,
            "trades": trades,
            "metrics": performance_metrics(curve.Portfolio_Value, trades, budget),
            "initial_capital": budget,
        }
        weights["Cash"] = cash if weights else 1.0
    portfolio = combine_portfolio_results(
        results, initial_capital=budget, weights=weights
    )
    assert np.isclose(
        portfolio["trades"].Net_PnL.sum(),
        portfolio["equity_curve"].iloc[-1] - budget,
        atol=1e-7,
    )
    return (
        portfolio,
        signals,
        pd.DataFrame(
            rows,
            columns=[
                "Pair",
                "Status",
                "Weight",
                "Allocated_Profit",
                *portfolio["metrics"].keys(),
            ],
        ),
    )
