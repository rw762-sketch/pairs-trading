"""Build a concise project brief from saved, separately identified experiments.

This is a presentation builder, not a strategy runner. It never fits a model or
changes a return. Figures and their SVG originals sit beside the final report.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import MultipleLocator, PercentFormatter
import numpy as np
import pandas as pd

from research_report_pdf import render_report


ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "results/2026-09-18_17-58-16_complete-research-report_494-stocks"
CLEAN = ROOT / "results/2026-09-18_21-29-27_pairs-trading-clean-report_494-stocks"
PILOT = ROOT / "results/2026-09-18_22-52-31_ols-strategy-improvement_50-stocks"
BLUE, ORANGE, GRAY = "#24567b", "#ae4b13", "#647783"


def artifact(directory: Path, kind: str, extension: str = "csv") -> Path:
    return directory / f"{directory.name}_{kind}.{extension}"


def table(headers, rows):
    def safe(value):
        return str(value).replace("|", r"\|").replace("\n", " ")
    return "\n".join([
        "| " + " | ".join(map(safe, headers)) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *["| " + " | ".join(map(safe, row)) + " |" for row in rows],
    ])


def pct(value):
    return f"{float(value):.2%}"


def dollars(value):
    value = float(value)
    return f"{'-' if value < 0 else ''}${abs(value):,.2f}"


def figure_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.labelsize": 11, "axes.titlesize": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.spines.bottom": False,
        "axes.labelcolor": "#233746", "xtick.color": "#344a59",
        "ytick.color": "#344a59", "savefig.facecolor": "white",
    })


def save_figure(fig, path):
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.14)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)


def baseline_curve(data, path):
    fig, ax = plt.subplots(figsize=(10, 3.15))
    dates = pd.to_datetime(data["Date"])
    ax.plot(dates, data["Cumulative_Return"], color=BLUE, lw=2)
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 3, 5, 7, 9, 11]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_ylabel("Net portfolio return")
    ax.grid(axis="y", color="#e1e7ec", lw=0.7)
    ax.annotate(pct(data["Cumulative_Return"].iloc[-1]),
                (dates.iloc[-1], data["Cumulative_Return"].iloc[-1]),
                xytext=(8, 0), textcoords="offset points", color=BLUE,
                va="center", fontweight="bold")
    ax.set_xlim(dates.iloc[0], dates.iloc[-1] + pd.Timedelta(days=36))
    fig.tight_layout()
    save_figure(fig, path)


def flow_diagram(path, ecm=False):
    if ecm:
        labels = [
            "Prior 252 closes\nFit hedge beta", "Residual spread\ns = A - alpha - beta B",
            "Fit the spread-change model\nLagged spread + lagged change",
            "Require stable dynamics\nk < 0; roots inside 1", "Forecast after execution\ns(t+6) - s(t+1)",
            "Clear the cost hurdle\nThen execute next close",
        ]
        fig, ax = plt.subplots(figsize=(10, 2.55))
        centers = [(0.16, 0.76), (0.50, 0.76), (0.84, 0.76),
                   (0.16, 0.23), (0.50, 0.23), (0.84, 0.23)]
        width, height, font = 0.285, 0.33, 11.5
        arrows = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    else:
        labels = ["Form the pair\nUse earlier prices", "Fit OLS\nEstimate alpha and beta",
                  "Measure the gap\nPrior-window z-score", "Execute next close\nRecord both legs + costs"]
        fig, ax = plt.subplots(figsize=(10, 1.45))
        centers = [(0.125, 0.5), (0.375, 0.5), (0.625, 0.5), (0.875, 0.5)]
        width, height, font = 0.215, 0.64, 11.5
        arrows = [(0, 1), (1, 2), (2, 3)]
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    for (x, y), label in zip(centers, labels):
        patch = FancyBboxPatch((x - width / 2, y - height / 2), width, height,
                              boxstyle="round,pad=0.01", facecolor="#eef4f8",
                              edgecolor="#9cb1bf", lw=1)
        ax.add_patch(patch)
        ax.text(x, y, label, ha="center", va="center", fontsize=font, color="#1e405a", linespacing=1.5)
    for start, end in arrows:
        x1, y1 = centers[start]
        x2, y2 = centers[end]
        if ecm and start == 2:
            # A thin flow line wraps from the top-right box to the bottom-left.
            ax.plot([x1, x1, x2, x2], [y1-height/2, .495, .495, y2+height/2+.03],
                    color=GRAY, lw=1)
            a, b = (x2, y2+height/2+.055), (x2, y2+height/2)
        else:
            a, b = (x1+width/2+.012, y1), (x2-width/2-.012, y2)
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10,
                                    lw=1, color=GRAY))
    fig.tight_layout(pad=0.15)
    save_figure(fig, path)


def pilot_bars(performance, path):
    order = ["baseline", "quality", "entry_filter", "risk_limits", "combined"]
    names = ["Monthly OLS", "Stable beta + recovery", "Entry + cost filter", "Stops + allocation limits", "All three changes"]
    later = performance[performance.Period == "later"].set_index("Variant")
    values = [later.loc[key, "Total_Return"] for key in order]
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.barh(np.arange(5), values, color=BLUE, height=.62)
    ax.set_yticks(np.arange(5), names)
    ax.invert_yaxis()
    ax.axvline(0, color=GRAY, lw=.8)
    ax.set_xlim(min(values)*1.32, .002)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.xaxis.set_major_locator(MultipleLocator(.01))
    ax.set_xlabel("Later-period net total return")
    ax.grid(axis="x", color="#e1e7ec", lw=.7)
    ax.set_axisbelow(True)
    for number, value in enumerate(values):
        ax.text(value - .0006, number, pct(value), ha="right", va="center", fontsize=11, color="#233746")
    fig.tight_layout()
    save_figure(fig, path)


def ecm_curves(directory, path):
    """Read saved accounts without constructing a new performance simulation."""
    files = list(directory.glob("*later*account.csv"))
    if len(files) != 2:
        raise ValueError(f"Expected two later account CSVs in {directory}; got {files}")
    fig, ax = plt.subplots(figsize=(10, 3.1))
    curves = []
    for file in sorted(files):
        frame = pd.read_csv(file)
        date_column = "Date" if "Date" in frame else frame.columns[0]
        value_column = "Account_Value" if "Account_Value" in frame else "Portfolio_Value"
        dates = pd.to_datetime(frame[date_column])
        values = frame[value_column] / 100000 - 1
        is_ecm = "ecm" in file.name.split("_50-stocks_")[-1]
        label = "ECM entry gate" if is_ecm else "Monthly OLS control"
        color, style = (ORANGE, "-") if is_ecm else (BLUE, "--")
        ax.plot(dates, values, color=color, linestyle=style, lw=1.9, label=label)
        curves.append((dates, values, label, color))
    ax.axhline(0, color=GRAY, lw=.8)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 3, 5, 7, 9, 11]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_ylabel("Net portfolio return")
    ax.grid(axis="y", color="#e1e7ec", lw=.7)
    ax.legend(frameon=False, loc="best", fontsize=10)
    fig.tight_layout()
    save_figure(fig, path)


def build_brief(ecm_dir: Path, output_stem: Path | None = None):
    now = datetime.now(ZoneInfo("America/New_York"))
    stamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    if output_stem is None:
        output_stem = ROOT / "output/pdf" / f"{stamp}_pairs_trading_project_brief"
    output_stem = output_stem.resolve()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    md_path, pdf_path = output_stem.with_suffix(".md"), output_stem.with_suffix(".pdf")
    assets = output_stem.parent / f"{output_stem.name}_figures"
    assets.mkdir(parents=True, exist_ok=True)

    def link(path, label):
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return f"[{label}]({os.path.relpath(path, output_stem.parent)})"

    def image(name, caption):
        return f"![{caption}]({os.path.relpath(assets / name, output_stem.parent)})"

    def csv(directory, kind, label):
        return link(artifact(directory, kind), label)

    meta = json.loads(artifact(BASELINE, "run-metadata", "json").read_text())
    baseline = pd.read_csv(artifact(BASELINE, "performance-comparison"))
    training, later = baseline.iloc[0], baseline.iloc[1]
    performance = pd.read_csv(artifact(PILOT, "all-performance"))
    ecm = pd.read_csv(artifact(ecm_dir, "all-performance"))
    # This explicit schema check prevents silently confusing studies or methods.
    required = {"Period", "Variant", "Total_Return", "Num_Trades", "Max_Drawdown", "Total_Costs"}
    if not required.issubset(ecm.columns):
        raise ValueError(f"ECM performance lacks {required - set(ecm.columns)}")
    ecm_variants = list(ecm.Variant.unique())
    ecm_key = next(key for key in ecm_variants if "ecm" in key)
    control_key = next(key for key in ecm_variants if "ecm" not in key)
    ecm_index = ecm.set_index(["Period", "Variant"])
    measured = ecm_index.loc[("later", ecm_key)]
    control = ecm_index.loc[("later", control_key)]
    figure_style()
    flow_diagram(assets / "procedure.png")
    flow_diagram(assets / "ecm-procedure.png", ecm=True)
    baseline_curve(pd.read_csv(artifact(BASELINE, "later-daily-portfolio")), assets / "baseline-return.png")
    pilot_bars(performance, assets / "pilot-returns.png")
    ecm_curves(ecm_dir, assets / "ecm-returns.png")

    baseline_metrics = []
    for label, key, formatter in [
        ("Net total return", "Total_Return", pct),
        ("Annualized return", "Annualized_Return", pct),
        ("Annualized volatility", "Volatility", pct),
        ("Sharpe; zero risk-free rate", "Sharpe_Ratio", lambda x: f"{x:.2f}"),
        ("Maximum drawdown", "Max_Drawdown", pct),
        ("Closed pair trades", "Num_Trades", lambda x: str(int(x))),
        ("Win rate, after costs", "Win_Rate", pct),
        ("Fees + borrowing", "Total_Costs", dollars),
    ]:
        baseline_metrics.append([label, formatter(training[key]), formatter(later[key])])
    variants = [
        ("baseline", "Monthly OLS"), ("quality", "Stable beta + recovery"),
        ("entry_filter", "Entry + cost filter"), ("risk_limits", "Stops + allocation"),
        ("combined", "All three changes"),
    ]
    pilot_rows = []
    for key, label in variants:
        item = performance[performance.Variant == key].set_index("Period")
        pilot_rows.append([label, pct(item.loc["validation_1", "Total_Return"]),
                           pct(item.loc["validation_2", "Total_Return"]),
                           pct(item.loc["later", "Total_Return"]),
                           int(item.loc["later", "Num_Trades"])])
    ecm_rows = []
    for period, label in [("validation_1", "Earlier 1"), ("validation_2", "Earlier 2"), ("later", "Later")]:
        c, m = ecm_index.loc[(period, control_key)], ecm_index.loc[(period, ecm_key)]
        ecm_rows.append([label, pct(c.Total_Return), pct(m.Total_Return),
                         f"{(m.Total_Return-c.Total_Return)*100:+.3f}",
                         f"{int(c.Num_Trades)} / {int(m.Num_Trades)}"])
    no_profit = "The ECM version still loses money in the later period." if measured.Total_Return < 0 else "The ECM version earns a positive later-period return in this experiment."
    question_rows = [
        ["Does correlation establish cointegration?", "No. Return co-movement and a stationary price combination are different properties."],
        ["Did the 52 fixed relationships persist?", "2 of 52 later Engle-Granger retests have raw p < 0.05; these refits are diagnostics, not past trade selectors."],
        ["Are thresholds important?", "All nine baseline later threshold variants lose money: -4.09% to -2.72%."],
        ["Do costs explain the whole loss?", "No. The 494-stock baseline loses 2.36% before costs and 3.62% after costs."],
        ["Is profit concentrated?", "Five winning trades supply 16.7% of positive net trade profit. There are 21 profitable and 31 losing baseline pairs."],
        ["Has live viability been established?", "No. The studies omit executable quotes, impact, margin constraints and borrow recalls; no untouched final period remains."],
    ]
    coverage_rows = [
        ["Candidate table / heatmap", "Clean report pp. 3-4: all 52 pairs", csv(BASELINE, "selected-pairs-statistics", "Pair statistics CSV")],
        ["Cointegration p-value table", "Clean report pp. 3-4", csv(BASELINE, "all-candidate-p-values", "All 5,734 raw candidates CSV")],
        ["Selected spread chart", "Clean report p. 5", csv(BASELINE, "cpt-ess-later-signals", "CPT-ESS signal data CSV")],
        ["Z-score + entry / exit lines", "Clean report p. 5", csv(BASELINE, "aep-duk-later-signals", "AEP-DUK signal data CSV")],
        ["Backtest equity curve", "Clean report p. 2", csv(BASELINE, "later-daily-portfolio", "Daily account CSV")],
        ["Trade log", "Clean report p. 7: sample", csv(BASELINE, "later-trade-log", "Complete later trades CSV")],
        ["Training vs later performance", "Clean report p. 2", csv(BASELINE, "performance-comparison", "Period metrics CSV")],
        ["Sharpe, return, volatility, drawdown", "Clean report p. 2", "Same period metrics CSV"],
        ["Win rate, trade return, turnover", "Clean report p. 2", "Same period metrics CSV"],
        ["Entry / exit sensitivity", "Clean report p. 6", csv(BASELINE, "threshold-sensitivity", "All threshold results CSV")],
        ["Overall cumulative return", "Clean report p. 2; brief p. 3", "Same daily account CSV"],
    ]

    pages = []
    pages.append(f"""# Pairs trading: what the research shows

Project brief | {now:%Y-%m-%d %H:%M %Z} | Cached US stock prices through 2026-09-17

**Research question:** can deviations between related stock prices predict a trade that remains profitable after execution costs?

**Current result:** the 494-stock study returns **{pct(later.Total_Return)}**, a loss of **{dollars(100000-later.Final_Portfolio_Value)}**, in the later period. The five completed 50-stock variants also lose money later. A new linear error-correction entry gate is evaluated separately on pages 5-6.

## Two study sizes, kept separate

{table(['Study', 'Universe and selection', 'Purpose'], [
['Full baseline', '494 retained stocks; 52 fixed pairs; 84 distinct selected stocks', 'Describe the assignment strategy and all-pair portfolio.'],
['Research pilot', 'Fixed 50-stock sample; 100 same-industry hypotheses per monthly screen', 'Compare specific rule changes under matched costs and execution.']])}

{table(['Baseline period', 'Dates', 'Daily closes'], [
['Training / formation', '2023-09-19 to 2025-10-22', '526'],
['Later historical test', '2025-10-23 to 2026-09-17', '226']])}

## What the baseline actually trades

- Select same-sub-industry pairs with positive beta, different issuers, provisional I(1) compatibility and raw training cointegration p < 0.05.
- Fit A = alpha + beta x B + residual. Trade the spread A - beta x B using a z-score based on the preceding 60 closes.
- Enter beyond +/-2; exit toward +/-0.5, after 20 trading intervals, or at the period end. A close-based signal executes at the following close.
- Divide one $100,000 account equally across all 52 pairs; keep hedge shares fixed within each trade. Idle capital remains cash.
- Charge 0.10% of actual traded dollars on entry and exit of both legs, plus 2% annual short borrowing.

**Interpretation:** cointegration is a property of a price combination under statistical assumptions. A low p-value is not a profitability score. Training performance is fitted; the later period has already informed research, so it is not an untouched holdout.

This brief explains the decisions. The linked full report and CSVs preserve the detailed assignment outputs (page 8).
""")
    pages.append(f"""## 1. From regression to an actual two-stock trade

**Figure 1. Procedure for the historical simulation.** Pair selection and regression precede the period being traded; order execution follows signal observation.

{image('procedure.png', 'Earlier prices produce the hedge and the spread signal. The trade ledger then records share quantities, prices and costs for both stocks.')}

### What linear regression contributes

{table(['Object', 'Meaning in this project'], [
['A = alpha + beta x B + residual', 'OLS finds the fitted price relationship by minimizing squared residuals.'],
['beta', 'The hedge uses beta shares of B for each share of A; it is not a percentage portfolio weight.'],
['Spread = A - beta x B', 'The trading gap; its average need not be zero because the intercept can be nonzero.'],
['z = (spread - prior mean) / prior standard deviation', 'A unit-free measure of how unusual the current gap is relative to recent history.']])}

### Worked share hedge: an illustration, not an observed trade

Suppose A costs $110, B costs $50, and the fitted beta is 2. The spread is $10. If it is unusually high, the strategy shorts A and buys B. With a $1,000 gross entry allocation:

{table(['Quantity or result', 'Calculation'], [
['Units of A', '1,000 / (110 + 2 x 50) = 4.7619 shares sold short'],
['Units of B', '2 x 4.7619 = 9.5238 shares bought'],
['If A falls to $104 and B stays $50', 'Gross profit = 4.7619 x $6 = $28.57'],
['Entry + exit trading cost at 0.10%', '$1.00 + $0.9714 = $1.9714'],
['Profit after those trading fees', '$26.60 before short-borrow expense; borrowing depends on elapsed days.']])}

The price move is hypothetical. A spread can continue widening, the fitted relationship can change, and a delayed exit can lose more than the value suggested by a threshold.

Method sources: [Statsmodels OLS](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html) and [Engle-Granger cointegration](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.coint.html).
""")
    pages.append(f"""## 2. Full baseline: 494 stocks, all 52 selected pairs

The original screen saved 5,734 raw candidates from 121,771 pair combinations. Additional strategy filters retain 52 pairs. No pair is dropped from this portfolio because it later loses money.

{table(['Metric', 'Training', 'Later period'], baseline_metrics)}

**Figure 2. Later-period net return, 2025-10-23 to 2026-09-17.** One $100,000 account, equal fixed pair allocations and the stated trading/borrowing costs.

{image('baseline-return.png', 'The portfolio ends at $96,382.91. The later-period cumulative return is negative despite strong fitted training performance.')}

Source: {csv(BASELINE, 'later-daily-portfolio', 'Baseline daily account CSV')}; {csv(BASELINE, 'performance-comparison', 'complete performance metrics CSV')}.

**Coverage and caution:** {link(artifact(CLEAN, 'report', 'pdf'), 'The clean eight-page report')} lists every selected pair's p-values and net profit on pages 3-4. Raw p-values are exploratory; none of the original candidates clears the conservative Bonferroni threshold across all 121,771 tests. Stationary-price hubs are flagged rather than advertised as the best trading stocks.
""")
    pages.append(f"""## 3. Separate pilot: five OLS variants on 50 stocks

Each variant uses monthly screens on the preceding 252 closes. Positions close at monthly boundaries; each method's remaining capital carries forward within a fold. Each fold starts a separate $100,000 account. Costs are 10 basis points per traded dollar plus 2% annual borrowing.

{table(['Earlier fold 1', 'Earlier fold 2', 'Later comparison'], [['2024-09-19 to 2025-03-31', '2025-04-01 to 2025-10-22', '2025-10-23 to 2026-09-17']])}

{table(['Variant', 'Earlier 1', 'Earlier 2', 'Later', 'Later trades'], pilot_rows)}

**Figure 3. All five pilot variants lose money in the later comparison.** US 50-stock sample, 2025-10-23 to 2026-09-17; total return after stated costs.

{image('pilot-returns.png', 'The stable-beta filter has the smallest later loss in this set. Smaller losses can also result from fewer trades or more idle cash.')}

Stable beta adds recovery-speed and hedge-stability checks. The entry filter waits for movement back toward zero and a cost-distance hurdle. The risk version adds an adverse z stop, a shorter holding cap, non-overlapping tickers and an allocation cap. The final row combines all three.

**No independent holdout:** these ideas followed earlier inspection of the later period. The baseline was selected using a predefined earlier-fold gate; that does not erase the earlier inspection or establish statistical significance. These 50-stock outcomes must not be compared directly with the 52-pair baseline to isolate a rule's effect.

Source: {csv(PILOT, 'all-performance', 'All five-variant, three-period statistics CSV')} and {link(artifact(PILOT, 'report', 'md'), 'exact rules and full pilot records')}.
""")
    pages.append(f"""## 4. New hypothesis: forecast whether the spread can repay its costs

The additional strategy retains linear regression. It asks whether a fitted correction model projects enough movement in the intended direction after the order can execute. Its exact rules were fixed before this new simulation.

**Figure 4. The error-correction entry gate.** Same 50 stocks and monthly selection as the pilot; only the additional entry gate differs from its matched OLS control.

{image('ecm-procedure.png', 'The hedge model first defines the residual spread. A second linear regression forecasts spread changes; only a stable model with sufficient projected movement can pass the entry gate.')}

### Exact model and entry condition

Fit the formation residual s = A - alpha - beta x B, then fit:

**Delta s(t) = a + k x s(t-1) + g x Delta s(t-1) + error.**

Require full regression rank, k < 0 and both implied AR(2) companion roots strictly inside the unit circle. Freeze the fitted coefficients for that month's decisions. This is one residual-spread equation, not a full vector error-correction model.

At a baseline z-score entry opportunity, use today's residual and change to forecast s(t+1) and s(t+6). Direction d is +1 for a long spread and -1 for a short spread. Enter only if:

**d x [forecast s(t+6) - forecast s(t+1)] / [price A + beta x price B] > 2 x estimated round-trip cost.**

The cost estimate is 2 x 0.001 + 0.02 x 7/365.25 x the entry short-leg fraction. Actual charged costs still use executed dollar amounts and actual holding days. The projected interval excludes the next-close execution gap; the five-interval forecast is an entry heuristic, not a new exit deadline or a guaranteed return.

Same baseline z thresholds, equal allocation, 20-bar cap, monthly liquidation and next-close execution remain in place. Cash earns zero interest. The exact gate is this project's research hypothesis; the general error-correction formulation is described in [Statsmodels VECM documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.vector_ar.vecm.VECM.html).
""")
    pages.append(f"""## 5. New strategy result: compare the matched control

Both methods use the same fixed 50-stock sample, monthly 252-close formation, periods and costs. Each cell below is the return of a separate $100,000 account; do not compound the three periods as one continuous strategy.

{table(['Period', 'OLS control', 'ECM gate', 'Change (pp)', 'Trades: OLS / ECM'], ecm_rows)}

Change is ECM minus control in percentage points (pp); it is rounded to three decimals.

**Figure 5. Matched later-period cumulative return, 2025-10-23 to 2026-09-17.** The solid orange line includes the additional forecast gate; the dashed blue line is its OLS control.

{image('ecm-returns.png', f'Later net return: OLS control {pct(control.Total_Return)}; ECM gate {pct(measured.Total_Return)}. These are observed historical outcomes, not an independent validation.')}

{table(['Later-period measure', 'OLS control', 'ECM gate'], [
['Maximum drawdown', pct(control.Max_Drawdown), pct(measured.Max_Drawdown)],
['Trading + borrowing costs', dollars(control.Total_Costs), dollars(measured.Total_Costs)],
['Closed trades', int(control.Num_Trades), int(measured.Num_Trades)]])}

**What this establishes:** {no_profit} The forecast gate changes which trades occur. Differences in exposure, idle cash and selection can affect return and drawdown; this single run does not establish a durable edge.

**What it cannot establish:** the two earlier folds are development data and the later period has already been examined. The new rule is not a fresh holdout test, even though its settings were recorded before this run.

Source: {csv(ecm_dir, 'all-performance', 'All ECM/control period statistics CSV')} and {link(artifact(ecm_dir, 'report', 'md'), 'ECM experiment report and trade records')}.
""")
    pages.append(f"""## 6. What the evidence answers - and what remains

{table(['Research question', 'Answer supported by these saved results'], question_rows)}

### The next experiment should improve the evidence

1. Obtain a longer, point-in-time universe and price history, with executable bid/ask information and borrowing assumptions where possible.
2. Keep a chronological development set and a genuinely unexamined final period. Record the candidate methods and decision rule before evaluation.
3. Test the cointegration cutoff, hedge lookback, stop and holding limit on the development folds. Account for multiple pair tests and repeated strategy trials.
4. Check stock-level exposure, liquidity and cost stress. Compare risk and capital usage alongside return so that extra cash is not mistaken for a better signal.
5. Freeze one specification before forward paper trading. Preserve losses and rejected ideas in the research record.

The individual-price I(1) checks are provisional. Current index membership and full-period data availability introduce historical selection. Adjusted closes, fractional shares and assumed short availability simplify execution. The early full-baseline in-sample return uses pairs and a hedge fitted on that same window.

**Assignment status:** all eleven expected output types are available through the index on the next page. Parameter optimization is still incomplete: the project has tested several variants and thresholds, but has not established a validated optimum or live profitability.
""")
    pages.append(f"""## 7. Evidence index and presentation sources

This brief supplements the detailed results. {link(artifact(CLEAN, 'report', 'pdf'), 'Open the clean eight-page baseline report')} for the full pair table, sample trade log, spread/z-score plots, sensitivity and performance measures.

{table(['Requested assignment output', 'Where to read it', 'Machine-readable record'], coverage_rows)}

### Study records

- Baseline assumptions and reconciliation: {link(artifact(BASELINE, 'run-metadata', 'json'), '494-stock baseline metadata')}.
- All pilot variant/period pair contributions: {csv(PILOT, 'all-pair-profits', '50-stock pilot pair profits CSV')}.
- New experiment: {link(artifact(ecm_dir, 'report', 'md'), 'linear error-correction report')} and its linked plan, selection, account and trade records.

### Why this presentation is different

Following the UK Government Analysis Function's [chart guidance](https://analysisfunction.civilservice.gov.uk/policy-store/data-visualisation-charts/) and [table guidance](https://analysisfunction.civilservice.gov.uk/policy-store/data-visualisation-tables/), the brief uses one main message per figure, titles and source links in document text, short readable tables, direct values, light gridlines and zero-based bar axes. Chart descriptions accompany the figures; numerical tables and full CSV records remain available. This is a presentation improvement, not a claim of formal PDF accessibility certification.

The project calculations use locally saved market data. External sources support statistical definitions and design choices; they do not certify this project's returns. The separate learning textbook explains the mathematics and exercises in greater depth.
""")
    md_path.write_text("\n\n<!-- pagebreak -->\n\n".join(page.strip() for page in pages) + "\n", encoding="utf-8")
    render_report(md_path, pdf_path, profile="clean")
    return md_path, pdf_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ecm-dir", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path)
    args = parser.parse_args()
    for path in build_brief(args.ecm_dir.resolve(), args.output_stem):
        print(path)


if __name__ == "__main__":
    main()
