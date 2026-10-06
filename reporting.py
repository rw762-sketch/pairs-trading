"""One HTML/PDF report and auditable CSV outputs for each pipeline run."""

from pathlib import Path
from html import escape
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image,
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from visualization import make_charts


def write_report(output, run, portfolio, pair_metrics, selected, signals):
    output = Path(output)
    make_charts(output, portfolio, signals)
    m = portfolio["metrics"]
    c = run["config"]
    capital = m["Initial_Capital"]
    stats = [
        ("Net total return", f"{m['Total_Return']:.2%}"),
        ("Annualized return (CAGR)", f"{m['Annualized_Return']:.2%}"),
        ("Net profit", f"${m['Final_Portfolio_Value'] - capital:,.2f}"),
        ("Final account", f"${m['Final_Portfolio_Value']:,.2f}"),
        ("Sharpe ratio", f"{m['Sharpe_Ratio']:.3f}"),
        ("Annualized volatility", f"{m['Volatility']:.2%}"),
        ("Maximum drawdown", f"{m['Max_Drawdown']:.2%}"),
        ("Win rate", f"{m['Win_Rate']:.1%}"),
        ("Completed trades", str(m["Num_Trades"])),
        ("Annualized turnover", f"{m['Turnover']:.2f}x"),
        ("Average allocated trade profit", f"${m['Avg_Trade_PnL']:,.2f}"),
        ("Average trade return", f"{m['Avg_Trade_Return']:.2%}"),
        ("Transaction costs", f"${m['Total_Transaction_Costs']:,.2f}"),
        ("Borrow costs", f"${m['Total_Borrow_Costs']:,.2f}"),
    ]
    method = (
        f"{run['requested_stocks']} requested securities; {run['training_eligible_stocks']} training-eligible. "
        f"K-means produced {run['clustering']['actual_clusters']} clusters. All {run['tested_pairs']:,} within-cluster pairs "
        f"received cointegration tests ({run['failed_tests']} failed tests retained with status). "
        f"{run['selected_pairs']} pairs qualified; {run['cash_sleeves']} unavailable-price sleeves remained in cash. "
        "No correlation or industry filter precedes the tests."
    )
    selection = (
        f"Eligibility: {c['correction']} cointegration p < {c['pvalue']}, positive OLS beta, provisional individual "
        f"I(1) diagnostics, half-life {c['half_life_min']} to {c['half_life_max']} closes and at least {c['min_crossings']} annualized mean crossings. No shared tickers between selected pairs. Rank by training raw p-value, then half-life. "
        "Raw mode is exploratory and does not control family-wide false positives; Holm-adjusted p-values are also saved."
    )
    execution = (
        f"Initial capital ${capital:,.0f}, equal allocations capped at {c['max_pair_weight']:.0%} per pair, with unused cash retained; beta is fixed from training, "
        f"shares are fixed within each trade. Z-score uses the preceding {c['lookback']} closes. "
        f"Entry +/-{c['entry']}, exit +/-{c['exit']}, adverse stop +/-{c['stop']}; "
        f"maximum holding {str(c['max_hold']) + ' trading intervals' if c['max_hold'] else 'disabled'}. Orders execute at the next close. "
        f"Fees/slippage {c['fee']:.2%} of traded dollars, plus {c['borrow']:.1%} annual short borrowing."
    )
    limitations = (
        "Current constituents introduce historical selection bias. Clustering is a computational filter and can miss "
        "cross-cluster opportunities. This is a retrospective chronological holdout already used in project research, "
        "not a fresh test. Missing evaluation prices retain that sleeve in cash for the full fold; this availability convention "
        "can affect performance. Adjusted closes proxy executions. Fractional shares and short availability are assumed; "
        "liquidity, recalls, market impact, taxes and cash interest are not modeled. No parameter optimization or walk-forward "
        "profitability claim is made by this pipeline."
    )
    dates = f"Training: {run['training_start']} to {run['training_end']}. Evaluation: {run['evaluation_start']} to {run['evaluation_end']}."
    allocation = min(capital / max(len(selected), 1), c["max_pair_weight"] * capital)
    display = []
    if len(pair_metrics):
        for row in pair_metrics.head(20).itertuples():
            display.append(
                [
                    row.Pair,
                    f"{row.Total_Return:.2%}",
                    f"{row.Sharpe_Ratio:.2f}",
                    f"{row.Max_Drawdown:.2%}",
                    str(row.Num_Trades),
                    f"${row.Allocated_Profit:,.2f}",
                ]
            )
    html_rows = "".join(
        f"<tr><td>{escape(k)}</td><td>{escape(v)}</td></tr>" for k, v in stats
    )
    pair_rows = "".join(
        "<tr>" + "".join(f"<td>{escape(str(x))}</td>" for x in row) + "</tr>"
        for row in display
    )
    zchart = (
        '<h2>Z-score examples</h2><p>First six evaluated pairs by training significance; orange entry and green exit bands.</p><img src="figures/zscores.png" alt="Historical z-scores">'
        if signals
        else ""
    )
    (
        output / "report.html"
    ).write_text(f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Clustered pairs trading results</title>
<style>body{{font:16px system-ui;max-width:1050px;margin:40px auto;padding:0 24px;color:#142b45}}p{{line-height:1.6}}table{{border-collapse:collapse;width:100%;margin:20px 0}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #dce4eb}}th{{background:#142b45;color:white}}tr:nth-child(even){{background:#eef4f7}}img{{width:100%}}a{{color:#1d4ed8}}</style>
<h1>Clustered pairs trading</h1><p>{dates}</p><p>{method}</p><p><a href="report.pdf">Download clean PDF</a></p>
<h2>Portfolio statistics</h2><table>{html_rows}</table><h2>Selection and execution</h2><p>{selection}</p><p>{execution}</p>
<h2>Portfolio return and drawdown</h2><img src="figures/portfolio.png" alt="Portfolio return and drawdown">
<h2>Pair results</h2><p>First 20 pairs in training selection order. Dollar profits reflect actual equal allocations. All pairs are in the CSV.</p><table><tr><th>Pair</th><th>Return</th><th>Sharpe</th><th>Max DD</th><th>Trades</th><th>Profit</th></tr>{pair_rows}</table>{zchart}
<h2>Complete records</h2><p><a href="cointegration-tests.csv">Every pair test</a> | <a href="selected-pairs.csv">Selected pairs</a> | <a href="cluster-assignments.csv">Cluster assignments</a> | <a href="pair-metrics.csv">Every pair's metrics</a> | <a href="portfolio-metrics.csv">Portfolio metrics</a> | <a href="portfolio-trades.csv">Trade log</a> | <a href="portfolio-account.csv">Daily account</a></p><h2>Limitations</h2><p>{limitations}</p></html>""")
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleX",
            fontName="Helvetica-Bold",
            fontSize=23,
            leading=28,
            textColor=colors.HexColor("#142b45"),
            spaceAfter=14,
        )
    )
    styles.add(ParagraphStyle(name="BodyX", fontSize=9.5, leading=14, spaceAfter=9))
    styles.add(ParagraphStyle(name="CellX", fontSize=8, leading=10))
    story = []

    def p(text, style="BodyX"):
        story.append(Paragraph(text, styles[style]))

    def tab(headers, rows, widths):
        cells = [
            [Paragraph(escape(str(x)), styles["CellX"]) for x in row]
            for row in [headers, *rows]
        ]
        t = Table(cells, colWidths=widths, repeatRows=1)
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce8ee")),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#f0f5f8")],
                    ),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(t)
        story.append(Spacer(1, 12))

    def new(title):
        story.append(PageBreak())
        p(title, "TitleX")

    p("Clustered pairs trading", "TitleX")
    p(dates)
    p("Portfolio statistics", "Heading2")
    tab(["Metric", "Result"], stats, [320, 190])
    p(method)
    p(
        "All allocated trade profits reconcile to the account change. Returns and risk statistics use the combined daily portfolio, not averages of pair Sharpe ratios."
    )
    new("Screening and strategy")
    p(method)
    p(selection)
    p(execution)
    tab(
        ["Setting", "Value"],
        [
            ["Requested price window", f"{c['start']} to {c['end']} (exclusive end)"],
            [
                "Total / training / evaluation closes",
                f"{run['history_closes']} / {run['training_closes']} / {run['evaluation_closes']}",
            ],
            [
                "All possible / tested within-cluster pairs",
                f"{run['all_possible_pairs']:,} / {run['tested_pairs']:,}",
            ],
            ["Clustering seed / initializations", f"{c['seed']} / 10"],
            ["Cointegration family correction", c["correction"]],
            [
                "Selected pairs / allocation per pair",
                f"{len(selected)} / ${allocation:,.2f}"
                if len(selected)
                else "0 / account remains in cash",
            ],
        ],
        [270, 240],
    )
    p("Definitions", "Heading2")
    p(
        "Sharpe: mean daily return / sample daily standard deviation x sqrt(252), using zero risk-free rate. Volatility: daily standard deviation x sqrt(252). Annualized return: CAGR over actual calendar time. Drawdown: minimum account / running peak - 1. Win rate: completed trades with positive net profit. Turnover: gross dollars traded / initial capital / elapsed years. Half-life: -log(2) / log(phi) from training AR(1), defined only for 0 < phi < 1."
    )
    p(limitations)
    new("Portfolio return and drawdown")
    story.append(Image(str(output / "figures/portfolio.png"), width=510, height=357))
    p(
        "Daily equity is after transaction and borrowing costs. The running peak includes initial capital."
    )
    p(
        f"Generated {run['generated_at']}. Statistics and all pair tests are saved in the companion CSV files."
    )
    new("Pair results")
    p(
        "First 20 pairs by training selection order. Profits use actual equal allocations; full metrics and trade logs are in the CSV files."
    )
    tab(
        ["Pair", "Return", "Sharpe", "Max DD", "Trades", "Profit"],
        display,
        [95, 85, 70, 85, 65, 110],
    )
    p(
        "Unavailable-price sleeves remain in cash and are identified in pair-metrics.csv."
    )
    if signals:
        new("Historical z-scores")
        p(
            "First six evaluated pairs by training significance. Orange dashed lines: entry thresholds. Green dotted lines: exit bands. Signals execute on the next close; these are historical observations."
        )
        story.append(Image(str(output / "figures/zscores.png"), width=510, height=408))

    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#536477"))
        canvas.drawString(51, 28, "Clustered pairs trading | Historical research")
        canvas.drawRightString(561, 28, str(doc.page))

    SimpleDocTemplate(
        str(output / "report.pdf"),
        pagesize=(612, 792),
        leftMargin=51,
        rightMargin=51,
        topMargin=42,
        bottomMargin=55,
        title="Clustered pairs trading results",
    ).build(story, onFirstPage=footer, onLaterPages=footer)


def write_walk_forward_report(output, plan, windows, portfolios):
    """Matched baseline/filter comparisons; all accounts carry across windows."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    output = Path(output)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    rows = []
    for name, portfolio in portfolios.items():
        m = portfolio["metrics"]
        curve = portfolio["equity_curve"]
        capital = m["Initial_Capital"]
        axes[0].plot(curve.index, curve / capital - 1, label=name)
        axes[1].plot(
            curve.index, curve / curve.cummax().clip(lower=capital) - 1, label=name
        )
        rows.append(
            [
                name,
                f"{m['Total_Return']:.2%}",
                f"{m['Sharpe_Ratio']:.3f}",
                f"{m['Max_Drawdown']:.2%}",
                f"{m['Win_Rate']:.1%}",
                str(m["Num_Trades"]),
                f"{m['Turnover']:.2f}x",
            ]
        )
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend()
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylabel("Net cumulative return")
    axes[1].set_ylabel("Drawdown")
    fig.tight_layout()
    fig.savefig(figures / "comparison.png", dpi=150)
    plt.close(fig)
    c = plan["config"]
    method = (
        f"{plan['window_count']} non-overlapping trading windows. Before each window, use only the preceding "
        f"{c['formation']} closes to refit K-means, test every within-cluster pair and fit hedge ratios. "
        f"Trade the following {c['trading_window']} closes (last window may be partial), then liquidate. "
        "Each variant starts once at $100,000; capital carries between windows. All variants use identical "
        "data, costs, exits and window boundaries. Baseline retains the prior unbounded finite-half-life selection. "
        f"Filtered requires half-life {c['half_life_min']} to {c['half_life_max']} closes, at least "
        f"{c['min_crossings']} annualized mean crossings, and no shared tickers across pairs. "
        f"Allocate at most {c['max_pair_weight']:.0%} per pair, retaining unused capital in cash."
    )
    if "previous_loss_filter" in portfolios:
        method += " Previous-loss variant: " + plan["previous_loss_rule"]
    caveat = (
        "Allocation limits apply to initial gross notional, not a continuously enforced mark-to-market cap. "
        "With non-overlapping pairs, each stock initially receives at most the pair cap; later price changes can "
        "change exposure. The limits do not control common sector/factor risks. Less loss can reflect more cash "
        "or lower turnover, not a stronger signal. Raw p-values remain exploratory; thresholds are fixed, "
        "not selected by these results. Current constituents, overlapping formation history, missing-price cash "
        "sleeves and previously inspected dates remain limitations. Trading windows do not overlap, but their "
        "returns are not assumed statistically independent. This is retrospective, not an untouched holdout."
    )
    fold_rows = []
    for row in windows.itertuples():
        fold_rows.append([
            str(row.Window), row.Variant,
            str(row.Trading_Start.date()), str(row.Trading_End.date()),
            f"{row.Total_Return:.2%}", str(int(row.Selected_Pairs)),
            str(int(row.Num_Trades)),
        ])

    def html_table(headers, table_rows):
        return (
            "<table><tr>"
            + "".join(f"<th>{escape(x)}</th>" for x in headers)
            + "</tr>"
            + "".join(
                "<tr>" + "".join(f"<td>{escape(str(x))}</td>" for x in row) + "</tr>"
                for row in table_rows
            )
            + "</table>"
        )

    headers = [
        "Variant",
        "Net return",
        "Sharpe",
        "Max drawdown",
        "Win rate",
        "Trades",
        "Turnover",
    ]
    fold_headers = ["Window", "Variant", "Start", "End", "Return", "Pairs", "Trades"]

    (output / "report.html").write_text(
        f"""<!doctype html><meta charset="utf-8"><title>Chronological pairs-trading comparison</title><style>body{{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#142b45}}p{{line-height:1.6}}table{{border-collapse:collapse;width:100%;margin:24px 0}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #dce4eb}}th{{background:#142b45;color:white}}tr:nth-child(even){{background:#eef4f7}}img{{width:100%}}</style><h1>Repeated chronological evaluation</h1><p>{method}</p><p><a href="report.pdf">Download PDF</a></p>{html_table(headers, rows)}<img src="figures/comparison.png" alt="Baseline versus filtered returns and drawdowns"><h2>Window results</h2>{html_table(fold_headers, fold_rows)}<h2>Complete records</h2><p><a href="comparison.csv">Full overall statistics</a> | <a href="window-results.csv">Full window statistics</a> | <a href="plan.json">Plan recorded before simulation</a> | <a href="filtered-trades.csv">Filtered trades</a> | <a href="baseline-trades.csv">Baseline trades</a></p><h2>Interpretation</h2><p>{caveat}</p>"""
    )
    styles = getSampleStyleSheet()
    story = []

    def p(text, style="BodyText"):
        story.append(Paragraph(text, styles[style]))
        story.append(Spacer(1, 10))

    def table(headers, values, widths):
        t = Table(
            [
                [Paragraph(escape(str(x)), styles["BodyText"]) for x in row]
                for row in [headers, *values]
            ],
            colWidths=widths,
            repeatRows=1,
        )
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce8ee")),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#f0f5f8")],
                    ),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(t)
        story.append(Spacer(1, 14))

    p("Repeated chronological evaluation", "Title")
    p(method)
    table(headers, rows, [67, 73, 65, 80, 65, 70, 70])
    p("Settings fixed before simulation", "Heading2")
    p(
        f"Entry +/-{c['entry']}; exit +/-{c['exit']}; adverse stop +/-{c['stop']}; holding cap {str(c['max_hold']) + ' trading intervals' if c['max_hold'] else 'disabled'}. Fees {c['fee']:.2%} of traded dollars; annual short borrowing {c['borrow']:.1%}. Signals execute on the next close. K-means seed {c['seed']}; {c['clusters']} clusters. Cointegration correction: {c['correction']}."
    )
    p(caveat)
    story.append(PageBreak())
    p("Matched window results", "Title")
    table(fold_headers, fold_rows, [35, 115, 75, 75, 70, 60, 60])
    p(
        "Window returns refer to each variant's capital at the start of that window. Overall returns come from the chained daily account; window returns must not be added. Formation ends strictly before trading begins."
    )
    p("Statistics", "Heading2")
    p(
        "Sharpe and volatility annualize daily returns using 252 closes and zero risk-free rate. Drawdown uses the running peak including initial capital. Win rate counts strictly positive completed net trades. Turnover counts actual gross traded dollars per initial capital per elapsed year. All profits reconcile to the account change. Full CAGR, volatility, costs and average trade statistics are in comparison.csv."
    )
    story.append(PageBreak())
    p("Return and drawdown comparison", "Title")
    story.append(Image(str(figures / "comparison.png"), width=510, height=357))
    p(
        "Baseline and filtered portfolios share the same chronological windows and execution assumptions. Both close positions at each window boundary. No variant is promoted automatically using later profits."
    )

    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.drawString(51, 28, "Clustered pairs trading | Chronological comparison")
        canvas.drawRightString(561, 28, str(doc.page))

    SimpleDocTemplate(
        str(output / "report.pdf"),
        pagesize=(612, 792),
        leftMargin=51,
        rightMargin=51,
        topMargin=42,
        bottomMargin=55,
        title="Repeated chronological pairs-trading evaluation",
    ).build(story, onFirstPage=footer, onLaterPages=footer)
