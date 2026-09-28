"""
Visualization and analysis module for pairs trading results.
Generates charts and summary reports.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


def plot_equity_curve(equity_curve, title="Portfolio Equity Curve", figsize=(14, 6)):
    """
    Plot the equity curve.
    
    Args:
        equity_curve (pd.Series or pd.DataFrame): Portfolio values over time
        title (str): Plot title
        figsize (tuple): Figure size
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if isinstance(equity_curve, pd.DataFrame):
        equity_curve = equity_curve['Portfolio_Value']
    
    ax.plot(equity_curve.index, equity_curve.values, linewidth=2, color='steelblue')
    ax.fill_between(equity_curve.index, equity_curve.values, alpha=0.3, color='steelblue')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    plt.tight_layout()
    return fig


def plot_spread_and_zscore(signals, title="Spread and Z-Score", figsize=(14, 8),
                           entry_threshold=None, exit_threshold=None):
    """
    Plot spread and z-score with trading signals.
    
    Args:
        signals (pd.DataFrame): Signals DataFrame
        title (str): Plot title
        figsize (tuple): Figure size
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    entry = float(signals.attrs.get('entry_threshold', 2.0)
                  if entry_threshold is None else entry_threshold)
    exit_level = float(signals.attrs.get('exit_threshold', 0.0)
                       if exit_threshold is None else exit_threshold)

    # A price spread need not be centered on zero. Display its rolling center.
    ax1.plot(signals.index, signals['Spread'], label='Spread', color='steelblue', linewidth=1.5)
    mean_column = next((column for column in ['Rolling_Mean', 'Moving_Mean']
                        if column in signals), None)
    if mean_column is not None:
        center = signals[mean_column]
        center_label = 'Rolling mean used for z-score'
    else:
        lookback = int(signals.attrs.get('lookback', 60))
        center = signals['Spread'].rolling(lookback).mean()
        center_label = f'{lookback}-day rolling mean (display)'
    ax1.plot(signals.index, center, color='#b66d27', linestyle='--', linewidth=1.3,
             label=center_label)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.set_ylabel('Spread', fontsize=11)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot z-score with entry/exit thresholds
    ax2.plot(signals.index, signals['ZScore'], label='Z-Score', color='darkgreen', linewidth=1.5)
    ax2.axhline(y=entry, color='red', linestyle='--', alpha=0.6,
                label=f'Entry +{entry:g}')
    ax2.axhline(y=-entry, color='green', linestyle='--', alpha=0.6,
                label=f'Entry −{entry:g}')
    if exit_level == 0:
        ax2.axhline(y=0, color='black', linestyle=':', alpha=.65, label='Exit at 0')
    else:
        ax2.axhline(y=exit_level, color='black', linestyle=':', alpha=.65,
                    label=f'Exit levels ±{abs(exit_level):g}')
        ax2.axhline(y=-exit_level, color='black', linestyle=':', alpha=.65)
    ax2.fill_between(signals.index, signals['ZScore'], 0, alpha=0.2, color='darkgreen')
    
    # These are signal changes; delayed executions are recorded in the trade log.
    changes = (signals['Signal_Change'] if 'Signal_Change' in signals
               else pd.Series(0, index=signals.index))
    trades = signals.loc[changes.notna() & (changes != 0)]
    for idx in trades.index:
        color = 'green' if changes.loc[idx] > 0 else 'red'
        marker = '^' if changes.loc[idx] > 0 else 'v'
        ax2.scatter(idx, trades.loc[idx, 'ZScore'], color=color, marker=marker, s=45,
                    zorder=5, label='Signal change' if idx == trades.index[0] else None)
    
    ax2.set_ylabel('Z-Score', fontsize=11)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_price_comparison(signals, figsize=(14, 6)):
    """
    Plot both prices in the pair.
    
    Args:
        signals (pd.DataFrame): Signals DataFrame
        figsize (tuple): Figure size
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax2 = ax.twinx()
    
    ax.plot(signals.index, signals['Price1'], label='Price 1', color='steelblue', linewidth=2)
    ax2.plot(signals.index, signals['Price2'], label='Price 2', color='coral', linewidth=2)
    
    ax.set_ylabel('Price 1', fontsize=11, color='steelblue')
    ax2.set_ylabel('Price 2', fontsize=11, color='coral')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_title('Price Comparison', fontsize=14, fontweight='bold')
    
    ax.tick_params(axis='y', labelcolor='steelblue')
    ax2.tick_params(axis='y', labelcolor='coral')
    
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_drawdown(equity_curve, figsize=(14, 6)):
    """
    Plot drawdown from peak equity.
    
    Args:
        equity_curve (pd.Series or pd.DataFrame): Portfolio values
        figsize (tuple): Figure size
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    if isinstance(equity_curve, pd.DataFrame):
        equity_curve = equity_curve['Portfolio_Value']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max
    
    ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.5, color='red', label='Drawdown')
    ax.plot(drawdown.index, drawdown.values, color='darkred', linewidth=2)
    
    ax.set_title('Portfolio Drawdown', fontsize=14, fontweight='bold')
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    return fig


def plot_cointegration_heatmap(cointegrated_pairs, figsize=(12, 10), max_tickers=18):
    """
    Plot a readable subset of saved cointegration p-values.

    Choose stocks with the most saved passing partners. Missing results and
    self-pairs are masked; they are never represented as an invented p-value.
    One triangle displays the originally tested orientation's p-value.
    
    Args:
        cointegrated_pairs (list): List of cointegration results
        figsize (tuple): Figure size
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    from cointegration_review import normalize_pair_results

    pairs_df = normalize_pair_results(cointegrated_pairs)
    if pairs_df.empty:
        return None
    if max_tickers < 2:
        raise ValueError('A heatmap requires at least two tickers.')
    max_tickers = min(int(max_tickers), 20)
    passing = pairs_df.loc[pairs_df['Coint_PValue'] < .05]
    counts = pd.concat([passing['Ticker1'], passing['Ticker2']]).value_counts()
    universe = sorted(set(pairs_df['Ticker1']) | set(pairs_df['Ticker2']))
    tickers = sorted(universe, key=lambda t: (-counts.get(t, 0), t))[:max_tickers]
    matrix = pd.DataFrame(np.nan, index=tickers, columns=tickers)
    for row in pairs_df.itertuples(index=False):
        if row.Ticker1 in matrix.index and row.Ticker2 in matrix.columns:
            matrix.loc[row.Ticker1, row.Ticker2] = row.Coint_PValue
            matrix.loc[row.Ticker2, row.Ticker1] = row.Coint_PValue
    mask = matrix.isna().to_numpy() | np.triu(np.ones(matrix.shape, dtype=bool))
    annotations = np.empty(matrix.shape, dtype=object)
    annotations[:] = ''
    for i, j in zip(*np.where(~mask)):
        pvalue = matrix.iat[i, j]
        annotations[i, j] = f'{pvalue:.0e}' if pvalue < .001 else f'{pvalue:.3f}'

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor('#e6e9ee')
    sns.heatmap(matrix, mask=mask, cmap='YlGn_r', annot=annotations, fmt='',
                annot_kws={'fontsize': 7}, cbar_kws={'label': 'Raw training p-value'},
                ax=ax, vmin=0, vmax=.05, linewidths=.5, linecolor='white', square=True)
    ax.set_title(f'Cointegration candidates | selected {len(tickers)} of {len(universe)} stocks',
                 fontsize=14, fontweight='bold', pad=18)
    ax.tick_params(axis='both', labelsize=10)
    ax.set_xticklabels(tickers, rotation=45, ha='right')
    ax.set_yticklabels(tickers, rotation=0)
    ax.set_xlabel('Stocks ranked by number of saved passing partners', fontsize=10)
    fig.text(.5, .025,
             'Gray = no saved value / self-pair / duplicate triangle. '
             'Raw p-values are unadjusted for multiple tests.\n'
             'Only the originally tested regression orientation is shown; '
             'a low p-value does not establish trading quality.',
             ha='center', fontsize=9, color='#475569')
    fig.tight_layout(rect=(0, .075, 1, 1))
    return fig


def plot_cointegration_partner_counts(ranked_stocks, max_stocks=20, figsize=(13, 9)):
    """Plot raw candidate-partner counts from build_cointegration_review()."""
    if ranked_stocks.empty:
        return None
    shown = ranked_stocks.head(min(int(max_stocks), 20)).iloc[::-1]
    labels = [f'{row.Ticker}  |  {row.Company}' for row in shown.itertuples()]
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(labels, shown['Partner_Count'], color='#35658b', height=.68)
    ax.bar_label(bars, padding=5, fontsize=10)
    ax.set_xlim(0, max(shown['Partner_Count'].max() * 1.14, 1))
    ax.set_xlabel('Distinct partners passing the original training screen (raw p < 0.05)')
    ax.set_title('Most candidate partners | exploratory ranking', fontweight='bold', fontsize=16, pad=18)
    ax.grid(axis='x', alpha=.2)
    ax.grid(axis='y', visible=False)
    fig.text(.5, .02,
             'Stationary price-level stocks can appear as misleading hubs. '
             'Check I(1) assumptions and multiple testing before selecting trades.',
             ha='center', fontsize=9, color='#475569')
    fig.tight_layout(rect=(0, .04, 1, 1))
    return fig


def generate_performance_report(metrics, pair_id=None):
    """
    Generate a text report of performance metrics.
    
    Args:
        metrics (dict): Performance metrics dictionary
        pair_id (str): Pair identifier for the report
    
    Returns:
        str: Formatted report text
    """
    report = []
    
    if pair_id:
        report.append(f"{'='*60}")
        report.append(f"Performance Report: {pair_id}")
        report.append(f"{'='*60}\n")
    
    report.append("RETURNS")
    report.append(f"  Total Return:        {metrics.get('Total_Return', 0):>10.2%}")
    report.append(f"  Annualized Return:   {metrics.get('Annualized_Return', 0):>10.2%}")
    
    report.append("\nRISK METRICS")
    report.append(f"  Volatility (Annual): {metrics.get('Volatility', 0):>10.2%}")
    report.append(f"  Max Drawdown:        {metrics.get('Max_Drawdown', 0):>10.2%}")
    report.append(f"  Sharpe Ratio:        {metrics.get('Sharpe_Ratio', 0):>10.2f}")
    
    report.append("\nTRADE STATISTICS")
    report.append(f"  Number of Trades:    {metrics.get('Num_Trades', 0):>10.0f}")
    report.append(f"  Win Rate:            {metrics.get('Win_Rate', 0):>10.2%}")
    report.append(f"  Avg Trade P&L:       {metrics.get('Avg_Trade_PnL', 0):>10.2f}")
    
    report.append("\nCOSTS & EXECUTION")
    report.append(f"  Total Costs:         {metrics.get('Total_Transaction_Costs', 0):>10.2f}")
    report.append(f"  Turnover (Annual):   {metrics.get('Turnover', 0):>10.2f}")
    
    report.append("\nFINAL PORTFOLIO")
    report.append(f"  Final Value:         {metrics.get('Final_Portfolio_Value', 0):>10.2f}")
    
    return "\n".join(report)


def generate_comprehensive_report(backtest_results, portfolio_results, output_path='backtest_report.txt'):
    """
    Generate a comprehensive backtest report.
    
    Args:
        backtest_results (dict): Individual pair backtest results
        portfolio_results (dict): Combined portfolio results
        output_path (str): Path to save the report
    """
    report = []
    
    report.append("="*80)
    report.append("PAIRS TRADING STRATEGY - COMPREHENSIVE BACKTEST REPORT")
    report.append("="*80)
    report.append("")
    
    # Portfolio summary
    report.append("PORTFOLIO PERFORMANCE")
    report.append("-"*80)
    if portfolio_results:
        report.append(generate_performance_report(portfolio_results['metrics'], "Combined Portfolio"))
        report.append("\nPortfolio Weights:")
        for pair_id, weight in portfolio_results['weights'].items():
            report.append(f"  {pair_id}: {weight:.2%}")
    report.append("")
    
    # Individual pair results
    report.append("\nINDIVIDUAL PAIR RESULTS")
    report.append("-"*80)
    
    for pair_id, result in backtest_results.items():
        report.append(generate_performance_report(result['metrics'], pair_id))
        report.append("")
    
    # Summary statistics
    report.append("\nSUMMARY STATISTICS")
    report.append("-"*80)
    if backtest_results:
        sharpe_ratios = [r['metrics']['Sharpe_Ratio'] for r in backtest_results.values()]
        returns = [r['metrics']['Annualized_Return'] for r in backtest_results.values()]
        
        report.append(f"Number of Pairs Tested:      {len(backtest_results)}")
        report.append(f"Average Sharpe Ratio:        {np.mean(sharpe_ratios):.2f}")
        report.append(f"Average Annual Return:       {np.mean(returns):.2%}")
        report.append(f"Best Sharpe Ratio:           {np.max(sharpe_ratios):.2f}")
        report.append(f"Worst Sharpe Ratio:          {np.min(sharpe_ratios):.2f}")
    
    report_text = "\n".join(report)
    
    # Save to file
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    logger.info(f"Report saved to {output_path}")
    return report_text


if __name__ == "__main__":
    from data_fetcher import fetch_price_data, clean_price_data, split_data
    from cointegration import find_cointegrated_pairs
    from signal_generation import generate_all_signals
    from backtester import backtest_all_pairs, combine_portfolio_results
    from datetime import datetime, timedelta
    
    # Example usage
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'JPM', 'V', 'JNJ', 'WMT', 'PG']
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')
    
    price_data = fetch_price_data(tickers, start_date, end_date)
    price_data = clean_price_data(price_data)
    train_data, test_data = split_data(price_data)
    
    pairs = find_cointegrated_pairs(train_data, sample=5)
    signal_data = generate_all_signals(test_data, pairs)
    
    results = backtest_all_pairs(signal_data)
    portfolio = combine_portfolio_results(results)
    
    if results:
        first_pair = list(results.keys())[0]
        plot_equity_curve(results[first_pair]['equity_curve'], title=f"{first_pair} Equity Curve")
        plt.savefig('equity_curve.png', dpi=150, bbox_inches='tight')
        
        if portfolio:
            plot_equity_curve(portfolio['equity_curve'], title="Portfolio Equity Curve")
            plt.savefig('portfolio_equity.png', dpi=150, bbox_inches='tight')
        
        report = generate_comprehensive_report(results, portfolio)
        print(report)
