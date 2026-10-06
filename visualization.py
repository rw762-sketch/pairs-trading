"""Portfolio and z-score plots for the current pipeline only."""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def make_charts(output, portfolio, signals):
    output = Path(output)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    curve = portfolio["equity_curve"]
    capital = portfolio["metrics"]["Initial_Capital"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(curve.index, curve / capital - 1, color="#087e83", lw=1.7)
    axes[1].fill_between(
        curve.index,
        curve / curve.cummax().clip(lower=capital) - 1,
        0,
        color="#b85656",
        alpha=0.6,
    )
    for ax in axes:
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Net cumulative return")
    axes[1].set_ylabel("Drawdown")
    fig.tight_layout()
    fig.savefig(figures / "portfolio.png", dpi=150)
    plt.close(fig)
    pairs = list(signals)[:6]  # Training significance ordering, never later profits.
    if pairs:
        fig, axes = plt.subplots(3, 2, figsize=(10, 8))
        for ax, pair in zip(axes.flat, pairs):
            sig = signals[pair]
            ax.plot(sig.index, sig.ZScore, color="#2563eb", lw=1)
            for level in (-sig.attrs["entry_threshold"], sig.attrs["entry_threshold"]):
                ax.axhline(level, color="#c48313", ls="--", lw=0.8)
            for level in (-sig.attrs["exit_threshold"], sig.attrs["exit_threshold"]):
                ax.axhline(level, color="#15803d", ls=":", lw=0.8)
            ax.set_title(pair, loc="left", fontsize=10)
            ax.grid(alpha=0.15)
            ax.tick_params(axis="x", rotation=25, labelsize=7)
        for ax in list(axes.flat)[len(pairs) :]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(figures / "zscores.png", dpi=150)
        plt.close(fig)
    return figures
