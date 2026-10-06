# Frozen official strategy replay

These files make `python main.py` work on a fresh clone without downloading a different price history. The default runner verifies every bundled input against `manifest.json`, then checks the price snapshot against the hash in `selected_strategy.json` before loading it.

The bundle preserves the original Yahoo Finance adjusted-close snapshot, all 16,106 final formation tests, the six-window persistence evidence, and the original retained-pair/trade records. The replay applies the current persistence, recovery, and allocation rules to those saved screens, then recomputes signals and trades from the frozen prices. It does not rerun clustering or cointegration estimation; those saved screening inputs can be examined directly. The `holdout` and `walk-forward` modes run new research screens instead.

The expected replay is three pairs and 22 trades over September 17, 2025 through September 17, 2026, returning **3.900813% net** on $100,000 initial capital. `plan.json`, `selected-pairs.csv`, and `unlimited-trades.csv` are historical reference records from the earlier equal 20% allocation; the official specification is the root `selected_strategy.json`. The runner recomputes the official evidence weights and reconciles the resulting dollar profit against the scaled reference trades. The archived plan's absolute local path has been replaced by a repository-relative path; `manifest.json` retains the original plan hash and describes that metadata-only change.

This history was inspected during strategy development. The result is a reproducible historical replay, not untouched final validation. The universe is a September 2026 S&P 500 snapshot, so historical constituent and survivorship limitations remain.

The input pickle was written using pandas 3.0.6. Install the repository requirements in a recent Python environment; current pandas can read this frozen snapshot. Treat the committed pickle as a trusted repository artifact, and retain the hash checks when changing input handling.

Local snapshots take precedence when the default cache exists. A missing default cache uses this bundle; an explicitly supplied missing alternative cache fails. Changed or corrupt inputs also fail instead of silently selecting different data.
