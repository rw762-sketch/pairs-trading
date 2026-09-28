# Pairs trading: dictionary and practice workbook

Use [the dictionary](#dictionary), attempt [the exercises](#exercises), then check [the worked answers](#worked-answers). Numerical examples are invented for teaching, not measured project results.

The project's monthly OLS experiment uses 252 preceding closes for fitting, 60 for deviations, and subsequent-close execution. These are research choices; statistical significance does not establish executable profit.

## Dictionary

### A–D

**ADF test.** The augmented Dickey–Fuller test examines a unit-root null hypothesis. Failure to reject does not establish that a series has a unit root. [2](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html)

**Adjusted close.** A historical closing-price series adjusted for specified corporate actions. Check the provider's treatment of dividends and splits before calculating returns.

**Alpha.** In `A = α + βB + ε`, α is the regression intercept. Performance alpha instead means return unexplained by a specified benchmark model.

**Annualized return.** Growth expressed on a yearly scale: `(ending equity / starting equity)^(1/years) − 1`. Annualizing a short period does not predict next year's return.

**Autocorrelation.** Association between a series and its earlier values. Persistent spread deviations can make today's observation informative about tomorrow's.

**Backtest.** A historical simulation of signals, orders, positions, costs, and account value. Its results depend on the rules and execution assumptions.

**Beta.** A price-regression slope describes A's fitted relationship with B. Market beta instead describes return sensitivity to a market benchmark; the two are different quantities.

**Bid–ask spread.** The gap between the quoted buying and selling prices. This trading friction is different from the statistical spread between two stocks.

**Bootstrap.** Repeated resampling used to estimate uncertainty. Time-series block methods preserve some local dependence; independently shuffling observations can destroy it. [10](https://arch.readthedocs.io/en/latest/bootstrap/timeseries-bootstraps.html)

**Borrow fee.** A charge for borrowing shares to sell short. Availability and rates can change; an assumed constant fee is a simplification.

**Cointegration.** Two I(1) price series are cointegrated when a nonzero linear combination is stationary. Their individual trends can wander while their adjusted difference remains stable. [3](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.coint.html)

**Confidence interval.** A range from a method with specified repeated-sampling coverage. A 95% interval does not assign 95% probability to a fixed parameter lying inside it. [5](https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm)

**Correlation.** A standardized measure of linear association, between −1 and 1. Specify whether you correlate prices or returns; high correlation alone does not establish cointegration.

**Covariance.** Average joint movement around two means. Its units depend on both variables, so dividing by their standard deviations produces correlation.

**Daily return.** Fractional change over one trading interval: `r = P_today/P_previous − 1`. In pandas, `pct_change()` gives fractions: 0.01 means 1%. [11](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pct_change.html)

**Drawdown.** Decline from the account's previous highest value. Maximum drawdown is the deepest such decline, including losses before eventual recovery.

### E–I

**Engle–Granger test.** A two-step cointegration test using a fitted relationship and its residuals. Use cointegration-specific critical values, not ordinary residual ADF p-values as substitutes. [3](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.coint.html)

**Entry and exit thresholds.** Prespecified boundaries triggering decisions. Here entry is beyond `|z| = 2`; long spread exits at `z >= -0.5`, short spread at `z <= +0.5`, including moves across zero.

**Error-correction model.** A model linking short-run changes to a lagged departure from a long-run relationship. An adjustment coefficient describes the response to that departure. [12](https://www.statsmodels.org/stable/generated/statsmodels.tsa.vector_ar.vecm.VECM.html)

**Equity curve.** Account value through time, including realized and unrealized profit or loss and charged costs. State whether deposits and withdrawals occur.

**Execution lag.** The delay between observing a signal and filling its order. Seeing today's closing price does not guarantee an executable fill at that same close.

**Formation window.** Historical observations used to select pairs and estimate parameters before a trading window begins. Rolling formation windows permit periodic reselection.

**Gross and net profit/loss.** Gross P&L combines both legs' price gains and losses. Net P&L subtracts the modeled transaction and holding costs.

**Half-life.** Estimated time for an expected deviation to halve. For an AR(1) deviation with `0 < φ < 1`, it is `ln(0.5)/ln(φ)` periods.

**Hedge ratio.** The relative position size used to construct the spread. A price-regression β of 1.5 means 1.5 B shares per A share, not equal dollars.

**Hypothesis test.** A procedure comparing data with a specified null hypothesis. Rejecting a statistical null does not establish economic profitability.

**I(0) and I(1).** Informally, I(0) is stationary; I(1) becomes stationary after one difference. ADF checks provide provisional evidence, not proof of these classifications.

**In sample.** The observations used to fit or select a model. Performance measured there benefits from having influenced the model's construction.

### L–P

**Leverage.** Exposure relative to account equity. With $20,000 gross exposure and $10,000 equity, gross leverage is 2; offsetting legs still carry risk.

**Liquidity.** The ability to trade a desired size promptly without moving the price substantially. A quoted price alone does not establish available liquidity.

**Log return.** `ln(P_today/P_previous)`. Log returns add across time, whereas simple returns compound by multiplying `1 + r`.

**Long position.** Ownership of an asset whose price gain benefits the position. Ten shares gaining $2 each produce $20 before costs.

**Look-ahead bias.** Using information unavailable at the simulated decision time. Fitting a hedge ratio on the entire backtest period leaks future observations.

**Margin.** Collateral and account requirements associated with borrowing or short positions. A broker can demand additional equity or liquidate positions. [8](https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-bulletins-29)

**Mark to market.** Revaluing open positions at current prices. Unrealized losses affect account equity even before a trade closes.

**Mean reversion.** A tendency for deviations to move back toward an estimated center. It does not guarantee a particular trade will recover.

**Multiple comparisons.** Testing many hypotheses creates more opportunities for false discoveries. Procedures such as Holm or Bonferroni adjust the family of tests. [9](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html)

**Notional and gross exposure.** A leg's notional is shares times price. Gross exposure adds the absolute notionals of the long and short legs.

**Ordinary least squares (OLS).** Fits coefficients by minimizing squared residuals. Statsmodels' array-based OLS requires an explicitly added constant when an intercept is wanted. [1](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html)

**Out of sample.** Observations excluded from a particular fitting or selection step. Once repeatedly inspected for redesign, they no longer provide a fresh final evaluation.

**Overfitting.** Choosing rules that exploit accidental patterns in the observed sample. Trying more variants and reporting only the winner can exaggerate apparent skill.

**p-value.** Under the null and model assumptions, the probability of a test statistic at least as extreme as observed. It is not the probability the strategy fails.

**Paper trading.** Recording simulated orders prospectively as new data arrive. It tests the workflow but cannot fully reproduce real fills or share-borrow availability.

**Portfolio weight.** A position's allocation as a fraction of account capital under a stated convention. Here, a 20% pair allocation is a gross-exposure budget.

### R–Z

**Regression residual.** The observed value minus its fitted value: `ε = A − α − βB`. Small residuals alone do not establish cointegration.

**Risk-free rate.** The comparison return used when calculating excess returns. Match its period to the portfolio returns and disclose any zero-rate assumption.

**Sharpe ratio.** Mean excess return divided by its standard deviation. Multiplying daily Sharpe by `√252` requires assumptions about time dependence; it does not measure every risk. [6](https://web.stanford.edu/~wfsharpe/art/sr/SR.htm)

**Short position.** Selling borrowed shares and later buying them back. A rising price causes losses; dividends and borrowing costs also matter. [7](https://www.sec.gov/investor/pubs/regsho.htm)

**Slippage.** The difference between the intended reference price and actual execution price. It can depend on order size, liquidity, timing, and market conditions.

**Standard deviation.** The square root of variance, measuring dispersion in the original units. Return dispersion is different from uncertainty about the mean return. [4](https://www.itl.nist.gov/div898/handbook/eda/section3/eda356.htm)

**Standard error.** Estimated variability of an estimator across samples. For independent, identically distributed observations, the mean's standard error is `s/√n`. [5](https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm)

**Stationarity.** In the weak sense, constant mean and finite variance, with covariance depending on lag rather than calendar date. A flat-looking chart is insufficient evidence.

**Stop-loss.** A rule requesting exit after a specified adverse move. Execution delays and price gaps can make the realized loss larger than the threshold suggests.

**Survivorship bias.** Distortion caused by studying only assets that remain available today. Today's index constituents do not reproduce the membership investors knew historically.

**Trade log.** A record of entry, exit, direction, share quantities, prices, costs, and P&L. It should reconcile with the portfolio's accounting.

**Trading spread.** Here, `S = A − βB`. It equals the regression residual plus α; its natural estimated center need not be zero.

**Transaction costs.** Expenses caused by trading, including modeled fees and execution friction. Charge both legs at entry and exit using actual traded notional.

**Turnover.** Trading activity relative to capital. This project's metric annualizes total absolute traded notional divided by initial capital; other reports may use different conventions.

**Validation.** Data used to choose among proposed rules before final evaluation. Chronological splits preserve the direction of time. [13](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)

**Variance.** Squared dispersion around the mean; sample variance uses denominator `n − 1`. Out-of-sample variance uses the evaluation returns, not formation prices. [4](https://www.itl.nist.gov/div898/handbook/eda/section3/eda356.htm)

**Volatility.** Usually the standard deviation of returns over a stated period. Annualized daily volatility commonly uses `s × √252`, with scaling assumptions.

**Z-score.** Deviation from an estimated mean measured in standard deviations: `z = (S − μ)/σ`. In this project, μ and σ use preceding observations.

## Exercises

Exercises 4–6 share one fictional position with fixed quantities.

1. **Returns:** A price rises from $50 to $55. Calculate its simple and log returns.
2. **Regression:** The fitted relation is `A = 10 + 1.5B`. If A is $88 and B is $50, calculate fitted A, residual, and trading spread.
3. **Signal:** The spread is 13, its preceding mean is 10, and standard deviation is 1. What is z? With positive β, which legs express expected convergence?
4. **Sizing:** A costs $100 and B $50. With β = 1.5 and $3,500 gross budget, size a long-spread trade.
5. **P&L:** Those shares exit at A = $102 and B = $51. Find gross P&L. Why is adding the stocks' percentage returns incorrect?
6. **Costs:** Charge 0.1% on every traded dollar at entry and exit, plus $1 total borrow cost. Find net P&L and return on $3,500 allocated capital.
7. **Uncertainty:** For 100 independent daily returns with mean 0.04% and standard deviation 1%, calculate standard error and an approximate 95% interval using 1.96.
8. **Tests:** A pair has p = 0.01 among 100 tests. Does this establish profitability? What is the 5% Bonferroni threshold? Under 100 true nulls, how many 5% false rejections are expected?
9. **Persistence:** A fitted AR(1) deviation has φ = 0.8. Estimate half-life. Does that create a guaranteed exit date?
10. **Risk:** Equity moves through 100, 110, 99, and 105. Find maximum drawdown. Separately, estimate annualized Sharpe from daily excess mean 0.04% and standard deviation 1%.
11. **Leakage:** You compare five rules, inspect their later-period returns, choose the best, and call that period untouched. Identify the problem and a better protocol.
12. **Evidence:** Daily returns occur in persistent clusters. Explain why an independent bootstrap may mislead, then name two things prospective paper trading should record.

## Worked answers

1. `55/50 − 1 = 10%`; `ln(1.1) ≈ 0.09531`, or 9.531% log return. These are different return conventions.
2. Fitted A is `10 + 1.5×50 = 85`. Residual is `88 − 85 = 3`; spread is `88 − 75 = 13`. The intercept explains the difference.
3. `z = (13 − 10)/1 = 3`. The spread is unusually high: short A and buy β shares of B per A share, if the trading rules permit entry. A positive z alone is insufficient evidence of profit.
4. Each spread unit uses `$100 + 1.5×$50 = $175` gross exposure. Buy `3500/175 = 20` A shares and short 30 B shares. Long notional is $2,000; short notional is $1,500.
5. A earns `20×2 = $40`; B loses `30×1 = $30`. Gross P&L is $10. Share quantities and short signs determine dollars; adding unweighted stock returns ignores both.
6. Entry fee: `$3500×0.001 = $3.50`. Exit fee: `(20×102 + 30×51)×0.001 = $3.57`. Net: `10 − 3.50 − 3.57 − 1 = $1.93`, approximately 0.0551% of allocated capital.
7. Standard error is `1%/√100 = 0.1%`. Approximate interval: `0.04% ± 0.196%`, giving −0.156% to 0.236%. It includes zero; dependence would invalidate this simple standard-error calculation.
8. No: a cointegration p-value does not measure trading profitability. Bonferroni requires `0.05/100 = 0.0005`; 0.01 fails. With calibrated tests under all true nulls, the expected count is `100×0.05 = 5`.
9. `ln(0.5)/ln(0.8) ≈ 3.11` observations. This describes the fitted expected decay of a deviation, not a guaranteed individual path or holding period.
10. The peak-to-trough fall is `99/110 − 1 = −10%`. Sharpe is approximately `√252 × 0.0004/0.01 = 0.635`, subject to the stated scaling assumptions.
11. Selection has used the purported final evaluation. Prespecify variants, select on earlier validation, freeze the choice, and evaluate on later data. Previously inspected dates must remain labeled exploratory; genuinely new data provide stronger subsequent evidence.
12. Independent resampling removes the clusters and can misstate uncertainty. Compare appropriate block resampling choices. Paper trading should timestamp signal availability and record requested orders, simulated fills, fees, and borrow assumptions.

## Four-week study plan

**Week 1 — Prices and accounting.** Read sources 7, 8, and 11. Complete exercises 1–6. Reconstruct a logged trade's legs and costs, then reconcile its contribution to the pair total.

**Week 2 — Regression and stationarity.** Read sources 1–3 and 12. Explain α, β, spread, and residual aloud. Complete exercises 8–9; identify the tests' assumptions.

**Week 3 — Honest evaluation.** Read sources 4–6, 9, 10, and 13. Complete exercises 7 and 10–12. Draw formation, validation, and evaluation dates; mark where results influenced decisions.

**Week 4 — Reproducible research.** Read source 14. Write one hypothesis and evaluation protocol. Explain distance selection versus cointegration. Practice prospective paper logging; simulated profits do not establish readiness for real execution.

## Primary-source reading path

Read assumptions before copying code.

1. [Statsmodels: OLS](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html) — intercept and slope.
2. [Statsmodels: ADF](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html) — unit-root testing.
3. [Statsmodels: cointegration](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.coint.html) — assumptions and critical values.
4. [NIST: measures of scale](https://www.itl.nist.gov/div898/handbook/eda/section3/eda356.htm) — variance and standard deviation.
5. [NIST: confidence limits](https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm) — sampling uncertainty.
6. [William Sharpe: The Sharpe Ratio](https://web.stanford.edu/~wfsharpe/art/sr/SR.htm) — study excess returns and time scaling.
7. [SEC: short sales, section I](https://www.sec.gov/investor/pubs/regsho.htm) — read position mechanics; older settlement information is not used here.
8. [Investor.gov: margin accounts](https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-bulletins-29) — understand borrowing and collateral.
9. [Statsmodels: multiple tests](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html) — correction procedures.
10. [arch: time-series bootstraps](https://arch.readthedocs.io/en/latest/bootstrap/timeseries-bootstraps.html) — temporal resampling.
11. [pandas: fractional changes](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pct_change.html) — returns and units.
12. [Statsmodels: error correction](https://www.statsmodels.org/stable/generated/statsmodels.tsa.vector_ar.vecm.VECM.html) — connect long-run relationships with short-run changes.
13. [scikit-learn: chronological splits](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) — understand ordered evaluation.
14. [Gatev, Goetzmann, and Rouwenhorst: original pairs-trading research](https://depot.som.yale.edu/icf/papers/fileuploads/2573/original/08-03.pdf) — study normalized-price distance, formation, and trading windows; its selection method differs from cointegration.
