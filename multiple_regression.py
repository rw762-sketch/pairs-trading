"""Formation-only multiple OLS regression used as a pairs entry filter."""
import numpy as np
import pandas as pd

FEATURES = ['deviation', 'momentum_1', 'momentum_5', 'volatility_20', 'peer_return_5']
RULES = {'formation_days': 252, 'minimum_rows': 120, 'entry_step': 1,
         'exit_step': 6, 'cost_multiple': 2., 'borrow_days': 7,
         'fee': .001, 'borrow': .02, 'max_condition_number': 1e6}


def features(prices, ticker1, ticker2, alpha, beta):
    """All predictors use prices through the observation close only."""
    a, b = prices[ticker1], prices[ticker2]
    spread = a - alpha - beta * b
    gross = a + beta * b
    daily = spread.diff() / gross.shift(1)
    frame = pd.DataFrame({
        'deviation': (spread - spread.rolling(60).mean().shift(1)) / gross,
        'momentum_1': spread.diff() / gross,
        'momentum_5': spread.diff(5) / gross,
        'volatility_20': daily.rolling(20).std(),
        'peer_return_5': b.pct_change(5, fill_method=None),
    })
    return frame.replace([np.inf, -np.inf], np.nan), spread, gross


def training_data(formation, ticker1, ticker2, alpha, beta):
    x, spread, gross = features(formation, ticker1, ticker2, alpha, beta)
    # Slice formation BEFORE shifting; labels must finish before trading starts.
    y = (spread.shift(-RULES['exit_step']) - spread.shift(-RULES['entry_step'])) / gross
    return x.assign(target=y).dropna()


def fit_model(training):
    model = {'valid': False, 'status': 'insufficient_rows', 'observations': len(training)}
    if len(training) < RULES['minimum_rows']:
        return model
    x, y = training[FEATURES].to_numpy(), training.target.to_numpy()
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return dict(model, status='nonfinite_input')
    means, scales = x.mean(axis=0), x.std(axis=0)
    if (scales <= 1e-12).any():
        return dict(model, status='constant_predictor')
    design = np.column_stack([np.ones(len(x)), (x - means) / scales])
    coefficients, _, rank, singular = np.linalg.lstsq(design, y, rcond=None)
    condition = float(singular[0] / singular[-1]) if singular[-1] else float('inf')
    model.update(rank=int(rank), condition_number=condition)
    if rank != len(FEATURES) + 1 or condition > RULES['max_condition_number']:
        return dict(model, status='unidentified_or_ill_conditioned')
    if not np.isfinite(coefficients).all():
        return dict(model, status='nonfinite_coefficients')
    error = y - design @ coefficients
    denominator = np.sum((y - y.mean()) ** 2)
    model.update(valid=True, status='valid', means=means.tolist(), scales=scales.tolist(),
                 coefficients=coefficients.tolist(), training_rmse=float(np.sqrt(np.mean(error ** 2))),
                 training_r2=float(1 - np.sum(error ** 2) / denominator) if denominator else None)
    return model


def predict(model, x):
    if not model['valid']:
        return np.full(len(x), np.nan)
    design = np.column_stack([np.ones(len(x)),
                              (x[FEATURES].to_numpy() - model['means']) / model['scales']])
    return design @ model['coefficients']


def apply_gate(signals, prediction):
    """Forecast filters entries; existing z-score exits remain unconditional."""
    output = signals.copy()
    z = signals.ZScore.to_numpy()
    direction = np.where(z < -2, 1, np.where(z > 2, -1, 0))
    gross = signals.Price1 + signals.Hedge_Ratio * signals.Price2
    short_fraction = np.where(direction == 1, signals.Hedge_Ratio * signals.Price2, signals.Price1) / gross
    cost = 2 * RULES['fee'] + RULES['borrow'] * RULES['borrow_days'] / 365.25 * short_fraction
    passed = np.asarray(np.isfinite(prediction) & (direction != 0) & (direction * prediction > RULES['cost_multiple'] * cost))
    position, positions, reasons = 0, [], []
    for i, value in enumerate(z):
        reason = ''
        if not np.isfinite(value):
            position, reason = 0, 'invalid_zscore'
        elif position == 0 and passed[i]:
            position, reason = int(direction[i]), 'entry_multiple_regression'
        elif (position == 1 and value >= -.5) or (position == -1 and value <= .5):
            position, reason = 0, 'mean_reversion'
        positions.append(position)
        reasons.append(reason)
    output['Signal'], output['Signal_Reason'] = positions, reasons
    output['Signal_Change'] = output.Signal.diff().fillna(output.Signal)
    output['Prediction'] = prediction
    output['Estimated_Roundtrip_Cost'] = cost
    output['Entry_Gate_Passed'] = passed
    return output


class RegressionGate:
    """Callable extension to the existing, unchanged allocation/execution engine."""
    def __init__(self):
        self.models, self.diagnostics = [], []

    def __call__(self, signals, prices, window, pair):
        formation = prices.loc[window['formation_start']:window['formation_end']]
        training = training_data(formation, pair.Ticker1, pair.Ticker2, pair.Alpha, pair.Beta)
        model = fit_model(training)
        self.models.append(dict(model, Pair=pair.Pair, Formation_End=str(window['formation_end']),
                                Last_Training_Feature_Date=str(training.index[-1]) if len(training) else None,
                                Trading_Start=str(signals.index[0]), Alpha=pair.Alpha, Beta=pair.Beta))
        # Restrict input to this month's last close; predictors themselves are causal.
        x, _, _ = features(prices.loc[:signals.index[-1]], pair.Ticker1, pair.Ticker2, pair.Alpha, pair.Beta)
        output = apply_gate(signals, predict(model, x.loc[signals.index]))
        audit = output.join(x.loc[signals.index]).reset_index(names='Date')
        audit['Pair'], audit['Formation_End'] = pair.Pair, window['formation_end']
        self.diagnostics.append(audit)
        return output
