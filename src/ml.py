import pandas as pd


def _as_long(frame: pd.DataFrame) -> pd.Series:
    long = frame.stack()
    long.index = long.index.set_names(['date', 'asset'])
    return long


def prepare_ml_features(
    prices: dict[str, pd.DataFrame],
    regimes: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.Series]:
    """Build one feature row per date and ticker. One ticker is one column."""
    closes = prices['close']
    features = {
        'ret_1d': prices['return'],
        'ret_5d': closes / closes.shift(5) - 1,
        'ret_20d': closes / closes.shift(20) - 1,
        'ret_60d': closes / closes.shift(60) - 1,
        'dist_sma20': closes / closes.rolling(20).mean() - 1,
        'dist_sma50': closes / closes.rolling(50).mean() - 1,
        'dist_sma200': closes / closes.rolling(200).mean() - 1,
        'rolling_vol': regimes['rolling_vol'],
        'volume_ratio': prices['volume'] / prices['volume'].rolling(20).mean(),
    }
    long = pd.concat({name: _as_long(frame) for name, frame in features.items()}, axis=1)
    dummies = pd.get_dummies(_as_long(regimes['regime']), prefix='regime', dtype=float)
    target = _as_long(prices['forward_return']).rename('target')
    combined = pd.concat([long, dummies, target], axis=1).dropna()
    X = combined.drop(columns=['target'])
    y = combined['target']
    return X, y


def train_test_split_by_date(
    X: pd.DataFrame,
    y: pd.Series,
    split_date: str = '2022-01-01',
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split train and test by date, with no shuffle."""
    dates = X.index.get_level_values('date')
    split = pd.Timestamp(split_date)
    train_mask = dates < split
    test_mask = dates >= split
    return X[train_mask], X[test_mask], y[train_mask], y[test_mask]


def train_model_and_get_signals(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> pd.DataFrame:
    """Fit a sklearn model and return 0/1 target weights on the test dates."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scores = pd.Series((y_pred > 0).astype(float), index=X_test.index)
    assets = X_test.index.get_level_values('asset').unique()
    return scores.unstack('asset').reindex(columns=assets)
