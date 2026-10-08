import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from src.ml import prepare_ml_features, train_model_and_get_signals, train_test_split_by_date
from src.signals import compute_macd_signal, compute_volatility_scaled_signal


def _cap_gross_exposure(weights: pd.DataFrame) -> pd.DataFrame:
    """Scale each day so absolute weights sum to at most 1."""
    gross = weights.abs().sum(axis=1)
    scale = gross.where(gross > 1, 1.0)
    return weights.div(scale, axis=0)


def sma_cross(closes: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.DataFrame:
    """Long when the fast SMA is above the slow SMA."""
    sma_fast = closes.rolling(fast).mean()
    sma_slow = closes.rolling(slow).mean()
    return _cap_gross_exposure((sma_fast > sma_slow).astype(float))


def macd(closes: pd.DataFrame, fast: int = 12, slow: int = 26, signal_span: int = 9) -> pd.DataFrame:
    """Long when DIF is above DEA."""
    raw = compute_macd_signal(closes, fast=fast, slow=slow, signal_span=signal_span).astype(float)
    return _cap_gross_exposure(raw)


def rsi(closes: pd.DataFrame, window: int = 14, oversold: float = 35.0, overbought: float = 65.0) -> pd.DataFrame:
    """Long on oversold RSI, flat on overbought RSI."""
    delta = closes.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_value = 100.0 - (100.0 / (1.0 + rs))

    weights = pd.DataFrame(np.nan, index=closes.index, columns=closes.columns)
    weights = weights.mask(rsi_value < oversold, 1.0)
    weights = weights.mask(rsi_value > overbought, 0.0)
    return _cap_gross_exposure(weights.ffill().fillna(0.0))


def bollinger(closes: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Long on a close above the upper band, flat on a close below the middle band."""
    mid_band = closes.rolling(window).mean()
    std = closes.rolling(window).std()
    upper_band = mid_band + num_std * std

    weights = pd.DataFrame(np.nan, index=closes.index, columns=closes.columns)
    weights = weights.mask(closes > upper_band, 1.0)
    weights = weights.mask(closes < mid_band, 0.0)
    return _cap_gross_exposure(weights.ffill().fillna(0.0))


def _fit_weight_frame(prices, regimes, model, split_date: str) -> pd.DataFrame:
    closes = prices['close']
    X, y = prepare_ml_features(prices, regimes)
    X_train, X_test, y_train, _y_test = train_test_split_by_date(X, y, split_date=split_date)
    predicted = train_model_and_get_signals(model, X_train, y_train, X_test)
    weights = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    weights.loc[predicted.index, predicted.columns] = predicted
    return _cap_gross_exposure(weights.fillna(0.0))


def random_forest(
    prices: dict[str, pd.DataFrame],
    regimes: dict[str, pd.DataFrame],
    split_date: str = '2022-01-01',
    n_estimators: int = 100,
    max_depth: int = 3,
) -> pd.DataFrame:
    """Train before split_date. Long when the predicted next return is positive."""
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    return _fit_weight_frame(prices, regimes, model, split_date)


def ridge(
    prices: dict[str, pd.DataFrame],
    regimes: dict[str, pd.DataFrame],
    split_date: str = '2022-01-01',
    alpha: float = 10.0,
) -> pd.DataFrame:
    """Train before split_date. Long when the predicted next return is positive."""
    model = Ridge(alpha=alpha)
    return _fit_weight_frame(prices, regimes, model, split_date)


def pca(
    prices: dict[str, pd.DataFrame],
    regimes: dict[str, pd.DataFrame],
    split_date: str = '2022-01-01',
    n_components: int = 3,
) -> pd.DataFrame:
    """Regress the next return on principal components fit to the training features."""
    closes = prices['close']
    X, y = prepare_ml_features(prices, regimes)
    X_train, X_test, y_train, _y_test = train_test_split_by_date(X, y, split_date=split_date)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    n_components = min(n_components, X_train_scaled.shape[0], X_train_scaled.shape[1])
    pca_model = PCA(n_components=n_components)
    Z_train = pca_model.fit_transform(X_train_scaled)
    Z_test = pca_model.transform(X_test_scaled)

    model = LinearRegression()
    model.fit(Z_train, y_train)
    scores = pd.Series((model.predict(Z_test) > 0).astype(float), index=X_test.index)
    predicted = scores.unstack('asset')

    weights = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    weights.loc[predicted.index, predicted.columns] = predicted
    return _cap_gross_exposure(weights.fillna(0.0))


def hybrid(
    prices: dict[str, pd.DataFrame],
    regimes: dict[str, pd.DataFrame],
    split_date: str = '2022-01-01',
    target_vol: float = 0.12,
) -> pd.DataFrame:
    """Long only in an uptrend when the forest predicts a positive return, then scale by volatility."""
    closes = prices['close']
    bull_market = closes > closes.rolling(200).mean()
    ml_weights = random_forest(prices, regimes, split_date=split_date)
    combined = (bull_market & (ml_weights > 0)).astype(float)
    scaled = compute_volatility_scaled_signal(regimes['rolling_vol'], combined, target_vol=target_vol)
    return _cap_gross_exposure(scaled)
