import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from src.ml import prepare_ml_features, train_model_and_get_signals, train_test_split_by_date
from src.signals import compute_macd_signal, compute_volatility_scaled_signal


def sma_cross(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    """Long when the fast SMA is above the slow SMA."""
    sma_fast = df['close'].rolling(fast).mean()
    sma_slow = df['close'].rolling(slow).mean()
    return (sma_fast > sma_slow).astype(float)


def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal_span: int = 9) -> pd.Series:
    """Long when DIF is above DEA."""
    return compute_macd_signal(df, fast=fast, slow=slow, signal_span=signal_span).astype(float)


def rsi(df: pd.DataFrame, window: int = 14, oversold: float = 35.0, overbought: float = 65.0) -> pd.Series:
    """Long on oversold RSI, flat on overbought RSI."""
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    signals = pd.Series(np.nan, index=df.index)
    signals[rsi < oversold] = 1.0
    signals[rsi > overbought] = 0.0
    return signals.ffill().fillna(0.0)


def bollinger(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.Series:
    """Long on a close above the upper band, flat on a close below the middle band."""
    mid_band = df['close'].rolling(window).mean()
    std = df['close'].rolling(window).std()
    upper_band = mid_band + num_std * std

    signals = pd.Series(np.nan, index=df.index)
    signals[df['close'] > upper_band] = 1.0
    signals[df['close'] < mid_band] = 0.0
    return signals.ffill().fillna(0.0)


def random_forest(
    df: pd.DataFrame,
    split_date: str = '2022-01-01',
    n_estimators: int = 100,
    max_depth: int = 3,
) -> pd.Series:
    """Train before split_date. Long when the predicted next return is positive."""
    X, y = prepare_ml_features(df)
    X_train, X_test, y_train, _y_test = train_test_split_by_date(X, y, split_date=split_date)

    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    predicted = train_model_and_get_signals(model, X_train, y_train, X_test)

    signals = pd.Series(0.0, index=df.index)
    signals.loc[predicted.index] = predicted
    return signals


def ridge(df: pd.DataFrame, split_date: str = '2022-01-01', alpha: float = 10.0) -> pd.Series:
    """Train before split_date. Long when the predicted next return is positive."""
    X, y = prepare_ml_features(df)
    X_train, X_test, y_train, _y_test = train_test_split_by_date(X, y, split_date=split_date)

    model = Ridge(alpha=alpha)
    predicted = train_model_and_get_signals(model, X_train, y_train, X_test)

    signals = pd.Series(0.0, index=df.index)
    signals.loc[predicted.index] = predicted
    return signals


def pca(
    df: pd.DataFrame,
    split_date: str = '2022-01-01',
    n_components: int = 3,
) -> pd.Series:
    """Regress the next return on principal components fit to the training features."""
    X, y = prepare_ml_features(df)
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
    predicted = pd.Series((model.predict(Z_test) > 0).astype(float), index=X_test.index)

    signals = pd.Series(0.0, index=df.index)
    signals.loc[predicted.index] = predicted
    return signals


def hybrid(
    df: pd.DataFrame,
    split_date: str = '2022-01-01',
    target_vol: float = 0.12,
) -> pd.Series:
    """Long only in an uptrend when the forest predicts a positive return, then scale by volatility."""
    bull_market = df['close'] > df['close'].rolling(200).mean()
    ml_signal = random_forest(df, split_date=split_date)
    combined = (bull_market & (ml_signal > 0)).astype(float)
    return compute_volatility_scaled_signal(df, combined, target_vol=target_vol)
