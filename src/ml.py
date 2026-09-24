import pandas as pd
import numpy as np

def prepare_ml_features(df: pd.DataFrame, target_type: str = 'continuous') -> tuple[pd.DataFrame, pd.Series]:
    feats = pd.DataFrame(index=df.index)
    
    # 1. 动量特征
    feats['ret_1d'] = df['return']
    feats['ret_5d'] = df['close'] / df['close'].shift(5) - 1
    feats['ret_20d'] = df['close'] / df['close'].shift(20) - 1
    feats['ret_60d'] = df['close'] / df['close'].shift(60) - 1
    
    # 2. 均线相对偏离度
    feats['dist_sma20'] = df['close'] / df['close'].rolling(20).mean() - 1
    feats['dist_sma50'] = df['close'] / df['close'].rolling(50).mean() - 1
    feats['dist_sma200'] = df['close'] / df['close'].rolling(200).mean() - 1
    
    # 3. 波动率与量能特征
    feats['rolling_vol'] = df['rolling_vol']
    feats['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    # 4. 市场状态 One-Hot 哑变量
    if 'regime' in df.columns:
        regime_dummies = pd.get_dummies(df['regime'], prefix='regime', dtype=float)
        feats = pd.concat([feats, regime_dummies], axis=1)
        
    # 5. 目标变量生成
    if target_type == 'binary':
        target = (df['forward_return'] > 0).astype(int).rename('target')
    else:
        target = df['forward_return'].rename('target')
        
    # 6. 严防 NaN：统一合并清洗后切分
    combined = pd.concat([feats, target], axis=1).dropna()
    
    X = combined.drop(columns=['target'])
    y = combined['target']
    
    return X, y

def train_test_split_by_date(
    X: pd.DataFrame, 
    y: pd.Series, 
    split_date: str = '2022-01-01'
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """严格按日期切分训练集与测试集 (严防时间泄露)"""
    train_mask = X.index < split_date
    test_mask = X.index >= split_date
    
    return X[train_mask], X[test_mask], y[train_mask], y[test_mask]

def train_model_and_get_signals(
    model, 
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_test: pd.DataFrame
) -> pd.Series:
    """训练任意给定的 sklearn 模型，并生成测试集上的 0/1 目标持仓信号"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # 预测明日收益大于 0 则做多，否则空仓
    signals = pd.Series((y_pred > 0).astype(float), index=X_test.index)
    return signals