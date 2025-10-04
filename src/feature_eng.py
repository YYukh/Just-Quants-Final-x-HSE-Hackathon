import pandas as pd
import numpy as np

def generate_features(df, 
                      window_short=5, 
                      window_mid=20, 
                      window_long=60,
                      window_vol=10,
                      alpha=0.1,
                      winsorize_limit=0.01):
    """
    Generate comprehensive financial features: advanced + classical + volatility + lags.
    
    Parameters:
    - df: pd.DataFrame with MultiIndex columns: (['open','high','low','close','volume'], [tickers])
    - window_short/mid/long: windows for moving averages and momentum
    - window_vol: window for volatility estimation
    - alpha: smoothing for EWMA
    - winsorize_limit: fraction to winsorize (e.g., 0.01 = 1% tails clipped)
    
    Returns:
    - df_out: DataFrame with original + all features, same MultiIndex structure
    """
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Input DataFrame must have MultiIndex columns: (price_type, ticker)")
    
    tickers = df['close'].columns
    df_out = df.copy()

    # --- Базовые доходности ---
    log_ret = np.log(df['close'] / df['close'].shift(1))
    simple_ret = df['close'].pct_change()
    
    # Уберите импорт winsorize
    # И используйте свою функцию
    def winsorize_series(series, limits=(0.01, 0.01)):
        if series.empty:
            return series.copy()
        lower_q = series.quantile(limits[0])
        upper_q = series.quantile(1 - limits[1])
        return series.clip(lower=lower_q, upper=upper_q)

    # Внутри generate_features:
    winsorized_ret = log_ret.copy()
    for ticker in tickers:
        winsorized_ret[ticker] = winsorize_series(
            log_ret[ticker].fillna(0), 
            limits=(winsorize_limit, winsorize_limit)
        )
    
    log_ret.columns = pd.MultiIndex.from_product([['log_return'], tickers])
    simple_ret.columns = pd.MultiIndex.from_product([['simple_return'], tickers])
    winsorized_ret.columns = pd.MultiIndex.from_product([['winsorized_log_return'], tickers])
    
    df_out = pd.concat([df_out, log_ret, simple_ret, winsorized_ret], axis=1)

    # --- Лаги (до 3 дней назад) ---
    for lag in [1, 2, 3]:
        lag_ret = log_ret.shift(lag)
        lag_ret.columns = pd.MultiIndex.from_product([[f'log_return_lag_{lag}'], tickers])
        df_out = pd.concat([df_out, lag_ret], axis=1)

    # --- Скользящие средние и отклонения ---
    ma_short = df['close'].rolling(window_short).mean()
    ma_mid = df['close'].rolling(window_mid).mean()
    ma_long = df['close'].rolling(window_long).mean()
    
    # Отношение цены к MA
    price_ma_short = df['close'] / (ma_short + 1e-8)
    price_ma_mid = df['close'] / (ma_mid + 1e-8)
    price_ma_long = df['close'] / (ma_long + 1e-8)
    
    # Пересечения MA (моментум)
    ma_cross_short_mid = (ma_short > ma_mid).astype(int)
    ma_cross_mid_long = (ma_mid > ma_long).astype(int)
    
    for name, data in [
        ('ma_short', ma_short),
        ('ma_mid', ma_mid),
        ('ma_long', ma_long),
        ('price_ma_short', price_ma_short),
        ('price_ma_mid', price_ma_mid),
        ('price_ma_long', price_ma_long),
        ('ma_cross_short_mid', ma_cross_short_mid),
        ('ma_cross_mid_long', ma_cross_mid_long)
    ]:
        data.columns = pd.MultiIndex.from_product([[name], tickers])
        df_out = pd.concat([df_out, data], axis=1)

    # --- Волатильность ---
    # 1. Rolling std of log returns
    vol_rolling = log_ret.rolling(window_vol).std()
    
    # 2. EWMA volatility
    vol_ewma = log_ret.ewm(alpha=alpha).std()
    
    # 3. Parkinson volatility (high-low proxy)
    parkinson_vol = (1 / (4 * np.log(2))) * (np.log(df['high'] / df['low']) ** 2)
    parkinson_vol = np.sqrt(parkinson_vol.rolling(window_vol).mean())
    
    for name, data in [
        ('vol_rolling', vol_rolling),
        ('vol_ewma', vol_ewma),
        ('vol_parkinson', parkinson_vol)
    ]:
        data.columns = pd.MultiIndex.from_product([[name], tickers])
        df_out = pd.concat([df_out, data], axis=1)

    # --- Моментум и ускорение ---
    momentum_short = df['close'] / df['close'].shift(window_short) - 1
    momentum_mid = df['close'] / df['close'].shift(window_mid) - 1
    
    # Ускорение: изменение момента
    acceleration = momentum_short - momentum_short.shift(1)
    
    for name, data in [
        ('momentum_short', momentum_short),
        ('momentum_mid', momentum_mid),
        ('acceleration', acceleration)
    ]:
        data.columns = pd.MultiIndex.from_product([[name], tickers])
        df_out = pd.concat([df_out, data], axis=1)

    # --- RSI-подобный осциллятор (без циклов) ---
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    
    rsi.columns = pd.MultiIndex.from_product([['rsi_14'], tickers])
    df_out = pd.concat([df_out, rsi], axis=1)

    # --- Advanced features from previous version ---
    # Intraday Range
    intraday_range = (df['high'] - df['low']) / (df['close'] + 1e-8)
    intraday_range.columns = pd.MultiIndex.from_product([['intraday_range'], tickers])
    df_out = pd.concat([df_out, intraday_range], axis=1)
    
    # Close vs Open
    close_open = (df['close'] - df['open']) / (df['open'] + 1e-8)
    close_open.columns = pd.MultiIndex.from_product([['close_open'], tickers])
    df_out = pd.concat([df_out, close_open], axis=1)
    
    # Position in range
    position_in_range = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    position_in_range.columns = pd.MultiIndex.from_product([['position_in_range'], tickers])
    df_out = pd.concat([df_out, position_in_range], axis=1)
    
    # Body-to-range
    body = (df['close'] - df['open']).abs()
    body_range_ratio = body / (df['high'] - df['low'] + 1e-8)
    body_range_ratio.columns = pd.MultiIndex.from_product([['body_range_ratio'], tickers])
    df_out = pd.concat([df_out, body_range_ratio], axis=1)
    
    # EWMA return (уже есть, но оставим для ясности)
    ewma_ret = simple_ret.ewm(alpha=alpha).mean()
    ewma_ret.columns = pd.MultiIndex.from_product([['ewma_return'], tickers])
    df_out = pd.concat([df_out, ewma_ret], axis=1)
    
    # Volume spike
    vol_ma = df['volume'].rolling(window=3, min_periods=1).mean()
    volume_spike = df['volume'] / (vol_ma + 1e-8)
    volume_spike.columns = pd.MultiIndex.from_product([['volume_spike'], tickers])
    df_out = pd.concat([df_out, volume_spike], axis=1)
    
    # Robust Z-score
    q75 = df['close'].rolling(window=20, min_periods=1).quantile(0.75)
    q25 = df['close'].rolling(window=20, min_periods=1).quantile(0.25)
    robust_z = (df['close'] - q25) / (q75 - q25 + 1e-8)
    robust_z.columns = pd.MultiIndex.from_product([['robust_z_score'], tickers])
    df_out = pd.concat([df_out, robust_z], axis=1)
    
    # Micro noise
    micro_noise = (df['high'] - df['low']) / (df['close'] + 1e-8) - close_open.abs()
    micro_noise.columns = pd.MultiIndex.from_product([['micro_noise'], tickers])
    df_out = pd.concat([df_out, micro_noise], axis=1)
    
    # Overnight gap
    gap = df['open'] / (df['close'].shift(1) + 1e-8) - 1
    gap.columns = pd.MultiIndex.from_product([['overnight_gap'], tickers])
    df_out = pd.concat([df_out, gap], axis=1)
    
    # VWAP deviation
    vwap = (df['high'] + df['low'] + df['close']) / 3.0
    vw_dev = (df['close'] - vwap) * df['volume']
    vw_dev.columns = pd.MultiIndex.from_product([['vw_price_dev'], tickers])
    df_out = pd.concat([df_out, vw_dev], axis=1)

    return df_out.sort_index(axis=1)