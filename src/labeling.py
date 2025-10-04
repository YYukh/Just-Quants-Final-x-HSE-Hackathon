import datetime as dt
import numpy as np
import pandas as pd


def applyPtSlOnT1(close, events, ptSl):
    """Tripple barrier method

    Args:
        close (pd.Series): Close prices.
        events (pd.DataFrame): A pandas dataframe, with columns:
            - t1: The timestamp of vertical barrier. When the value is np.nan, there will not be a vertical barrier.
            - trgt: The unit width of the horizontal barriers.
        ptSl (list):
            - ptSl[0]: The factor that multiplies trgt to set the width of the upper barrier.
                If 0, there will not be an upper barrier.
            - ptSl[1]: The factor that multiplies trgt to set the width of the lower barrier.
                If 0, there will not be a lower barrier.

    Returns:
        pd.DataFrame: Timestamps of each barrier touch
    """

    # apply stop loss/profit taking, if it takes place before t1 (end of event)
    out = events[["t1"]].copy(deep=True)
    if ptSl[0] > 0:
        pt = ptSl[0] * events["trgt"]
    else:
        pt = pd.Series(index=events.index)  # NaNs
    if ptSl[1] > 0:
        sl = -ptSl[1] * events["trgt"]
    else:
        sl = pd.Series(index=events.index)  # NaNs
    for loc, t1 in events["t1"].fillna(close.index[-1]).items():
        df0 = close[loc:t1]  # path prices
        df0 = df0 / close[loc] - 1  # *events.at[loc,'side'] # path returns
        out.loc[loc, "sl"] = df0[df0 < sl[loc]].index.min()  # earliest stop loss.
        out.loc[loc, "pt"] = df0[df0 > pt[loc]].index.min()  # earliest profit taking.
    return out


def three_barrier_std(close, ptSl=[1, 1], rolling_n=50, base_scaling_factor=2.0, base_period=20, period=20):
    """Labeling based on tripple barrier method and historical volatility

    Args:
        close (pd.DataFrame): Assets close prices
        ptSl (list, optional): applyPtSlOnT1 ptSl parameter. Defaults to [1, 1].
        rolling_n (int, optional): Rolling window to calculate standart deviation. Defaults to 50.
        scaling_factor (float, optional): Multiplier of standart deviation to calculate horizontal barrier size. Defaults to 2.0.

    Returns:
        pd.Series: 3-class labels
    """
    scaling_factor = base_scaling_factor * np.sqrt(period / base_period)
    events = pd.DataFrame(
        {
            "t1": close.index + dt.timedelta(days=period),
            "trgt": close.pct_change().rolling(rolling_n).std() * scaling_factor,
        }
    )
    out = applyPtSlOnT1(close, events, ptSl)
    
    for idx in out.index:
        sl_val = out.at[idx, 'sl']
        pt_val = out.at[idx, 'pt']
        if pd.notna(sl_val) or pd.notna(pt_val):
            out.at[idx, 't1'] = pd.NaT

    target = out.apply(
        lambda x: 1 if x.idxmin() == "pt" else -1 if x.idxmin() == "sl" else 0, axis=1
    )

    return target


def barrier_std(close, ptSl=[1, 1], rolling_n=50, base_scaling_factor=2.0, base_period=20, period=20):
    """Labeling based on tripple barrier method and historical volatility

    Args:
        close (pd.DataFrame): Assets close prices
        ptSl (list, optional): applyPtSlOnT1 ptSl parameter. Defaults to [1, 1].
        rolling_n (int, optional): Rolling window to calculate standart deviation. Defaults to 50.
        scaling_factor (float, optional): Multiplier of standart deviation to calculate horizontal barrier size. Defaults to 2.0.

    Returns:
        pd.Series: 3-class labels
    """
    scaling_factor = base_scaling_factor * np.sqrt(period / base_period)
    events = pd.DataFrame(
        {
            "t1": close.index + dt.timedelta(days=period),
            "trgt": close.pct_change().rolling(rolling_n).std() * scaling_factor,
        }
    )

    return events["trgt"]