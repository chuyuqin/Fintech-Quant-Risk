import numpy as np
import pandas as pd


def calculate_returns(prices, method="DISCRETE", date_column="Date"):
    if date_column not in prices:
        raise ValueError(f"Missing date column: {date_column}")
    columns = prices.columns.drop(date_column)
    values = prices[columns].to_numpy(dtype=float)
    if np.any(values <= 0):
        raise ValueError("Prices must be positive")
    ratios = values[1:] / values[:-1]
    if method.upper() == "DISCRETE":
        returns = ratios - 1
    elif method.upper() == "LOG":
        returns = np.log(ratios)
    else:
        raise ValueError("Method must be DISCRETE or LOG")
    result = prices.iloc[1:][[date_column]].reset_index(drop=True).copy()
    return pd.concat([result, pd.DataFrame(returns, columns=columns)], axis=1)
