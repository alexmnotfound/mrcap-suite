import numpy as np
from typing import List, Union

def add_ema(data: List[Union[float, int]], period: int = 14) -> List[float]:
    """
    Calculate Exponential Moving Average (EMA)
    
    Args:
        data: List of prices
        period: EMA period (default: 14)
    
    Returns:
        List of EMA values
    """
    data = np.array(data, dtype=float)
    multiplier = 2 / (period + 1)
    ema_values = [float(data[0])]  # Convert first value to Python float
    
    for price in data[1:]:
        ema_values.append(float((price * multiplier) + (ema_values[-1] * (1 - multiplier))))
    
    return ema_values 