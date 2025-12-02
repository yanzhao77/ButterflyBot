# utils/timeframe_utils.py
import re

def timeframe_to_pandas_freq(tf: str) -> str:
    match = re.match(r'^(\d+)([mhdw])$', tf.lower())
    if not match:
        return 'H'
    n, unit = match.groups()
    mapping = {'m': 'T', 'h': 'H', 'd': 'D', 'w': 'W'}
    return f"{n}{mapping[unit]}"