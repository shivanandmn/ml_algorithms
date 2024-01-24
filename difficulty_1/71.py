## 71 has all the code only the distance metric would change to time series
## untested

import numpy as np


def dynamic_time_warping(series1, series2):
    dtw = np.zeros((len(series1), len(series2)))
    dtw.fill(np.inf)
    dtw[0, 0] = 0
    for i in range(1, len(series1) + 1):
        for j in range(1, len(series2) + 1):
            cost = abs(series1[i - 1] - series2[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return dtw[len(series1), len(series2)]
