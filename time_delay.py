# Copyright 2021 The Procter & Gamble Company
#
# Licenced under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the original License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Function to get the optimal time delay from chaotic time series.
See: Liebert1989
The time delay is the minimum period between two uncorrelated points.
time delay >= sampling period
"""

import numpy as np

from utils import l1norm, ts2windows


def generalized_corr_integral(arr, radius):
    """
    The matrix of pairwise distances is symmetric with a null diagonal,
    by permutation of the norm.
    This function only computes the upper triangle.
    """
    n = len(arr)
    n_pairs = n * (n - 1) // 2
    dist_mat = np.zeros((n_pairs, 1), dtype="float32")
    idx = 0
    for i, x in enumerate(arr):
        x2 = arr[i+1:n]
        x1 = np.ones_like(x2) * x
        w = len(x2)
        dist_mat[idx:idx+w] = l1norm(x1, x2)
        idx += w

    return np.count_nonzero(dist_mat < radius) / n_pairs


def find_min_time_delay(arr, sample_period, radius=10, embedding=1):
    """
    For electrical power time series, the `radius` is in Watt.
    /!\ As the generalized integral requires to compute the pair-wise difference
    between each point, the method scales poorly.
    The original paper uses not more than 10k points for `arr`.
    """
    min_ln_corr = 1e9
    argmin_time_delay = 1
    for time_delay in range(1, len(arr)):
        # Sub-sampling
        if embedding < 2:
            subset = arr[::time_delay]
        else:
            subset = ts2windows(arr, embedding, time_delay)

        corr = np.log(generalized_corr_integral(subset, radius))
        if corr <= min_ln_corr:
            min_ln_corr = corr
            argmin_time_delay = time_delay
        else:
            # First minimum is found, stop.
            break

    return argmin_time_delay * sample_period


# Observations:
# - REFIT & UK-DALE: time delay in seconds
#   radius (W)      | 0.001  0.01  0.1   1  10  50  100  1000   10000  thr
#   ----------------------------------------------------------------------
#   kettle          |    29    29   29  36  36  38   38    31  308547  38
#   dish washer     |    40    40   40  47  44  36   38    42  308547  36
#   washing machine |    33    33   33  29  38  36   31    36  308547  44
#
#   No difference in the time delay for radii below 1.
#   Radius 10000 is higher than the signal amplitude, returns garbage.
#   Using activation threshold (thr column) looks like a good compromise as we
#   wish to neglect noise.
#
# - Time delay algorithm with embedding of 1 returns a delay matching the
#   shortest event in the given time series.
#   For kettle, if an activation lasts only 1 point (peak), then the time delay
#   will be equal to the sampling period.
#   => Time delay algortithm is sensitive to the data quality.
#
# - Increasing the embedding to 2, 3, 6, 30 returns a shorter time delay
#   matching the sampling period.

