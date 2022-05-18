# Copyright 2022 The Procter & Gamble Company
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
Compute the optimal window size using the false nearest neighbours method.
See Kennel1992, Frank2000.
"""

import numpy as np

from utils import ts2windows, nearest_neighbor


def false_nearest_neighbours(arr, time_delay=1, Rtol=15, max_win=21600, stop_at=1):
    """
    Original criterion is |x[-1] - x_r[-1]| / dist_old > Rtol.
    However, the distance dist_old is sometimes 0, leading to an undefined criterion.
    Thus, I implement the equivalent criterion: |x[-1] - x_r[-1]| > Rtol * dist_old
    """
    argmin_win = max_win * time_delay
    min_ratio = 1
    i = 1
    step = 1
    while i < max_win + 1:
        win_sz = max((1, i * time_delay))
        windows = ts2windows(arr, win_sz, time_delay)
        if i > 1:
            embed_err = np.abs(windows[:,-1] - windows[idx,-1]).reshape((-1, 1))
            criterion = embed_err > Rtol * dist
            false_nn = np.count_nonzero(criterion)
            false_nn_ratio = false_nn / len(arr)
            print("{} pts: {} false NN ({:.2f}%)".format(
                   win_sz, false_nn, false_nn_ratio * 100))
            if false_nn_ratio < min_ratio:
                min_ratio = false_nn_ratio
                argmin_win = win_sz
                step = 1
            else:
                step = 2

            if false_nn <= stop_at:
                break

        dist, neighbr_idx = nearest_neighbor(windows)
        idx = neighbr_idx.flatten()
        i += step

    return argmin_win

