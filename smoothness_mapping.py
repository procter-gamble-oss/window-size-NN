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
Compute the optimal window size using the smoothness of mapping.
In the original paper, the mapping is just an index translation of the input data.
In the case of NILM, the input data is aggregated, and the mapping is the appliance submetered data.
See Kil1997.
"""

import numpy as np

from utils import ts2windows, l1norm, nearest_neighbor


def smoothness(win_dist, neighbr_idx, mappedw):
    map_dist = l1norm(mappedw, mappedw[neighbr_idx.flatten()])
    win_dist = np.where(win_dist == 0, 1e-9, win_dist)
    return np.sum(map_dist / win_dist) / len(win_dist)


def smoothness_drop(arr, mapped_arr, time_delay=1, patience=0, min_inc=0.01):
    argmin_win = time_delay
    since_last_min = 0
    i = 1
    while i < len(arr) // time_delay:
        win_sz = max((1, i * time_delay))
        windows = ts2windows(arr, win_sz, time_delay)
        mappedw = ts2windows(mapped_arr, win_sz, time_delay)
        win_dist, neighbr_idx = nearest_neighbor(windows)
        smooth = smoothness(win_dist, neighbr_idx, mappedw)
        smooth = np.round(smooth, 1)
        if i > 1:
            smooth_inc = (smooth - smooth_old) / smooth_old
            print("{}\t{} pts: smoothness {:.1f} ({:.1f}%)".format(
                    since_last_min, win_sz, smooth, smooth_inc * 100))
            if smooth < smooth_old and abs(smooth_inc) >= min_inc:
                smooth_old = smooth
                argmin_win = win_sz
                since_last_min = 0
            else:
                since_last_min += 1

            if since_last_min > patience:
                break
        else:
            smooth_old = smooth

        i += 1 + since_last_min

    return argmin_win

