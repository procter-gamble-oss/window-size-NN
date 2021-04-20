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
Approach: get the min, max & median length from all datasets
Algorithm:
  1. activation detection with thresholding
  2. window expansion (+ X min margin on each side)
  3. exclude windows of low energy (noise), E < threshold*window_len
  4. aggregation: 90th percentile.
"""

import numpy as np

from utils import load_app_data, plt, print_results


def plot_windows(arr, win_idx):
    _, ax = plt.subplots(1, 1, dpi=200)
    ax.plot(arr)
    for start, end in win_idx:
        ax.axvline(start, alpha=0.7, color="red")
        ax.axvline(end, alpha=0.7, color="red")

    ax.set_xlabel("Sample index")
    ax.set_ylabel("Power [W]")
    plt.show()


def split_series(indices, margin):
    win_idx = []
    start = indices[0] - margin
    i_old = start
    for i in indices:
        if i - i_old > 2 * margin + 1:
            win_idx.append((start, i_old + margin))
            start = i - margin

        i_old = i

    win_idx.append((start, i_old + margin))
    return win_idx


def filter_windows(pow_data, windows, thres):
    filtered = []
    for i, win in enumerate(windows):
        slice = pow_data[win[0]:win[1]]
        if slice.sum() >= thres * len(slice):
            filtered.append(win)

    return filtered


def low_power_duration(arr, thr):
    """
    Some appliances, e.g. dishwasher, have low power modes in their cycle.
    We want the margin to capture this low power modes even if they are below the activation threshold.
    """
    low_power_idx = np.nonzero(np.logical_and(arr <= thr, arr > 1))[0]
    low_power_wins = split_series(low_power_idx, 0)
    #plot_windows(arr, low_power_wins)
    low_power_lens = [ e - s for s, e in low_power_wins ]
    margin = int(np.median(low_power_lens))
    margin = max((1, margin))
    return margin


def segment_filter_aggregate(arr, thr=10, margin=None):
    when_on_idx = np.nonzero(arr > thr)[0]
    if margin is None:
        margin = low_power_duration(arr, thr)

    win_idx = split_series(when_on_idx, margin)
    win_idx = filter_windows(arr, win_idx, thr)
    plot_windows(arr, win_idx)
    winlengths = [ e - s for s, e in win_idx ]
    return int(np.percentile(winlengths, 90))


def heuristic_window_size(sample_period, activation_thr, datasets):
    ret = { app: { "win_size": [] } for app in activation_thr }
    for app_name in activation_thr:
        print("\n{}".format(app_name))
        app_thr = activation_thr[app_name]
        datagen = load_app_data(sample_period, app_name, datasets)
        for appli_data in datagen:
            win_size = segment_filter_aggregate(appli_data, app_thr)
            ret[app_name]["win_size"].append(win_size * sample_period)

    return ret


if __name__ == "__main__":
    params = {
        "sample_period": 20, #s
        "activation_thr": {
            "kettle": 100,
            "dish washer": 30,
            "washing machine": 40,
        },
        "datasets": {
            "REFIT": {
                "path": "datasets/REFIT/refit.h5",
                "buildings": {
                    2: { "start_time": "2014-03-01", "end_time": "2014-03-15" },
                    3: { "start_time": "2014-09-01", "end_time": "2014-09-15" },
                    5: { "start_time": "2013-11-09", "end_time": "2013-11-24" },
                    6: { "start_time": "2014-11-09", "end_time": "2014-11-24" },
                    7: { "start_time": "2014-03-07", "end_time": "2014-03-22" },
                    9: { "start_time": "2014-05-01", "end_time": "2014-05-15" },
                    13: { "start_time": "2014-03-06", "end_time": "2014-03-21" },
                    19: { "start_time": "2014-06-01", "end_time": "2014-06-15" },
                }
            },
            "UK-DALE": {
                "path": "datasets/UK-DALE/ukdale2017.h5",
                "buildings": {
                    2: { "start_time": "2013-06-01", "end_time": "2013-06-15" },
                }
            },
        },
    }
    res = heuristic_window_size(**params)
    print(res)
    print_results(res)



# Observations
# - Best action for missing data: fill with zeros.
# - Expansion margin set to 4 min for dishwasher but is too long for kettle.
#     => Find the margin based on the low power (1 < pow <= thr) durations per
#        building. For that, median works best than mean, max because robust to
#        noise.
#
# - With automatic margin and 90th percentile aggregate:
#   res = {'kettle': {'win_size': [200, 360, 200, 180, 240, 260, 220, 180, 240]}, 'dish washer': {'win_size': [5920, 2900, 5300, 5800, 4220, 7820, 3440, 2120, 1700]}, 'washing machine': {'win_size': [1520, 3060, 1300, 1420, 5400, 4620, 1680, 2240, 2580]}}
#   Windows sizes in minutes:
#                    mean  max
#   kettle              4    6
#   dish washer        73  131
#   washing machine    45   90
#
# - Run duration: 5s.
#
