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
Compute the optimal window size using the false nearest neighbours method.
See Kennel1992, Frank2000.
"""

import numpy as np

from time_delay import find_min_time_delay
from utils import win2seqlen, ts2windows, chunk_with_activation, plot_time_delay, l2norm, nearest_neighbor, load_app_data, print_results


def estimate_attractor_size(arr):
    """
    Useful for a low number of points but not in our case (tested and no difference).
    """
    mean_vec = np.ones_like(arr) * arr.mean()
    return l2norm(arr, mean_vec) / np.sqrt(len(arr))


def false_nearest_neighbours(arr, sample_period, time_delay, false_nn_tol=10, stop_ratio=0.003):
    """
    Using Ball Tree nearest neighbours algorithm as it scales well and is
    deterministic (no random changes in the indices).
    """
    # Limit the window to 6 hours.
    max_win = 6 * 3600 // time_delay
    argmin_win = max_win * time_delay
    for i in range(1, max_win + 1):
        win_sz = i * time_delay if i else sample_period
        seqlen = win2seqlen(win_sz, sample_period)
        windows = ts2windows(arr, seqlen, time_delay // sample_period)
        dist, neighbr_idx = nearest_neighbor(windows)
        if i > 1:
            idx = neighbr_idx.flatten()
            embed_err = np.abs(windows[:,-1] - windows[idx,-1]).reshape((-1, 1))
            criterion = embed_err > false_nn_tol * dist_old
            false_nn = np.count_nonzero(criterion)
            false_nn_ratio = false_nn / len(arr)
            print("\t{} min {} s: {} false NN ({:.2f}%)".format(
                win_sz // 60, win_sz % 60, false_nn, false_nn_ratio * 100))
            if false_nn_ratio < stop_ratio:
                argmin_win = win_sz
                break

        dist_old = dist
        idx_old = neighbr_idx

    return argmin_win


def false_nn_win_size(sample_period, activation_thr, datasets, time_delay):
    ret = { app: { "time_delay": [], "win_size": [] } for app in activation_thr }
    for app_name in activation_thr:
        print("\n{}".format(app_name))
        app_thr = activation_thr[app_name]
        datagen = load_app_data(sample_period, app_name, datasets)
        for appli_data in datagen:
            app_chnk = chunk_with_activation(appli_data, len(appli_data) // 4, app_thr)
            if time_delay:
                app_td = time_delay
            else:
                app_td = find_min_time_delay(app_chnk, sample_period, app_thr)

            print("\tTime delay for {}: {} min {} s.".format(app_name, app_td // 60, app_td % 60))
            #step = app_td // sample_period
            #plot_time_delay(app_chnk, step)
            ret[app_name]["time_delay"].append(app_td)
            if app_name == "kettle":
                stop_tol = 0.005
            else:
                stop_tol = 0.001

            win_size = false_nearest_neighbours(appli_data, sample_period, app_td, stop_ratio=stop_tol)
            ret[app_name]["win_size"].append(win_size)

    return ret


if __name__ == "__main__":
    params = {
        "sample_period": 20, #s
        "time_delay": 0, # automatic time delay with 0
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
        }
    }
    res = false_nn_win_size(**params)
    print(res)
    print_results(res)



# Observations:
# - Original criterion is |x[-1] - x_r[-1]| / dist_old > Rtol. However, the
#   distance dist_old is sometimes 0, leading to an undefined criterion.
#   Thus, I implement instead the criterion: |x[-1] - x_r[-1]| > Rtol * dist_old
#
# - When using N windows resampled at time_delay with Rtol = 10 and stop at
#   0.05 %:
#   res = {'kettle': {'time_delay': [80, 40, 20, 40, 60, 40, 20, 20, 20], 'win_size': [160, 80, 60, 80, 120, 80, 40, 40, 60]}, 'dish washer': {'time_delay': [80, 20, 20, 40, 20, 40, 60, 20, 20], 'win_size': [320, 60, 60, 80, 60, 120, 180, 60, 60]}, 'washing machine': {'time_delay': [60, 40, 20, 40, 60, 20, 60, 20, 80], 'win_size': [180, 80, 60, 80, 180, 60, 180, 60, 160]}}
#   Windows sizes in minutes
#                    mean  max
#   kettle              2    3
#   dish washer         2    6
#   washing machine     2    3
#
# - When using N windows resampled at time_delay with Rtol = 10 and stop at
#   0.03 %:
#   res = {'kettle': {'time_delay': [80, 40, 20, 40, 60, 40, 20, 20, 20], 'win_size': [6400, 80, 60, 480, 180, 120, 60, 60, 60]}, 'dish washer': {'time_delay': [80, 20, 20, 40, 20, 40, 60, 20, 20], 'win_size': [400, 60, 60, 80, 60, 160, 180, 60, 60]}, 'washing machine': {'time_delay': [60, 40, 20, 40, 60, 20, 60, 20, 80], 'win_size': [180, 120, 60, 80, 180, 60, 780, 60, 160]}}
#   Windows sizes in minutes:
#                    mean  max
#   kettle             14  107
#   dish washer         3    7
#   washing machine     4   13
#
# - False NN struggles with kettle (short events near from each other).
#   When using N windows resampled at time_delay with Rtol = 10 and stop at
#   0.05% for kettle and 0.01% for the rest:
#   res = {'kettle': {'time_delay': [80, 40, 20, 40, 60, 40, 20, 20, 20], 'win_size': [160, 80, 60, 80, 120, 80, 40, 40, 60]}, 'dish washer': {'time_delay': [80, 20, 20, 40, 20, 40, 60, 20, 20], 'win_size': [1840, 60, 60, 120, 60, 240, 180, 60, 60]}, 'washing machine': {'time_delay': [60, 40, 20, 40, 60, 20, 60, 20, 80], 'win_size': [300, 120, 60, 240, 1740, 60, 1020, 60, 2000]}}
#   Windows sizes in minutes:
#                    mean  max
#   kettle              2    3
#   dish washer         5   31
#   washing machine    11   34
#
# - Run duration: 52min 10s.
#
# - Problem: the percentage of activation in the data varies between 0.5 and 8
#   %. Hence, 92 to 99.5% of the data equal 0 and thus is true nearest neighbor.
#   Validity of a fixed stop criterion?
#

