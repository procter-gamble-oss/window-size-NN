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

import nilmtk
import numpy as np

from time_delay import find_min_time_delay
from utils import win2seqlen, ts2windows, chunk_with_activation, l1norm, l2norm, plot_time_delay, nearest_neighbor, print_results

nilmtk.Appliance.allow_synonyms = False


def load_mapping_data(sample_period, app_name, datasets):
    load_kwargs = {
        "physical_quantity": "power",
        "ac_type": "active",
        "sample_period": sample_period,
    }
    for ds_name, ds_dict in datasets.items():
        ds = nilmtk.DataSet(ds_dict["path"])
        for b_id, tframe in ds_dict["buildings"].items():
            print("Loading {}, building {}".format(ds_name, b_id))
            ds.set_window(tframe["start_time"], tframe["end_time"])
            elecmeter = ds.buildings[b_id].elec
            maingen = elecmeter.mains().load(**load_kwargs)
            appgen = elecmeter[app_name].load(**load_kwargs)
            mains_data = next(maingen)
            app_data = next(appgen)
            idx = mains_data.index.intersection(app_data.index)
            mains_data = mains_data.loc[idx].fillna(0)
            app_data = app_data.loc[idx].fillna(0)
            yield mains_data.values, app_data.values

        ds.store.close()


def smoothness(win_dist, neighbr_idx, mappedw):
    map_dist = l1norm(mappedw, mappedw[neighbr_idx.flatten()])
    win_dist = np.where(win_dist == 0, 1e-9, win_dist)
    return np.sum(map_dist / win_dist) / len(win_dist)


def smoothness_drop(arr, mapped_arr, sample_period, time_delay, smooth_tol=10):
    argmin_win = time_delay
    for i in range(len(arr) // time_delay):
        win_sz = i * time_delay if i else sample_period
        seqlen = win2seqlen(win_sz, sample_period)
        windows = ts2windows(arr, seqlen, time_delay // sample_period)
        mappedw = ts2windows(mapped_arr, seqlen, time_delay // sample_period)
        win_dist, neighbr_idx = nearest_neighbor(windows)
        smooth = smoothness(win_dist, neighbr_idx, mappedw)
        if i:
            smooth_inc = (smooth - smooth_old) / smooth_old
            print("{} min {} s: smoothness {:.1f} ({:.1f}%)".format(
                    win_sz // 60, win_sz % 60, smooth, smooth_inc * 100))
            if smooth < smooth_old:
                argmin_win = win_sz

            if smooth < smooth_tol:
                break

        smooth_old = smooth

    return argmin_win


def smoothness_win_size(sample_period, activation_thr, datasets, time_delay):
    ret = { app: { "time_delay": [], "win_size": [] } for app in activation_thr }
    for app_name in activation_thr:
        print("\n{}".format(app_name))
        app_thr = activation_thr[app_name]
        datagen = load_mapping_data(sample_period, app_name, datasets)
        for mains_data, appli_data in datagen:
            app_chnk = chunk_with_activation(appli_data, len(appli_data) // 4, app_thr)
            if time_delay:
                app_td = time_delay
            else:
                app_td = find_min_time_delay(app_chnk, sample_period, app_thr)

            print("Time delay for {}: {} min {} s.".format(
                    app_name, app_td // 60, app_td % 60))
            ret[app_name]["time_delay"].append(app_td)
            step = app_td // sample_period
            #plot_time_delay(app_chnk, step)
            # For NILM, input = aggregate, mapping data = appliance data.
            win_size = smoothness_drop(mains_data, appli_data, sample_period, app_td)
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
        },
    }
    res = smoothness_win_size(**params)
    print(res)
    print_results(res)



# Observations
# - On all building and all appliances, smoothness falls brutally to 0 at some
#   point. Hence, instead of detecting a relative drop in the smoothness, I
#   implemented the iteration until the smoothness is below 10.
#
# - When using N windows resampled at time delay (original method):
#   res = {'kettle': {'time_delay': [80, 40, 20, 40, 60, 40, 20, 20, 20], 'win_size': [800, 120, 80, 320, 120, 120, 40, 80, 40]}, 'dish washer': {'time_delay': [80, 20, 20, 40, 20, 40, 60, 20, 20], 'win_size': [880, 80, 120, 120, 160, 160, 360, 120, 80]}, 'washing machine': {'time_delay': [60, 40, 20, 40, 60, 20, 60, 20, 80], 'win_size': [360, 80, 80, 280, 600, 240, 1200, 80, 160]}}
#   In minutes:
#                    mean  max
#   kettle              4   14
#   dish washer         4   15
#   washing machine     6   20
#
# - Run duration: 58min 46s
#
# - When using N windows at original sampling:
#   res = {'kettle': {'time_delay': [80, 40, 60, 80, 20, 40, 20, 20, 60], 'win_size': [240, 40, 120, 240, 40, 80, 40, 80, 480]}, 'dish washer': {'time_delay': [20, 20, 60, 40, 60, 40, 40, 40, 40], 'win_size': [160, 80, 120, 80, 180, 120, 120, 120, 440]}, 'washing machine': {'time_delay': [60, 20, 20, 40, 20, 20, 20, 20, 20], 'win_size': [120, 80, 80, 280, 200, 240, 880, 80, 440]}}
#   In minutes:
#                    mean  max
#   kettle              3    8
#   dish washer         3    8
#   washing machine     5   15
#
# - When using N * sample period / time delay windows resampled at time delay:
#   res = {'kettle': {'time_delay': [80, 40, 20, 40, 60, 40, 20, 20, 20], 'win_size': [240, 80, 80, 120, 60, 80, 40, 80, 40]}, 'dish washer': {'time_delay': [80, 20, 20, 40, 20, 40, 60, 20, 20], 'win_size': [160, 80, 80, 80, 160, 80, 120, 120, 40]}, 'washing machine': {'time_delay': [60, 40, 20, 40, 60, 20, 60, 20, 80], 'win_size': [60, 40, 80, 120, 120, 240, 300, 80, 80]}}
#   In minutes:
#                    mean  max
#   kettle              2    4
#   dish washer         2    3
#   washing machine     3    5
#
# => Original method gives values that are relevant with the state of the art
#    empirical ones.
# => Aggregating building data with the maximum of window sizes is the most
#    relevant as we want the embedding to contain the meaninful information for
#    all instances of the appliance.

