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
Experiment 1: Compute the window sizes using each of the methods for all problems.
"""
import numpy as np
import pandas as pd

from data_lorenz import generate_1d_lorenz
from data_nilm import load_nilm_mapping
from data_paf import load_paf_rr
from event_detection import event_window_size
from false_nearest_neighbours import false_nearest_neighbours
from smoothness_mapping import smoothness_drop


def print_results(res_dict, sample_s):
    results = pd.DataFrame(res_dict, index=["sequence length (pts)"]).T
    results["window size (s)"] = results["sequence length (pts)"] * sample_s
    print(results)
    print()


def window_sizes_lorenz(n_pts=1923, noise_pct=10, sample_s=0.13):
    noisy_lorenz = generate_1d_lorenz(n_pts, noise_pct=noise_pct, tmstep=sample_s)
    predict = np.roll(noisy_lorenz, -1)
    predict[-1] = noisy_lorenz[-1]
    results = {}
    results['FNN'] = false_nearest_neighbours(predict)
    results['SMO'] = smoothness_drop(noisy_lorenz, predict)
    results['GLR'] = event_window_size(predict, 'GLR', { 'step_min': 1, })
    thr_kwargs = { 'sample_s': sample_s, 'on_power_threshold': 5, }
    results['THR'] = event_window_size(predict, 'THR', thr_kwargs)
    results['MAD'] = event_window_size(predict, 'MAD', { 'step_min': 15, })
    msg = "Lorenz chaotic data"
    print(msg)
    print('-' * len(msg))
    print_results(results, sample_s)


def window_sizes_nilm(sample_s=7, non_zero_thr=10):
    all_data = pd.concat(load_nilm_mapping(sample_s=sample_s).values(), axis=0)
    results = {}
    kelly2015_params = {
        'kettle': { 'on_power_threshold': 2000, 'min_on_duration': 12, 'min_off_duration': 0, },
        'dish washer': { 'on_power_threshold': 10, 'min_on_duration': 1800, 'min_off_duration': 1800, },
        'washing machine': { 'on_power_threshold': 180, 'min_on_duration': 1800, 'min_off_duration': 160, },
        # washing machine threshold is lower than in Kelly2015 because UK-DALE-2
        # is not detected.
    }
    for app_name in all_data.columns[1:]:
        results[app_name] = {}
        app_data = all_data[app_name]
        mains_data = all_data['mains']
        non_zero_data = app_data[app_data > non_zero_thr].values
        non_zero_mains = mains_data[app_data > non_zero_thr].values
        results[app_name]['FNN'] = false_nearest_neighbours(non_zero_data, Rtol=60)
        results[app_name]['SMO'] = smoothness_drop(non_zero_data, non_zero_mains, patience=3)
        results[app_name]['GLR'] = event_window_size(app_data.values, 'GLR')
        thr_kwargs = kelly2015_params[app_name]
        results[app_name]['THR'] = event_window_size(app_data, 'THR', thr_kwargs)
        results[app_name]['MAD'] = event_window_size(app_data.values, 'MAD')

    msg = "NILM data"
    print(msg)
    print('-' * len(msg))
    for app_name, res in results.items():
        print(app_name)
        print_results(res, sample_s)


def window_sizes_paf():
    all_data = load_paf_rr(['p']).values()
    all_data = pd.concat(all_data, axis=0, ignore_index=True)
    paf_only = all_data[all_data['label'].isin(['before PAF', 'PAF'])]
    results = {}
    hrv_data = all_data['RR_ms'].values
    hrv_short = paf_only['RR_ms'].values
    results['FNN'] = false_nearest_neighbours(hrv_short, stop_at=8)
    predict = paf_only['label'].map({'before PAF': 0, 'PAF': 1 }).values
    results['SMO'] = smoothness_drop(hrv_short, predict, patience=3)
    results['GLR'] = event_window_size(hrv_data, 'GLR')
    thr_kwargs = { 'sample_s': 1, 'on_power_threshold': 900, 'min_on_duration': 10, }
    results['THR'] = event_window_size(hrv_data, 'THR', thr_kwargs)
    results['MAD'] = event_window_size(hrv_data, 'MAD')
    msg = "Atrial fibrillation data"
    print(msg)
    print('-' * len(msg))
    results = pd.DataFrame(results, index=["sequence length (pts)"]).T
    print(results)
    print()


if __name__ == "__main__":
    window_sizes_lorenz()
    window_sizes_nilm()
    window_sizes_paf()

