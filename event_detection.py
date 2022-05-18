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
Compute the window size using event detection.
Implemented event detection algorithms:
    - modified generalized likelihood ratio (Berges2011)
    - thresholding around 0, nilmtk.electric.get_activation (Kelly2015)
    - mean absolute deviation (Rehman2020)
"""

import numpy as np
import pandas as pd

from nilmtk.electric import get_activations
from scipy.stats import norm
from utils import ts2windows


def generalized_likelihood_ratio(arr, wlb_len=40, wla_len=40, we_len=160, votes_min=10, step_min=30):
    votes = np.zeros_like(arr)
    event_win = ts2windows(arr, we_len)
    before_win = ts2windows(arr, wlb_len, padding='before')
    after_win = ts2windows(arr, wla_len)
    before_mean = before_win.mean(axis=1)
    before_std = before_win.std(axis=1)
    before_std[before_std == 0] = 1
    after_mean = after_win.mean(axis=1)
    after_std = after_win.std(axis=1)
    after_std[after_std == 0] = 1
    for i in range(len(arr) - we_len):
        b_mean = before_mean[i:i+we_len]
        a_mean = after_mean[i:i+we_len]
        before_prob = norm.pdf(event_win[i], b_mean , before_std[i:i+we_len])
        before_prob[before_prob == 0] = 1e-8
        after_prob = norm.pdf(event_win[i], a_mean, after_std[i:i+we_len])
        likelihood_ratio = np.log(after_prob / before_prob)
        likelihood_ratio[np.abs(a_mean - b_mean) < step_min] = 0
        test_statistics = [ likelihood_ratio[j:].sum() for j in range(we_len) ]
        vote_idx = i + np.argmax(test_statistics)
        votes[vote_idx] += 1

    return np.nonzero(votes > votes_min)[0]


def mean_absolute_deviation(arr, w_len=5, step_min=250):
    windows = ts2windows(arr, w_len)
    win_avg = np.mean(windows, axis=1).reshape((-1, 1))
    deviations = np.sum(np.abs(windows - win_avg), axis=1) / w_len
    return np.nonzero(deviations > step_min)[0]


def event_window_size(arr, method, kwargs={}):
    method = method.lower()
    assert method in [ 'glr', 'thr', 'mad', ]
    if method in [ 'glr', 'mad', ]:
        kwargs['arr'] = arr
        if method == 'glr':
            event_idx = generalized_likelihood_ratio(**kwargs)
        else:
            event_idx = mean_absolute_deviation(**kwargs)

        lengths = (event_idx - np.roll(event_idx, 1))[1:]
    elif method == 'thr':
        if isinstance(arr, pd.DataFrame):
            kwargs['chunk'] = arr
        elif isinstance(arr, pd.Series):
            kwargs['chunk'] = pd.DataFrame(arr, index=arr.index)
        elif 'sample_s' in kwargs:
            sample_s = kwargs.pop('sample_s')
            idx = pd.date_range('now', periods=len(arr),
                                freq='{}S'.format(sample_s))
            kwargs['chunk'] = pd.DataFrame(arr, index=idx, dtype='float32')
        else:
            raise NotImplementedError

        activations = get_activations(**kwargs)
        lengths = [ len(a) for a in activations ]

    return int(np.round(np.mean(lengths)))

