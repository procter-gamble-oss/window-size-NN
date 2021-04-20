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

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
plt.style.use("seaborn-paper")
import multiprocessing
import nilmtk
import numpy as np
import pandas as pd
import sklearn.neighbors

# From: https://github.com/nilmtk/nilmtk/issues/720
nilmtk.Appliance.allow_synonyms = False
N_WORKERS = max(( 1, multiprocessing.cpu_count() - 2 ))


def load_app_data(sample_period, app_name, datasets):
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
            datagen = ds.buildings[b_id].elec[app_name].load(**load_kwargs)
            app_data = next(datagen)
            app_data.fillna(0, inplace=True)
            yield app_data.values

        ds.store.close()


def win2seqlen(winlen_sec, sample_period):
    seqlen = winlen_sec // sample_period
    if seqlen % 2 == 0:
        seqlen += 1

    return seqlen


def ts2windows(arr, seqlen, step=1, stride=1):
    n = max((0, seqlen // 2))
    arrlen = len(arr) # length before padding
    arr = np.pad(arr.flatten(), n, mode="constant", constant_values=0)
    windows = [ arr[i-n:i+n+1:step] for i in range(n, arrlen + n, stride) ]
    windows = np.array(windows, dtype="float32")
    return windows


def chunk_with_activation(arr, chnksz, thr):
    n_chnk = len(arr) // chnksz
    for i in range(n_chnk):
        chnk = arr[i*chnksz:(i+1)*chnksz]
        if np.count_nonzero(chnk > thr) >= 0.02 * chnksz:
            return chnk

    return arr[:chnksz]


def plot_time_delay(app_chnk, step):
    for i in range(len(app_chnk) // step):
        plt.axvline(i * step, alpha=0.3)

    plt.plot(app_chnk)
    plt.show()


def l1norm(x1, x2):
    ret = np.abs(x2 - x1)
    if len(ret.shape) > 1:
        if ret.shape[1] > 1:
            ret = np.sum(ret, axis=1).reshape((-1, 1))
    return ret


def l2norm(x1, x2):
    return np.sqrt(np.sum((x2 - x1)**2))


def nearest_neighbor(mat):
    """
    Sklearn implementation of nearest neighbors returns some windows as nearest neighbors of themselves.
    This function ensures that all the neighbors are distinct from the input points.
    """
    n_pts = mat.shape[0]
    neighbors = sklearn.neighbors.BallTree(mat, leaf_size=n_pts // 100, metric="euclidean")
    dist_mat, idx_mat = neighbors.query(mat, k=2, dualtree=True)
    self_nn = np.arange(n_pts) == idx_mat[:,0]
    true_idx = np.where(self_nn, idx_mat[:,1], idx_mat[:,0]).reshape((-1, 1))
    true_dist = np.where(self_nn, dist_mat[:,1], dist_mat[:,0]).reshape((-1, 1))
    return true_dist, true_idx


def print_results(res):
    table = np.zeros((len(res), 2), dtype="int32")
    for i, (app_name, app_dict) in enumerate(res.items()):
        winszs = pd.DataFrame(app_dict)["win_size"]
        table[i,:] = [ winszs.mean(), winszs.max() ]

    table = np.ceil(table / 60)
    table = pd.DataFrame(table, index=res.keys(), columns=["mean", "max"], dtype="int32")
    print("\nWindows sizes in minutes:")
    print(table)

