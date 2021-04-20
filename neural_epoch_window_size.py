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
Num epochs vs window size vs. accuracy experiment
"""

import nilmtk
import numpy as np
import pandas as pd
import pickle
import sklearn.metrics
import tensorflow as tf
import time

from natsort import natsorted
from nilmtk.losses import mae
from dae import DAE
from seq2point_reduced import ReducedSeq2Point
from WindowGRU import WindowGRU
from utils import plt, win2seqlen

nilmtk.Appliance.allow_synonyms = False


def load_chunks(mains_gen, appli_gens):
    last_appli_date = 0
    for i, mains_chunk in enumerate(mains_gen):
        print("Chunk {}".format(i), end="\r")
        mains_df = mains_chunk.dropna()
        idx = mains_df.index
        last_mains_date = idx[-1]
        # Mains and appliance records are not always synchronized.
        if i == 0 or last_mains_date > last_appli_date:
            appli_df_list = []
            try:
                for appli_gen in appli_gens:
                    appli_chunk = next(appli_gen)
                    appli_df = appli_chunk.dropna()
                    appli_df_list.append(appli_df)

            except StopIteration:
                break

        for appli_df in appli_df_list:
            appli_idx = appli_df.index
            idx = idx.intersection(appli_idx)

        last_appli_date = appli_idx[-1]
        if len(idx) > 10:
            mains_df = mains_df.loc[idx]
            appli_dfs = [ adf.loc[idx] for adf in appli_df_list ]
            yield mains_df, appli_dfs


def generate_elec_data(chunk_size, sample_period, train, test, power, appliances, train_mode=True, **kwargs):
    load_kwargs = {
        "chunksize": chunk_size,
        "physical_quantity": "power",
        "sample_period": sample_period,
    }
    if train_mode:
        datasets = train["datasets"]
    else:
        datasets = test["datasets"]

    mains_dfs = []
    appli_df_map = { app: [] for app in appliances }
    loaded_len = 0
    for ds_name, ds_dict in datasets.items():
        ds = nilmtk.DataSet(ds_dict["path"])
        for b_id, tf in ds_dict["buildings"].items():
            print("Loading {}, building {}".format(ds_name, b_id))
            ds.set_window(start=tf["start_time"], end=tf["end_time"])
            load_kwargs["ac_type"] = power["mains"]
            mains_gen = ds.buildings[b_id].elec.mains().load(**load_kwargs)
            load_kwargs["ac_type"] = power["appliance"]
            appli_gens = [
                    ds.buildings[b_id].elec[appli_name].load(**load_kwargs)
                    for appli_name in appliances
            ]
            for mains_chnk, appli_chnks in load_chunks(mains_gen, appli_gens):
                appli_df_pairs = list(zip(appliances, appli_chnks))
                if train_mode:
                    mains_dfs.append(mains_chnk)
                    for appli_name, appli_df in appli_df_pairs:
                        appli_df_map[appli_name].append(appli_df)

                    loaded_len += mains_chnk.shape[0]
                    if loaded_len > chunk_size:
                        yield mains_dfs, list(appli_df_map.items())
                        mains_dfs = []
                        appli_df_map = { app: [] for app in appliances }
                        loaded_len = 0

                else:
                    appli_df_pairs = [ ( an, [ adf ] ) for an, adf in appli_df_pairs ]
                    yield [ mains_chnk ], appli_df_pairs

        ds.store.close()
        del ds

    if len(mains_dfs):
        yield mains_dfs, list(appli_df_map.items())


def generate_inferences(params, alg):
    datagen = generate_elec_data(**params, train_mode=False)
    for mains_dfs, appli_dfs in datagen:
        infered = alg.disaggregate_chunk(mains_dfs)[0]
        mains_df = mains_dfs[0]
        mains_len = mains_df.shape[0]
        infer_len = infered.shape[0]
        if infer_len > mains_len:
            infered = infered[:mains_len]
        elif infer_len < mains_len:
            pad = mains_len - infer_len
            infered = pd.concat([ infered, pd.Series([ 0 ] * pad ) ])

        infered.index = mains_df.index
        yield mains_df, appli_dfs, infered


def load_chkpt(path, app_names):
    try:
        with open(path, "rb") as chkfile:
            metrics = pickle.load(chkfile)

    except FileNotFoundError:
        metrics = {
            "f1score": { a: { "epoch": 1 } for a in app_names },
            "mae": { a: {} for a in app_names },
        }


    return metrics


def save_chkpt(path, metrics):
    with open(path, "wb") as chkfile:
        pickle.dump(metrics, chkfile)


def f1score(app_gt, app_pred, app_name):
    thresholds = {
        "kettle": 100,
        "dish washer": 30,
        "washing machine": 40,
    }
    thr = thresholds.get(app_name, 10)
    gt_temp = np.array(app_gt)
    gt_temp = np.where(gt_temp < thr, 0, 1)
    pred_temp = np.array(app_pred)
    pred_temp = np.where(pred_temp < thr, 0, 1)
    return sklearn.metrics.f1_score(gt_temp, pred_temp)


def train_test_by_epoch(params, methods, n_epochs=30):
    """
    For testing, only uses the first metric name in the list.
    """
    chkpt_path = "neural_epoch_winsz.chkpt"
    metrics = load_chkpt(chkpt_path, methods.keys())
    for appli_name, f1_dict in metrics["f1score"].items():
        params["appliances"] = [ appli_name ]
        for alg_name, alg_dict in methods[appli_name].items():
            if alg_name not in f1_dict:
                f1_dict[alg_name] = {}

            for winlen, alg in alg_dict.items():
                if winlen not in f1_dict[alg_name]:
                    f1_dict[alg_name][winlen] = []

                    mae_metrics = []
                    for i in range(1, n_epochs + 1):
                        print("\nTraining {} ({}), epoch {}".format(alg_name, winlen, i))
                        for mains_dfs, appli_dfs in generate_elec_data(**params):
                            alg.partial_fit(mains_dfs, appli_dfs, current_epoch=i)

                        print("Testing {} ({}), epoch {}".format(alg_name, winlen, i))
                        f1_metrics = []
                        for mains_df, appli_dfs, infered in generate_inferences(params, alg):
                            _, appli_data = appli_dfs[0] # one appliance anyway
                            loss = f1score(appli_data[0], infered[[appli_name]], appli_name)
                            f1_metrics.append(loss)
                            if i == n_epochs:
                                loss = mae(appli_data[0], infered[[appli_name]])
                                mae_metrics.append(loss)

                        avg_metric = sum(f1_metrics) / len(f1_metrics)
                        f1_dict[alg_name][winlen].append(avg_metric)
                        print("Average F1-score: {}".format(avg_metric))
                        time.sleep(1)
                        if i > 4 and avg_metric < 0.1:
                            plot_inference(mains_df, appli_data[0], infered, appli_name)

                    #plot_inference(mains_df, appli_data[0], infered, appli_name)
                    mae_dict = metrics["mae"][appli_name]
                    if alg_name not in mae_dict:
                        mae_dict[alg_name] = {}

                    mae_dict[alg_name][winlen] = sum(mae_metrics) / len(mae_metrics)

                save_chkpt(chkpt_path, metrics)
                alg.clear_model_checkpoints()
                del alg

        f1_dict["epoch"] = n_epochs

    return metrics


def plot_inference(mains_df, gt_df, inf_df, appli_name):
    _, ax = plt.subplots(dpi=200)
    mains_df.plot(ax=ax, color="C0")
    gt_df.plot(ax=ax, color="C1")
    inf_df[[appli_name]].plot(ax=ax, color="C2")
    ax.set_ylabel("Power [W]")
    ax.set_title(appli_name)
    ax.legend([ "mains", appli_name, "infered" ])
    plt.tight_layout()
    plt.show()


def plot_metrics(m_dict, metric_name):
    """
    Plot data per appliance while expecting the metrics dictionary to look like:
    { "kettle": { "Seq2Point": { "EBH (6 min)": [], "10 min": [], } } }
    """
    m_dict = m_dict[metric_name]
    colors = { "LIT": "C0", "FNN": "C1", "SMO": "C2", "EBH": "C3", }
    suptitles = { "dish washer": "(a) dishwasher", "kettle": "(b) kettle",
                  "washing machine": "(c) washing machine" }
    for appli_name, algos in m_dict.items():
        n_epochs = algos.pop("epoch")
        fig, axes = plt.subplots(1, len(algos), sharex=True, sharey=False, dpi=200)
        for i, (alg_name, win_dict) in enumerate(algos.items()):
            try:
                m_ax = axes[i]
            except TypeError:
                m_ax = axes # case len(algos) == 1

            idx = list(range(1, n_epochs + 1))
            #win_dict = dict(natsorted(win_dict.items()))
            m_df = pd.DataFrame(win_dict, index=idx)
            for label in m_df.columns:
                for alg in colors:
                    if alg in label:
                        c = colors[alg]

                m_df[[label]].plot(ax=m_ax, color=c)

            m_ax.set_title(alg_name)
            m_ax.set_ylabel(metric_name)
            m_ax.set_ylim(0, 1)
            m_ax.legend(loc="lower right")
            m_ax.set_xlabel("epoch")

        fig.suptitle(suptitles[appli_name])
        fig.subplots_adjust(top=0.88, bottom=0.11, left=0.06, right=0.98, wspace=0.28)
        plt.show()


def model_params(winsz, sample_period, batchsz=512):
    # Keep n_epochs to 1 as the epochs will be managed by the calling function.
    ret = {
        "n_epochs": 1,
        "chunk_wise_training": True,
        "batch_size": batchsz,
        "sequence_length": win2seqlen(winsz, sample_period),
    }
    return ret


if __name__ == "__main__":
    params = {
        "power": { "mains": "active", "appliance": "active" },
        "sample_period": 20,
        "chunk_size": 2**21,
        "appliances": [ "kettle", "dish washer", "washing machine", ],
        "train": {
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
                    },
                },
            },
        },
        "test": {
        },
    }
    params["test"]["datasets"] = params["train"]["datasets"]

    methods = {
        "kettle": {
            "DAE": {
                "FNN (3 min)": DAE(model_params(3, params["sample_period"])),
                "EBH (6 min)": DAE(model_params(6, params["sample_period"])),
                "LIT (13 min)": DAE(model_params(13, params["sample_period"])),
                "SMO (14 min)": DAE(model_params(14, params["sample_period"])),
            },
            "reduced Seq2Point": {
                "EBH (8 min)": ReducedSeq2Point(model_params(8, params["sample_period"])),
                "LIT (13 min)": ReducedSeq2Point(model_params(13, params["sample_period"])),
                "SMO (14 min)": ReducedSeq2Point(model_params(14, params["sample_period"])),
            },
            "WindowGRU": {
                "FNN (3 min)":  WindowGRU(model_params(3, params["sample_period"])),
                "EBH (6 min)":  WindowGRU(model_params(6, params["sample_period"])),
                "LIT (13 min)": WindowGRU(model_params(13, params["sample_period"])),
                "SMO (14 min)": WindowGRU(model_params(14, params["sample_period"])),
            },
        },

        "dish washer": {
            "DAE": {
                "SMO (15 min)": DAE(model_params(15, params["sample_period"])),
                "FNN (31 min)": DAE(model_params(31, params["sample_period"])),
                "LIT (40 min)": DAE(model_params(40, params["sample_period"])),
                "EBH (131 min)": DAE(model_params(131, params["sample_period"])),
            },
            "reduced Seq2Point": {
                "SMO (15 min)":  ReducedSeq2Point(model_params(15, params["sample_period"])),
                "FNN (31 min)":  ReducedSeq2Point(model_params(31, params["sample_period"])),
                "LIT (40 min)":  ReducedSeq2Point(model_params(40, params["sample_period"], 256)),
                "EBH (131 min)": ReducedSeq2Point(model_params(131, params["sample_period"], 256)),
            },
            "WindowGRU": {
                "SMO (15 min)":  WindowGRU(model_params(15, params["sample_period"])),
                "FNN (31 min)":  WindowGRU(model_params(31, params["sample_period"])),
                "LIT (40 min)":  WindowGRU(model_params(40, params["sample_period"])),
                "EBH (131 min)": WindowGRU(model_params(131, params["sample_period"], 128)),
            },
        },

        "washing machine": {
            "DAE": {
                "SMO (20 min)": DAE(model_params(20, params["sample_period"])),
                "FNN (34 min)": DAE(model_params(34, params["sample_period"])),
                "EBH (90 min)": DAE(model_params(90, params["sample_period"])),
                "LIT (111 min)": DAE(model_params(111, params["sample_period"])),
            },
            "reduced Seq2Point": {
                "SMO (20 min)":  ReducedSeq2Point(model_params(20, params["sample_period"])),
                "FNN (34 min)":  ReducedSeq2Point(model_params(34, params["sample_period"])),
                "EBH (90 min)":  ReducedSeq2Point(model_params(90, params["sample_period"], 128)),
                "LIT (111 min)": ReducedSeq2Point(model_params(111, params["sample_period"], 128)),
            },
            "WindowGRU": {
                "SMO (20 min)":  WindowGRU(model_params(20, params["sample_period"])),
                "FNN (34 min)":  WindowGRU(model_params(34, params["sample_period"])),
                "EBH (90 min)":  WindowGRU(model_params(90, params["sample_period"], 256)),
                "LIT (111 min)":  WindowGRU(model_params(90, params["sample_period"], 128)),
            },
        }
    }

    metrics = train_test_by_epoch(params, methods)
    print(metrics)
    for app_name, alg_dict in metrics["mae"].items():
        print(app_name)
        mae_df = pd.DataFrame(alg_dict)
        print(mae_df)

    plot_metrics(metrics, "f1score")



# Observations
# - Windows of 3 and 6 minutes are too small for the Seq2Point convolutional
#   input layer. Testing 8 minutes instead.

