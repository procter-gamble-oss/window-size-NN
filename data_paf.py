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

""" Loader for paroxysmal atrial fibrillation (PAF) data.
"""

import matplotlib.pyplot as plt
import numpy as np
import os.path
import pandas as pd
import wfdb
import wfdb.processing

plt.style.use("tableau-colorblind10")


def ecg2hrv(sig, freq, adc_gain=None, adc_zero=None):
    """ Transforms the electrocardiogram into the heart beat variability.
    """
    qrs_idx = wfdb.processing.qrs.gqrs_detect(
                sig=sig, fs=freq, adc_gain=adc_gain, adc_zero=adc_zero)
    # Heart rates in beat/minute (bpm)
    hrs = wfdb.processing.hr.compute_hr(sig_len=sig.shape[0], qrs_inds=qrs_idx, fs=freq)
    return hrs


def load_paf_rr(prefixes=[ 'p', 'n', ], pos_prefix='p', cont_suffix='c',
id_range=range(1, 51, 2), folder="../../AF/datasets/Physionet-PAF/"):
    # Only use training set because testing set is not labeled.
    # NSR   NSR     before PAF  PAF
    # p15   p15c    p16         p16c
    #
    # Relevant positive records: 1, 3, 5, 7, 15, 27, 29, 31, 37, 43, 47, 49
    all_data = {}
    for i in id_range:
        for p in prefixes:
            chunks = []
            patient = '{}{:02d}'.format(p, i)
            print("Loading records for {}".format(patient))
            for j in range(i, i + 2):
                for c in [ '', cont_suffix, ]:
                    id = '{}{:02d}{}'.format(p, j, c)
                    full_id = os.path.join(folder, id)
                    record = wfdb.rdann(full_id, 'qrs')
                    qrs_idx = record.sample
                    if p == pos_prefix and j % 2 == 0:
                        l = 'PAF' if c == cont_suffix else 'before PAF'
                    else:
                        l = 'NSR'

                    rr = wfdb.processing.hr.calc_rr(qrs_idx, record.fs,
                                                    rr_units='seconds')
                    rr = np.round(rr * 1000)
                    data = {
                        'RR_ms': rr,
                        'label': [ l, ] * len(rr),
                    }
                    data = pd.DataFrame(data).dropna()
                    chunks.append(data)

            chunks = pd.concat(chunks, axis=0, ignore_index=True)
            all_data[patient] = chunks

    return all_data


def plot_first_ltaf():
    folder = "../../AF/datasets/Physionet-LTAF/"
    first = os.path.join(folder, '00')
    record = wfdb.rdrecord(first)
    print(record.__dict__)
    annots = wfdb.rdann(first, 'atr', summarize_labels=True)
    summary = annots.contained_labels
    print(summary)
    qrs_idx = wfdb.rdann(first, 'qrs').sample
    sig_names = [ n + str(i) for i, n in enumerate(record.sig_name) ]
    signal = pd.DataFrame(record.p_signal, columns=sig_names)
    hrv = wfdb.processing.hr.compute_hr(
            sig_len=signal.shape[0], qrs_inds=qrs_idx, fs=record.fs)
    hrv = pd.DataFrame(hrv, columns=['hrv'])
    labels = pd.DataFrame(annots.symbol, index=annots.sample, columns=['label'])
    labels = labels.reindex(signal.index).ffill()
    first_data = pd.concat((signal, hrv, labels), axis=1).dropna()
    label_map = dict(zip(summary['symbol'], summary['label_store']))
    first_data['label'] = first_data['label'].map(label_map)
    first_data.plot(subplots=True, sharex=True)
    plt.show()


def load_ltaf_rr(id_range=set(range(76)).union(range(100, 123)), max_pts=11000, folder="../../AF/datasets/Physionet-LTAF/"):
    # 75 patients.
    all_data = {}
    for i in id_range:
        patient = '{:02d}'.format(i)
        print("Loading records for {}".format(patient), end='')
        full_id = os.path.join(folder, patient)
        try:
            record = wfdb.rdann(full_id, 'qrs')
            qrs_idx = record.sample
            rr = wfdb.processing.hr.calc_rr(qrs_idx, record.fs, rr_units='seconds')
            rr = np.round(rr * 1000)
            rr = pd.DataFrame(rr, index=qrs_idx[1:], columns=['RR_ms'])
            annots = wfdb.rdann(full_id, 'atr', summarize_labels=True)
            labels = pd.DataFrame(annots.symbol, index=annots.sample,
                                  columns=['label'])
            data = pd.concat([rr, labels,], axis=1)
            data['label'] = data['label'].ffill()
            all_data[patient] = data.dropna().iloc[:max_pts]
            print("...OK")
        except FileNotFoundError:
            print("...Not found!")

    return all_data


def plot_rr(nsr_data, af_data, title, ymin=100, ymax=4000):
    _, ax = plt.subplots()
    nsr_data.plot.scatter(x='index', y='RR_ms', color='C0', ax=ax, label='NSR')
    af_data.plot.scatter(x='index', y='RR_ms', color='C1', ax=ax, label='PAF')
    ax.set_ylim(100, 4000)
    ax.set_title(title)
    plt.show()


if __name__ == '__main__':
    #plot_first_ltaf()
    total_rows = 0
    for title, data in load_paf_rr(['p',]).items():
        total_rows += data.shape[0]
        plot_rr(
                data[data['label'] != 'PAF'].reset_index(),
                data[data['label'] == 'PAF'].reset_index(),
                title
        )

    print("Total number of points: {}".format(total_rows))
    total_rows = 0
    for title, data in load_ltaf_rr().items():
        total_rows += data.shape[0]
        nsr_data = data[data['label'] == 'N'].reset_index()
        af_data = data[data['label'] != 'N'].reset_index()
        print(data.shape, nsr_data.shape, af_data.shape)
        plot_rr(nsr_data, af_data, title)

    print("Total number of points: {}".format(total_rows))
