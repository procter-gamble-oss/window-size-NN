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
Experiment 2: Train and test the model performance for selected sequence lengths.
"""
import multiprocessing
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import mean_absolute_error, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Reshape
from tensorflow.keras.layers import GlobalMaxPool1D, GRU, Bidirectional
from tensorflow.keras.models import Model

from data_lorenz import generate_1d_lorenz
from data_nilm import load_nilm_mapping, APPLIANCES
from data_paf import load_paf_rr
from utils import ts2windows


def network_mlp(seqlen, units=128, n_out=1):
    start = Input(shape=seqlen)
    hidden = Dense(units, activation='sigmoid')(start)
    final = Dense(n_out, activation='linear')(hidden)
    model = Model(inputs=start, outputs=final, name='MLP')
    adam = tf.keras.optimizers.Adam(learning_rate=0.002)
    model.compile(loss='mse', optimizer=adam)
    model.summary()
    return model


def network_seq2point(seqlen):
    """ Adapted from https://github.com/MingjunZhong/seq2point-nilm
    """
    input_layer = Input(shape=seqlen)
    reshape_layer = Reshape((1, seqlen, 1))(input_layer)
    conv_layer_1 = Conv1D(filters=30, kernel_size=10, strides=1, padding='same',
                    activation='relu')(reshape_layer)
    conv_layer_2 = Conv1D(filters=30, kernel_size=8, strides=1, padding='same',
                    activation='relu')(conv_layer_1)
    conv_layer_3 = Conv1D(filters=40, kernel_size=6, strides=1, padding='same',
                    activation='relu')(conv_layer_2)
    conv_layer_4 = Conv1D(filters=50, kernel_size=5, strides=1, padding='same',
                    activation='relu')(conv_layer_3)
    conv_layer_5 = Conv1D(filters=50, kernel_size=5, strides=1, padding='same',
                    activation='relu')(conv_layer_4)
    flatten_layer = Flatten()(conv_layer_5)
    label_layer = Dense(1024, activation='relu')(flatten_layer)
    output_layer = Dense(1, activation='linear')(label_layer)
    model = Model(inputs=input_layer, outputs=output_layer, name='Seq2Point')
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    return model


def network_bidirGRU(seqlen):
    """ See Gilon2020
    """
    input_layer = Input(shape=seqlen)
    reshape_layer = Reshape((seqlen, 1))(input_layer)
    conv_layer_1 = Conv1D(filters=100, kernel_size=3, strides=1,
                    activation='relu')(reshape_layer)
    conv_layer_2 = Conv1D(filters=100, kernel_size=3, strides=1,
                    activation='relu')(conv_layer_1)
    pooling = GlobalMaxPool1D()(conv_layer_2)
    reshape_layer_2 = Reshape((100, 1))(pooling)
    bidir_layer = Bidirectional(layer=GRU(100), merge_mode='concat')(reshape_layer_2)
    output_layer = Dense(1, activation='sigmoid')(bidir_layer)
    model = Model(inputs=input_layer, outputs=output_layer, name='bidirGRU')
    adam = tf.keras.optimizers.Adam(lr=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['acc'])
    model.summary()
    return model


def train(in_data, out_data, in_val, out_val, model, batchsz, epochs, patience):
    # See: Prechelt1998
    stop = EarlyStopping(
            monitor="val_loss",
            patience=patience,
            verbose=0,
            mode="min",
            restore_best_weights=True,
    )
    history = model.fit(
            in_data, out_data,
            batch_size=batchsz,
            epochs=epochs,
            validation_data=(in_val, out_val),
            verbose=2,
            callbacks=[ stop, ],
    )
    return history.history


def test(in_test, out_test, model, metrics=[]):
    ret = {}
    out_pred = model.predict(in_test)
    for func in metrics:
        ret[func.__name__] = func(out_test, out_pred)

    return ret, out_pred


def _crossval_worker(seqlen, model_func, train_kwargs, test_args, queue):
    tf.keras.backend.clear_session()
    model = model_func(seqlen)
    train_kwargs['model'] = model
    history = train(**train_kwargs)
    queue.put(history)
    test_res = {}
    for i, (in_test, out_test, metrics) in enumerate(test_args):
        perf, out_pred = test(in_test, out_test, model, metrics)
        update_results(perf, test_res)
        test_data = pd.DataFrame(in_test,
                columns=[ 'X_test{}'.format(i) for i in range(in_test.shape[1]) ])
        test_data_out = pd.DataFrame({
                'y_true': out_test.flatten(),
                'y_pred': out_pred.flatten(),
        })
        test_data = pd.concat([test_data, test_data_out], axis=1)
        test_data.to_csv('{}-{}-{}.csv'.format(model_func.__name__, seqlen, i),
                        index=False)

    test_res = pd.DataFrame(test_res).mean().to_dict()
    queue.put(test_res)


def cross_validate(folds, in_data_dict, out_data_dict, seqlen, model_func,
train_kwargs, metric_dict):
    assert 'patience' in train_kwargs
    ids = list(in_data_dict.keys())
    n = seqlen // 2
    in_data_dict = { k: ts2windows(v.values, seqlen, padding=None)
                     for k, v in in_data_dict.items() }
    out_data_dict = { k: np.reshape(v.values, (-1, 1))[n:-n]
                     for k, v in out_data_dict.items() }
    chksz = len(ids) // folds
    ids_sets = [ ids[k*chksz:(k+1)*chksz] for k in range(folds) ]
    ret = {}
    for k in range(folds):
        id_val = ids_sets[k][0]
        testset = ids_sets[k][1:]
        trainset = [ i for lst in ids_sets[:k] + ids_sets[k+1:] for i in lst ]
        train_kwargs['in_data'] = np.vstack([ in_data_dict[i] for i in trainset ])
        train_kwargs['out_data'] = np.vstack([ out_data_dict[i] for i in trainset ])
        train_kwargs['in_val'] = in_data_dict[id_val]
        train_kwargs['out_val'] = out_data_dict[id_val]
        test_args = []
        for i in testset:
            in_test = in_data_dict[i]
            out_test = out_data_dict[i]
            test_args.append((in_test, out_test, metric_dict[i]))

        # Workaround to prevent memory leak in TF to cause a OOM error.
        # See: https://github.com/tensorflow/tensorflow/issues/36465#issuecomment-582749350
        queue = multiprocessing.SimpleQueue()
        worker = multiprocessing.Process(
                target=_crossval_worker,
                args=(seqlen, model_func, train_kwargs, test_args, queue),
                daemon=True,
        )
        worker.start()
        history = queue.get()
        test_res = queue.get()
        worker.join()
        update_results(history, ret, -train_kwargs['patience'] - 1)
        update_results(test_res, ret)

    ret = pd.DataFrame(ret)

    return ret.mean().to_dict()


def normalize_per_series(ts_dict):
    scalers = {}
    transformed = {}
    for key, data in ts_dict.items():
        zscale = StandardScaler().fit(data)
        if isinstance(data, pd.DataFrame):
            transformed[key] = pd.DataFrame(zscale.transform(data),
                                index=data.index, columns=data.columns)
        else:
            transformed[key] = zscale.transform(data)
        scalers[key] = zscale

    return transformed, scalers


def binary_voting_window(series, seqlen, thr=0.5):
    assert set(series.values) == {0, 1,}
    windows = ts2windows(series.values, seqlen, padding=None)
    n = seqlen // 2
    votes = np.sum(windows, axis=1)
    votes = np.where(votes > seqlen * thr, 1, 0)
    votes = pd.Series(votes, index=series.index[n:-n])
    return votes


def relative_pred_err(x_true, y_true, y_pred):
    """ In Frank2000, the formula is only the sqrt of the division.
        However, if the prediction is perfect, the sqrt would tend toward 1, not
        toward 0.
        Thus, we compute the Manhattan distance to 1 to have a true minimization
        objective.
    """
    num = np.sum(np.power(x_true - y_pred, 2))
    denom = np.sum(np.power(x_true - y_true, 2))
    return abs(1 - np.sqrt(num / denom))


def nilm_binary_confusion(y_true, y_pred, thr):
    pos_gt = y_true > thr
    pos_pred = y_pred > thr
    neg_gt = y_true <= thr
    neg_pred = y_pred <= thr
    positive = np.logical_and(pos_gt, pos_pred)
    negative = np.logical_and(neg_gt, neg_pred)
    truepos = np.count_nonzero(positive)
    trueneg = np.count_nonzero(negative)
    falseneg = np.count_nonzero(pos_gt) - truepos
    falsepos = np.count_nonzero(pos_pred) - truepos
    return truepos, trueneg, falsepos, falseneg


def nilm_f1score(y_true, y_pred, thr=10, mean=0, std=1,):
    """ Binarize the time series before computing the F1-score.
        If the data is normalized, use the mean and std args to normalize the
        threshold.
    """
    thr = (thr - mean) / std
    truepos, _, falsepos, falseneg = nilm_binary_confusion(y_true, y_pred, thr)
    denom = 2 * truepos + falseneg + falsepos
    f1 = 0.
    if denom > 0:
        f1 = 2 * truepos / denom

    return f1


def update_results(history, res_dict, idx=None):
    for col, values in history.items():
        if col not in res_dict:
            res_dict[col] = []

        if idx is None:
            res_dict[col].append(values)
        else:
            res_dict[col].append(values[idx])


def neural_lorenz(seqlens=[ 5, 8, 15, 82, ], n_pts=1923, noise_pct=10,
sample_s=0.13, shift=-1, batchsz=1200, epochs=6000, patience=6, repeat=5):
    noisy_lorenz = generate_1d_lorenz(n_pts, noise_pct=noise_pct, tmstep=sample_s)
    predict = np.roll(noisy_lorenz, shift)
    train_kwargs = { 'batchsz': batchsz, 'epochs': epochs, 'patience': patience, }
    train_kwargs['out_data'] = np.reshape(predict[:batchsz], (-1, 1))
    x_test = np.reshape(noisy_lorenz[batchsz:shift], (-1, 1))
    y_test = np.reshape(predict[batchsz:shift], (-1, 1))
    train_kwargs['out_val'] = y_test
    winperf = {}
    for s in seqlens:
        train_kwargs['in_data'] = ts2windows(noisy_lorenz[:batchsz], s,
                                             padding='before')
        train_kwargs['in_val'] = ts2windows(noisy_lorenz[batchsz:shift], s,
                                            padding='before')
        res = { 'rel_err': [], 'mae': [], }
        for _ in range(repeat):
            mlp = network_mlp(s)
            train_kwargs['model'] = mlp
            history = train(**train_kwargs)
            update_results(history, res, -patience - 1)
            out_pred = mlp.predict(train_kwargs['in_val'])
            res['rel_err'].append(relative_pred_err(x_test, y_test, out_pred))
            res['mae'].append(mean_absolute_error(y_test, out_pred))

        res = pd.DataFrame(res)
        res = res.mean()
        winperf[s] = res

    winperf = pd.DataFrame(winperf)
    msg = "Lorenz chaotic data"
    print(msg)
    print('-' * len(msg))
    print(winperf)


def neural_nilm(seqlens={ 'kettle': [ 19, 24, 99, 111, 228, 255, 1162, ],
'dish washer': [ 99, 178, 528, 879, 1023, 1320, 1972, ],
'washing machine': [ 99, 106, 400, 511, 874, 983, 1688, ], },
sample_s=7, batchsz=512, epochs=6000, patience=6, folds=3):
    home_dict = load_nilm_mapping(sample_s=sample_s)
    normalized, scalers = normalize_per_series(home_dict)
    results = {}
    for i, app_name in enumerate(APPLIANCES):
        metrics = {}
        for k, sclr in scalers.items():
            avg = sclr.mean_[i+1]
            std = np.sqrt(sclr.var_)[i+1]
            def f1score(y_true, y_pred):
                return nilm_f1score(y_true, y_pred, 10, avg, std)

            metrics[k] = [ mean_absolute_error, f1score, ]

        app_in = { k: v['mains'] for k, v in normalized.items() }
        app_out = { k: v[app_name] for k, v in normalized.items() }
        app_res = {}
        for s in seqlens.get(app_name, []):
            train_kwargs = { 'batchsz': batchsz, 'epochs': epochs,
                            'patience': patience, }
            metric_res = cross_validate(folds, app_in, app_out, s,
                         network_seq2point, train_kwargs, metrics)
            app_res[s] = metric_res

        results[app_name] = pd.DataFrame(app_res)

    msg = "NILM data"
    print(msg)
    print('-' * len(msg))
    with pd.option_context('display.max_columns', None):
        for app_name, app_res in results.items():
            print(app_name)
            print(app_res)
            print()


def neural_paf(seqlens=[63, 85, 92, 100, 197, 300], batchsz=512, epochs=6000, patience=6, folds=3):
    patient_records = load_paf_rr(['p',])
    labelmap = {'NSR': 0, 'before PAF': 0, 'PAF': 1,}
    patients_out = { k: v['label'].map(labelmap)
                    for k, v in patient_records.items() }
    patient_rr = { k: v[['RR_ms']] for k, v in patient_records.items() }
    normalized, scalers = normalize_per_series(patient_rr)
    results = {}
    metrics = {}
    for k in scalers.keys():
        def f1score(y_true, y_pred):
            return nilm_f1score(y_true, y_pred, 0.5)

        metrics[k] = [ roc_auc_score, f1score, ]

    for s in seqlens:
        train_kwargs = { 'batchsz': batchsz, 'epochs': epochs,
                        'patience': patience, }
        metric_res = cross_validate(folds, normalized, patients_out, s,
                     network_bidirGRU, train_kwargs, metrics)
        results[s] = metric_res

    msg = "Atrial fibrillation data"
    print(msg)
    print('-' * len(msg))
    results = pd.DataFrame(results)
    with pd.option_context('display.max_columns', None):
        print(results)
        print()


if __name__ == '__main__':
    neural_lorenz()
    neural_nilm()
    neural_paf()

