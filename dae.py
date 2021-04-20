# Copyright 2021 The Procter & Gamble Company
#
# This file has been modified by The Procter & Gamble Company under compliance
# with the original license stated below:
#
# Copyright 2019 NILMTK-contrib Developers
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

import json
import numpy as np
import os
import pandas as pd

from collections import OrderedDict
from nilmtk.disaggregate import Disaggregator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv1D, Dense, Reshape, Flatten
from tensorflow.keras.models import Sequential


class DAE(Disaggregator):
    """
    Denoising autoencoder after Kelly2015.
    See: https://dx.doi.org/10.1145/2821650.2821672
    """
    MODEL_NAME = "DAE"

    def __init__(self, params):
        """
        The appliance_params parameter is a dictionary of the mean and standard
        deviation of the power consumption for each appliance, for instance:
        appliance_params["kettle"] = {
            "mean": 700,
            "std": 1000,
        }
        """
        self.models = OrderedDict()
        self.file_prefix = "{}-temp-weights".format(self.MODEL_NAME.lower())
        self.chunk_wise_training = params.get("chunk_wise_training", False)
        self.sequence_length = params.get("sequence_length", 99)
        self.n_epochs = params.get("n_epochs", 10)
        self.batch_size = params.get("batch_size", 512)
        self.mains_mean = params.get("mains_mean", 1000)
        self.mains_std = params.get("mains_std", 600)
        self.appliance_params = params.get("appliance_params", {})
        self.save_model_path = params.get("save-model-path", None)
        self.load_model_path = params.get("pretrained-model-path", None)
        if self.load_model_path:
            self.load_model()

    def partial_fit(self, train_main, train_appliances, do_preprocessing=True, current_epoch=0, **load_kwargs):
        # If no appliance wise parameters are specified, then they are computed from the data.
        if len(self.appliance_params) == 0:
            self.set_appliance_params(train_appliances)

        if do_preprocessing:
            train_main, train_appliances = self.call_preprocessing(train_main, train_appliances)

        train_main = np.concatenate(train_main, axis=0)
        for appliance_name, power in train_appliances:
            power = np.concatenate(power, axis=0)
            # Check if the appliance was already trained. If not then create a new model for it.
            if appliance_name not in self.models:
                print("{}: First model training for {}".format(self.MODEL_NAME, appliance_name))
                self.models[appliance_name] = self.return_network()
            # Retrain the particular appliance.
            else:
                print("{}: Retraining model for {}".format(self.MODEL_NAME, appliance_name))

            model = self.models[appliance_name]
            # Sometimes chunks can be empty after dropping NANS.
            if len(train_main) > 10:
                filepath = self.file_prefix + "-{}-epoch{}.h5".format(
                        "_".join(appliance_name.split()),
                        current_epoch,
                )
                checkpoint = ModelCheckpoint(
                        filepath,
                        monitor="val_loss",
                        verbose=0,
                        save_best_only=True,
                        mode="min"
                )
                model.fit(
                        train_main,
                        power,
                        batch_size=self.batch_size,
                        validation_split=.15,
                        epochs=self.n_epochs,
                        callbacks=[ checkpoint, ],
                )
                model.load_weights(filepath)

    def load_model(self):
        print ("{}: Loading the model using the pretrained weights.".format(self.MODEL_NAME))
        model_folder = self.load_model_path
        with open(os.path.join(model_folder, "model.json"), "r") as f:
            params_to_load = json.load(f)

        self.sequence_length = int(params_to_load['sequence_length'])
        self.mains_mean = params_to_load['mains_mean']
        self.mains_std = params_to_load['mains_std']
        self.appliance_params = params_to_load['appliance_params']
        for appliance_name in self.appliance_params:
            self.models[appliance_name] = self.return_network()
            self.models[appliance_name].load_weights(
                    os.path.join(model_folder, "_".join(appliance_name.split()) + ".h5")
            )

    def save_model(self):
        os.makedirs(self.save_model_path)
        params_to_save = {}
        params_to_save['appliance_params'] = self.appliance_params
        params_to_save['sequence_length'] = self.sequence_length
        params_to_save['mains_mean'] = self.mains_mean
        params_to_save['mains_std'] = self.mains_std
        for appliance_name in self.models:
            print ("{}: Saving model for {}.".format(self.MODEL_NAME, appliance_name))
            self.models[appliance_name].save_weights(
                    os.path.join(self.save_model_path, "_".join(appliance_name.split()) + ".h5")
            )

        with open(os.path.join(self.save_model_path, 'model.json'), 'w') as file:
            json.dump(params_to_save, file)

    def disaggregate_chunk(self, test_main_lst, do_preprocessing=True):
        if do_preprocessing:
            test_main_lst, _ = self.call_preprocessing(test_main_lst, [])

        test_predictions = []
        for mains_df in test_main_lst:
            disaggregation_dict = {}
            for app_name, model in self.models.items():
                prediction = model.predict(mains_df, batch_size=self.batch_size)
                disaggregation_dict[app_name] = pd.Series(self.denormalize_output(prediction, app_name))

            results = pd.DataFrame(disaggregation_dict, dtype="float32")
            test_predictions.append(results)

        return test_predictions

    def return_network(self):
        """
        Implementation notes:
            - In the paper, the variable dense layer size is (sequence_length - 3) * 8.
              However this changes the size of the network output and so,
              ground truth windows must be smaller than the sequence_length.
        """
        model = Sequential()
        model.add(Conv1D(8, 4, activation="linear",
                input_shape=(self.sequence_length, 1), padding="same", strides=1))
        model.add(Flatten())
        dense_layer_sz = (self.sequence_length - 3) * 8
        model.add(Dense(dense_layer_sz, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(dense_layer_sz, activation='relu'))
        model.add(Reshape((dense_layer_sz // 8, 8)))
        model.add(Conv1D(1, 4, activation="linear", padding="same", strides=1))
        model.compile(loss='mse', optimizer='adam')
        model.summary()
        return model

    def call_preprocessing(self, mains_lst, submeters_lst):
        new_mains = [ self.normalize(df, "mains") for df in mains_lst]
        appliance_list = []
        for app_name, app_df_list in submeters_lst:
            new_app_dfs = [ self.normalize(df, app_name) for df in app_df_list ]
            appliance_list.append((app_name, new_app_dfs))

        return new_mains, appliance_list

    def normalize(self, df, app_name):
        if app_name == "mains":
            avg = self.mains_mean
            std = self.mains_std
            n = self.sequence_length
        elif app_name in self.appliance_params:
            avg = self.appliance_params[app_name]["mean"]
            std = self.appliance_params[app_name]["std"]
            n = self.sequence_length - 3 # See: return_network
        else:
            raise IndexError("{}: Parameters for {} were not found!".format(
                self.MODEL_NAME, app_name))

        arr = df.values.flatten()
        arr = (arr - avg) / std
        pad = self.sequence_length // 2
        arr = np.pad(arr, pad, mode="constant", constant_values=( 0, 0, ))
        windows = np.array([
            arr[i:i + n]
            for i in range(len(arr) - self.sequence_length + 1)
        ])
        windows = windows.reshape(( -1, n, 1 ))
        return windows

    def denormalize_output(self, predicted, app_name):
        app_mean = self.appliance_params[app_name]["mean"]
        app_std = self.appliance_params[app_name]["std"]
        predicted = app_mean + predicted * app_std
        pad = self.sequence_length // 2
        padded_len = len(predicted) + self.sequence_length - 1
        sum_arr = np.zeros((padded_len, ), dtype="float32")
        cnt = predicted.shape[1]
        for i, window in enumerate(predicted):
            sum_arr[i:i + cnt] += window.flatten()

        sum_arr = np.where(sum_arr > 0, sum_arr, 0)
        sum_arr /= cnt
        return sum_arr[pad:-pad]

    def set_appliance_params(self, train_appliances):
        for app_name, df_list in train_appliances:
            app_data = pd.concat(df_list, axis=0)
            app_mean = app_data.mean()[0]
            app_std = app_data.std()[0]
            if app_std < 1:
                app_std = 100

            self.appliance_params[app_name] = { "mean": app_mean, "std": app_std }

