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

import pandas as pd
import numpy as np
import os

from collections import OrderedDict
from nilmtk.disaggregate import Disaggregator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv1D, Dense, Flatten
from tensorflow.keras.models import Sequential


class ReducedSeq2Point(Disaggregator):
    """
    Reduced Seq2Point architecture after Barber2020.
    See: https://dx.doi.org/10.1145/3427771.3427845
    """
    MODEL_NAME = "Seq2Point_reduced"

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
        self.appliance_params = params.get("appliance_params", {})
        self.mains_mean = params.get("mains_mean", 1800)
        self.mains_std = params.get("mains_std", 600)
        if self.sequence_length%2==0:
            raise ValueError("{}: sequence_length should be odd!".format(self.MODEL_NAME))

    def partial_fit(self, train_main, train_appliances, do_preprocessing=True, current_epoch=0, **load_kwargs):
        if len(self.appliance_params) == 0:
            self.set_appliance_params(train_appliances)

        if do_preprocessing:
            train_main, train_appliances = self.call_preprocessing(train_main, train_appliances)

        train_main = np.concatenate(train_main, axis=0)
        for appliance_name, power in train_appliances:
            power = np.concatenate(power, axis=0)
            if appliance_name not in self.models:
                print("{}: First model training for {}".format(self.MODEL_NAME, appliance_name))
                self.models[appliance_name] = self.return_network()
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

    def call_preprocessing(self, mains_lst, submeters_lst):
        # Transform aggregate data.
        new_mains = []
        for mains_df in mains_lst:
            mains = mains_df.values.flatten()
            mains = (mains - self.mains_mean) / self.mains_std
            pad = self.sequence_length // 2
            mains = np.pad(mains, pad, mode="constant", constant_values=0)
            mains = np.array([
                mains[i:i + self.sequence_length]
                for i in range(len(mains) - self.sequence_length + 1)
            ])
            mains = mains.reshape(( -1, self.sequence_length, 1 ))
            new_mains.append(mains)

        # Transform appliance-level data.
        appliance_list = []
        for app_name, app_df_list in submeters_lst:
            if app_name in self.appliance_params:
                app_mean = self.appliance_params[app_name]["mean"]
                app_std = self.appliance_params[app_name]["std"]
            else:
                raise IndexError("{}: Parameters for {} were not found!".format(
                    self.MODEL_NAME, app_name))

            new_app_readings = []
            for app_df in app_df_list:
                app_mat = app_df.values.reshape((-1, 1))
                app_mat = (app_mat - app_mean) / app_std
                new_app_readings.append(app_mat)

            appliance_list.append(( app_name, new_app_readings ))

        return new_mains, appliance_list

    def disaggregate_chunk(self, test_main_lst, do_preprocessing=True):
        if do_preprocessing:
            test_main_lst, _ = self.call_preprocessing(test_main_lst, [])

        test_predictions = []
        for mains_df in test_main_lst:
            disaggregation_dict = {}
            for app_name, model in self.models.items():
                prediction = model.predict(mains_df, batch_size=self.batch_size)
                app_mean = self.appliance_params[app_name]["mean"]
                app_std = self.appliance_params[app_name]["std"]
                prediction = app_mean + prediction * app_std
                prediction = prediction.flatten()
                prediction = np.where(prediction > 0, prediction, 0)
                disaggregation_dict[app_name] = pd.Series(prediction, dtype="float32")

            results = pd.DataFrame(disaggregation_dict, dtype="float32")
            test_predictions.append(results)

        return test_predictions

    def return_network(self):
        model = Sequential()
        model.add(Conv1D(20, 8, activation="relu", strides=1,
                         input_shape=(self.sequence_length, 1)))
        model.add(Conv1D(20, 6, activation="relu", strides=1))
        model.add(Conv1D(30, 5, activation="relu", strides=1))
        model.add(Conv1D(40, 4, activation="relu", strides=1))
        model.add(Conv1D(40, 4, activation="relu", strides=1))
        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(Dense(1, activation="linear"))
        model.compile(loss="mse", optimizer="adam", metrics=[ "mse", "mae" ])
        model.summary()
        return model

    def set_appliance_params(self, train_appliances):
        for app_name, df_list in train_appliances:
            app_data = pd.concat(df_list, axis=0)
            app_mean = app_data.mean()[0]
            app_std = app_data.std()[0]
            if app_std < 1:
                app_std = 100

            self.appliance_params[app_name] = { "mean": app_mean, "std": app_std }

