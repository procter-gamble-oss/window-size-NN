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

from collections import OrderedDict
from nilmtk.disaggregate import Disaggregator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Conv1D, GRU, Bidirectional, Dropout
from tensorflow.keras.models import Sequential


class WindowGRU(Disaggregator):
    """
    WindowGRU regressor after Krystalakos2018.
    See: https://doi.org/10.1145/3200947.3201011
    """
    MODEL_NAME = "WindowGRU"

    def __init__(self, params):
        """
        The appliance_params parameter is a dictionary of the maximum power
        consumption for each appliance, for instance:
        appliance_params["kettle"] = {
            "max": 3000,
        }
        """
        self.models = OrderedDict()
        self.file_prefix = "{}-temp-weights".format(self.MODEL_NAME.lower())
        self.chunk_wise_training = params.get("chunk_wise_training", False)
        self.sequence_length = params.get("sequence_length", 99)
        self.n_epochs = params.get("n_epochs", 10)
        self.batch_size = params.get("batch_size", 512)
        self.mains_max = params.get("mains_max", 10000)
        self.appliance_params = params.get("appliance_params", {})
        self.save_model_path = params.get("save-model-path", None)
        self.load_model_path = params.get("pretrained-model-path", None)

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
                # Sometimes, the Checkpoint is not created.
                # Better ignore it than loose 10hrs computation.
                try:
                    model.load_weights(filepath)
                except OSError as err:
                    print(err)

    def call_preprocessing(self, mains_lst, submeters_lst):
        # Transform aggregate data.
        new_mains = []
        for mains_df in mains_lst:
            mains = mains_df.values.flatten()
            mains = mains / self.mains_max
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
                app_max = self.appliance_params[app_name]["max"]
            else:
                raise IndexError("{}: Parameters for {} were not found!".format(
                    self.MODEL_NAME, app_name))

            new_app_readings = []
            for app_df in app_df_list:
                app_mat = app_df.values.reshape((-1, 1))
                app_mat = app_mat / app_max
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
                app_max = self.appliance_params[app_name]["max"]
                prediction = prediction * app_max
                prediction = prediction.flatten()
                prediction = np.where(prediction > 0, prediction, 0)
                disaggregation_dict[app_name] = pd.Series(prediction, dtype="float32")

            results = pd.DataFrame(disaggregation_dict, dtype="float32")
            test_predictions.append(results)

        return test_predictions

    def return_network(self):
        """
        This implementation is slow because non-compliant with the CudNN requirements.
        See: https://keras.io/api/layers/recurrent_layers/gru/
        """
        model = Sequential()
        model.add(Conv1D(16, 4, activation="relu",
                input_shape=(self.sequence_length, 1), padding="same", strides=1))
        model.add(Bidirectional(
            GRU(64, activation="relu", return_sequences=True),
            merge_mode="concat"))
        model.add(Dropout(0.5))
        model.add(Bidirectional(
            GRU(128, activation="relu", return_sequences=False),
            merge_mode="concat"))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation="linear"))
        model.compile(loss="mse", optimizer="adam")
        model.summary()
        return model

    def set_appliance_params(self, train_appliances):
        for app_name, df_list in train_appliances:
            app_data = pd.concat(df_list, axis=0)
            app_max = app_data.max()[0]
            self.appliance_params[app_name] = { "max": app_max, }

