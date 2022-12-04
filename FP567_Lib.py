'''
Can use below command in a linux terminal to delete files in certain folders with certain extensions.
If you run the below command in resources/Processed_Updates/by_year it will delete all files that
end with .npy in the folders that have the day_num.txt files.
find . -type f -iname *\*\*.npy -delete
'''

import os
import calendar
import shutil
import datetime
import matplotlib.pyplot
import json
import glob
import sys
import random
import pathlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from contextlib import redirect_stdout

PROJECT_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
PLOTS_DIR = "plots"
PATH_TO_RESOURCES = os.path.join(PROJECT_ROOT_DIR, "resources")
PATH_TO_MARKET_ITEM_DATA = os.path.join(PATH_TO_RESOURCES, "market_item_data")
PATH_TO_RAW_UPDATES = os.path.join(PATH_TO_RESOURCES, "Raw_Updates")
PATH_TO_SAME_UPDATES_IN_DIFF_FORMS = os.path.join(PATH_TO_RAW_UPDATES, "same_updates_in_diff_forms")
PATH_TO_RAW_BY_DAY_UPDATES = os.path.join(PATH_TO_SAME_UPDATES_IN_DIFF_FORMS, "by_day_updates")
PATH_TO_RAW_BY_YEAR_UPDATES = os.path.join(PATH_TO_SAME_UPDATES_IN_DIFF_FORMS, "by_year_updates")
PATH_TO_RAW_GAME_UPDATES = os.path.join(PATH_TO_SAME_UPDATES_IN_DIFF_FORMS, "game_updates")
PATH_TO_RAW_MISC_UPDATES = os.path.join(PATH_TO_RAW_UPDATES, "misc_updates")
PATH_TO_RAW_MISC_UPDATES_BEEN_PROCD = os.path.join(PATH_TO_RAW_MISC_UPDATES, "has_been_processed")
PATH_TO_RAW_NOISY_UPDATES = os.path.join(PATH_TO_RAW_UPDATES, "noisy_updates")
PATH_TO_NOISY_JUNK_UPDATES = os.path.join(PATH_TO_RAW_NOISY_UPDATES, "junk?")
PATH_TO_NOISY_DATE_UNKNOWN_UPDATES = os.path.join(PATH_TO_RAW_NOISY_UPDATES, "junk?")
PATH_TO_PROCESSED_UPDATES = os.path.join(PATH_TO_RESOURCES, "Processed_Updates")
PATH_TO_PROCESSED_UPDATES_BY_YEAR = os.path.join(PATH_TO_PROCESSED_UPDATES, "by_year")
PATH_TO_PLOTS = os.path.join(PROJECT_ROOT_DIR, "PLOTS_DIR")
PATH_TO_MODELS_DIRECTORY = os.path.join(PROJECT_ROOT_DIR, "models")

FORCASTED_DAY_LEN_STR = "forcasted_day_length" 
PATH_TO_FORCASTING_COMPONENTS = os.path.join(PATH_TO_RESOURCES, "forcasting")
PATH_TO_ASSEMBLED_FORCASTING_MATRIX = os.path.join(PATH_TO_FORCASTING_COMPONENTS, "forcasting_matrix.csv")
PATH_TO_ASSEMBLED_FORCASTING_MATRIX_SPECS = os.path.join(PATH_TO_FORCASTING_COMPONENTS, "forcasting_matrix_specs.json")

RUNESCAPE_YEARS = list(map(str, range(1998, 2023)))
MONTHS = list(map(str.lower, calendar.month_name[1:]))
WORD_MONTH_TO_NUMBER_MAP = {
    "january": "01", "february": "02", "march": "03", "april": "04", "may": "05", "june": "06",
    "july": "07", "august": "08", "september": "09", "october": "10", "november": "11", "december": "12"
} 
DAYS_OF_MONTH = list(map(str, range(1, 32)))
DAYS_OF_WEEK_ABRV = calendar.day_abbr[:]
NUMBER_OF_SECONDS_IN_A_UNIX_DAY = 86400
EMBEDDED_UPDATE_VEC_LEN = 28
BATCH_SIZE_STR = "batch_size"
PREDICT_SIZE_STR = "predict_size"
DAYS_TO_FORCAST_STR = "days_to_forcast"
TIME_WINDOW_SIZE_STR = "time_window_size"
NUMBER_ITEMS_PREDICTED_STR = "number_items_predicted"
WINDOW_SIZE_SCALE = "window_size_scale"

class MarketItem:
    def __init__(self, path_to_item_json, max_expected_item_feats=2):
        '''
        Given a path to a json market item, reads it in and makes a MarketItem out of it.
        '''
        item_data = json.load(open(path_to_item_json))
        item_keys = item_data.keys()
        item_vals = item_data.values()

        if len(item_keys) > 1 or len(item_vals) > 1:
            raise Exception(f"The json at {path_to_item_json} is malformed.")

        self.max_expected_item_feats = max_expected_item_feats
        self.path_to_item = path_to_item_json
        self.longest_feature_list_len = 0
        self.shortest_feature_list_len = sys.maxsize
        self.item_id = list(item_keys)[0]
        # this is a unix time -> [price, amount sold] dict
        self.time_to_info_dict = {}
        for unit_time_lis in list(item_vals)[0]:
            feats_list = unit_time_lis[1:]
            num_feats = len(feats_list)
            if num_feats > self.longest_feature_list_len:
                self.longest_feature_list_len = num_feats
            if num_feats < self.shortest_feature_list_len:
                self.shortest_feature_list_len = num_feats

            self.time_to_info_dict[unit_time_lis[0]] = feats_list

        self.number_of_unit_times = len(self.time_to_info_dict.keys())
        self.maybe_the_bond = self.longest_feature_list_len == 1 and self.shortest_feature_list_len == 1
    
    def has_complete_data(self) -> bool:
        return self.maybe_the_bond or self.shortest_feature_list_len == self.max_expected_item_feats and self.shortest_feature_list_len == self.longest_feature_list_len
    
    def get_info_at_unix_time(self, unix_time):
        if unix_time in self.time_to_info_dict:
            return self.time_to_info_dict[unix_time]
        else:
            return None

    def set_unix_time_to_info_dict(self, new_dict):
        self.time_to_info_dict = new_dict
    
        

class Market:
    def __init__(self):
        market_is = [MarketItem(path_to_market_item) for path_to_market_item in glob.glob(os.path.join(PATH_TO_MARKET_ITEM_DATA, "*.json"))]
        # market_items is a dict of item_id -> item's list of [time, price, amount_sold] instances.
        self.number_of_items = len(market_is)
        self.market_items = {}
        self.item_ids = []
        self.possible_bond_item_ids = []
        self.min_item_feature_count = sys.maxsize
        self.max_item_feature_count = 0
        for market_item in market_is:
            self.item_ids.append(market_item.item_id)
            self.market_items[market_item.item_id] = market_item
            if not market_item.maybe_the_bond:
                if market_item.longest_feature_list_len > self.max_item_feature_count:
                    self.max_item_feature_count = market_item.longest_feature_list_len
                if market_item.shortest_feature_list_len < self.min_item_feature_count:
                    self.min_item_feature_count = market_item.shortest_feature_list_len
            else:
                self.possible_bond_item_ids.append(market_item.item_id)

        self.forcasted_day_length = EMBEDDED_UPDATE_VEC_LEN + self.min_item_feature_count
        
        # number_of_infos_in_oldest_item is the number of [time, price, amount_sold] instances in the
        # item that has the most [time, price, amount_sold] instances, aka the longest runnning/oldest item.
        self.number_of_infos_in_oldest_item = np.array(list(map(lambda market_item: market_item.number_of_unit_times, market_is))).max()

        # markets_time_span is a list of the unix times from the item that has the most [time, price, amount_sold] instances.
        self.markets_time_span = sorted(list(self.get_items_that(
                lambda item: len(item.time_to_info_dict) == self.number_of_infos_in_oldest_item)[0].time_to_info_dict.keys()))

    
    def get_item_from_json_filename(self, file_name_json: str):
        return self.get_items_that(
            lambda item: item.path_to_item.endswith(file_name_json))[0]

    def get_items_that(self, predicate):
        '''
        Returns a list of MarketItems that match the given predicate.
        '''
        return [matching_item for matching_item in self.market_items.values() if predicate(matching_item)]
    
    
    def get_item_with_id(self, id: str) -> MarketItem:
        if id in self.market_items:
            return self.market_items[id]
        else:
            return None

    
    def balance_as_is(self, default_price, default_amount_sold):
        '''
        Balances the market based on it's current market_time_span
        '''
        self.balance(self.markets_time_span, default_price, default_amount_sold)

    
    def balance(self, list_of_unix_times_to_balance_around, default_price, default_amount_sold):
        '''
        Given a list of unix times, makes every item in the market have info for the list of unix times
        and the list of unix times only. If the item does not have info for a unix time in the list,
        then [default_price, default_amount_sold] is used.
        '''
        for item_id, item in self.market_items.items():
            balanced_info_dict = {}
            for unix_time in list_of_unix_times_to_balance_around:
                info_at_unix_time = item.get_info_at_unix_time(unix_time)
                balanced_info_dict[unix_time] = [default_price, default_amount_sold] if info_at_unix_time is None else info_at_unix_time

            self.market_items[item_id].set_unix_time_to_info_dict(balanced_info_dict)

        self.markets_time_span = sorted(list_of_unix_times_to_balance_around)

    
    def is_balanced(self) -> bool:
        '''
        Checks if the entire market's items have entirely the same span of unix times.
        '''
        market_items = list(self.market_items.values())
        item_0_times = sorted(list(market_items[0].time_to_info_dict.keys()))
        for item_i in market_items:
            item_i_times = sorted(list(item_i.time_to_info_dict.keys()))
            if len(item_0_times) != len(item_i_times) or item_0_times != item_i_times:
                return False
            else:
                item_0_times = item_i_times
        
        return True

    
    def build_features_matrix(self):
        market_rep = pd.DataFrame({})
        market_item_list = list(self.market_items.values())

        # self.markets_time_span is sorted during Market object construction, so we can
        # iterate over it now and know that we are increasing in time.
        for ms_unix_time in self.markets_time_span:
            items_feats_matrix = []
            for market_item in market_item_list:
                item_q_info = market_item.time_to_info_dict[ms_unix_time]
                # only build unit of time matrices using the feature number that is the
                # lowest feature number from all the market items.
                # This means if some market items have only price and no amount sold while some
                # market items have both price and amount sold, only the price feature will be used
                # for all market items, thus building a constant size unit of time matrix.

                # Also, price is the first item in the feats list, so reverse the list
                items_feats_matrix.append(item_q_info[:self.shortest_feature_list_len].reverse())
            
            embedded_day_matrix = np.tile(
                get_embedded_update_rep_on_date(convert_json_ms_unix_time_to_yyyy_m_d(ms_unix_time)),
                (self.number_of_items, 1))
            market_rep = pd.concat(
                [market_rep, pd.DataFrame(embedded_day_matrix), pd.DataFrame(items_feats_matrix)], axis=1)

        # make is so the items have their IDs as the DF row indices
        market_rep.set_index([item.item_id for item in market_item_list])
        self.market_with_updates_rep = market_rep

    def save_market_with_updates_rep_as_csv(
        self,
        path_to_save_it_to=PATH_TO_ASSEMBLED_FORCASTING_MATRIX,
        path_to_save_specs=PATH_TO_ASSEMBLED_FORCASTING_MATRIX_SPECS) -> None:

        self.market_with_updates_rep.to_csv(path_to_save_it_to)
        spec_dict = {
            FORCASTED_DAY_LEN_STR : self.forcasted_day_length
        }
        open(path_to_save_specs, "w").write(json.dumps(spec_dict, indent=4))


def make_numpy_mat_into_tf_sparse_tensor(mat, make_practice_sparse=False):
    '''
    Takes a dense numpy array and returns a sparse tensor.
    '''
    if make_practice_sparse:
        # if in here, then the given matrix is a practice matrix and want it to be
        # more sparse than it is to better resemble our training data later.
        total_rows = len(mat)
        for i in random.sample(range(total_rows), int(total_rows/2)):
            mat[i] = [0] * len(mat[i])

    non_zero_indices = np.nonzero(mat)
    row_indices = non_zero_indices[0]
    col_indices = non_zero_indices[1]

    idxs = list(zip(row_indices, col_indices))
    vals = mat[non_zero_indices]

    if len(idxs) != 0 and vals.shape[0] is not None and vals.shape[0] != 0:
        return tf.SparseTensor(
            indices=idxs,
            values=vals,
            dense_shape=mat.shape)
    else:
        return None

# Plotting time series function
# https://colab.research.google.com/github/ageron/handson-ml2/blob/master/15_processing_sequences_using_rnns_and_cnns.ipynb#scrollTo=YAJb9TLU7vkZ
def plot_time_series(
    time_series,
    number_of_steps,
    y=None,
    y_pred=None,
    x_label="$t$",
    y_label="$x(t)$") -> None:

    plt.plot(time_series, ".-")
    if y is not None:
        plt.plot(number_of_steps, y, "bo", label="Target")
    if y_pred is not None:
        plt.plot(number_of_steps, y_pred, "rx", markersize=10, label="Prediction")
    plt.grid(True)
    if x_label:
        plt.xlabel(x_label, fontsize=16)
    if y_label:
        plt.ylabel(y_label, fontsize=16, rotation=0)
    plt.hlines(0, 0, 100, linewidth=1)
    plt.axis([0, number_of_steps + 1, -1, 1])
    if y or y_pred:
        plt.legend(fontsize=14, loc="upper left")

class InputLayer(tf.keras.layers.InputLayer):
    def __init__(self, output_dim, **kwargs):
        super().__init__()
        self.output_dim = output_dim
        super(InputLayer, self).__init__(**kwargs)



    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)


def print_model_summary_to_file(model: tf.keras.Model, path_to_model_root: str) -> None:
    '''
    Prints the given model's summary to the given folder.
    '''

    with open(os.path.join(path_to_model_root, "Model_Summary.txt"), 'w') as f:
        with redirect_stdout(f):
            model.summary()


def save_training_params(
    batch_size,
    num_items_predicted,
    days_to_forcast,
    scale_of_window_size_to_days_to_predict,
    predict_size,
    time_window_size,
    path_to_save_at):
    '''
    Saves the given model hyper params to a json at the path given
    '''

    dict = {
        BATCH_SIZE_STR: str(batch_size),
        NUMBER_ITEMS_PREDICTED_STR: str(num_items_predicted),
        WINDOW_SIZE_SCALE: str(scale_of_window_size_to_days_to_predict),
        PREDICT_SIZE_STR: str(predict_size),
        DAYS_TO_FORCAST_STR: str(days_to_forcast),
        TIME_WINDOW_SIZE_STR: str(time_window_size),
    }

    # write the specs as a json
    open(os.path.join(path_to_save_at, "training_params.json"), "w").write(json.dumps(dict, indent=4))


def plot_loss(plt: matplotlib.pyplot, history, path_to_folder)-> None:

    # plot loss
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylim((0,4))
    plt.yticks(np.arange(0, 4.5, step=0.5))
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')

    # ensure that the path to save it to exists
    path = os.path.join(path_to_folder, PLOTS_DIR)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    save_fig(plt, os.path.join(path, "model_loss"))


def plot_accuracy(plt: matplotlib.pyplot, history, path_to_folder)-> None:

    # plot accuracy
    plt.clf()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.ylim((0,1))
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')

    # ensure that the path to save it to exists
    path = os.path.join(path_to_folder, PLOTS_DIR)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    save_fig(plt, os.path.join(path, "model_accuracy"))


def get_model(tensor_shape, batch_size, predict_size):

    '''
    The input to our model is two lists, each full of sparse tensors.
        X_training = 
        [
            sparse_tensor(items_feats from time 0 to time n),
            sparse_tensor(items_feats from time 0 to time n + p),
            ...,
            sparse_tensor(items_feats from time 0 to time n + i*p)
        ]
        Y_training = 
        [
            sparse_tensor(items_feats from time n to time n + p),
            sparse_tensor(items_feats from time (n + p) to time (n + p) + p),
            ...,
            sparse_tensor(items_feats from time (n + i*p) to time (n + i*p) + p)
        ]
    '''
    tf.debugging.disable_traceback_filtering()

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=tensor_shape, batch_size=batch_size, dtype=np.float16))
    model.add(tf.keras.layers.LSTM(units=tensor_shape[1], return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.30))
    step_down = int(tensor_shape[1]/2)
    model.add(tf.keras.layers.LSTM(step_down, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.20))
    step_down = int(step_down/2)
    model.add(tf.keras.layers.LSTM(step_down, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.20))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(predict_size)))

    model.compile(loss="mse", optimizer="adam", metrics=["last_time_step_mse"])

    return model


def get_forcasting_market_df(get_a_random_df=False) -> pd.DataFrame:
    '''
    Gets the market forcasting matrix as a DF either from disk (if it exists),
    or makes one, then returns it.
    '''
    if get_a_random_df:
        # This is used for practicing with time series.
        return pd.DataFrame(np.random.rand(3500, 30000)), 30

    if os.path.exists(PATH_TO_ASSEMBLED_FORCASTING_MATRIX):
        return pd.read_csv(PATH_TO_ASSEMBLED_FORCASTING_MATRIX), (
            json.load(open(PATH_TO_ASSEMBLED_FORCASTING_MATRIX_SPECS, "r")) if os.path.exists(PATH_TO_ASSEMBLED_FORCASTING_MATRIX_SPECS) else None)
    else:
        market = Market()
        market.balance_as_is(0, 0)
        if not market.is_balanced():
            raise Exception("Error. Market did not balance upon request.")
        # else
        market.build_features_matrix()
        return market.market_with_updates_rep, market.forcasted_day_length


def get_embedded_update_rep_on_date(date_object: datetime.date):
    embedded_update_path = os.path.join(
            PATH_TO_PROCESSED_UPDATES_BY_YEAR,
            date_object.year,
            date_object.month,
            f"{date_object.day}.embedded")
            
    return np.loadtxt(embedded_update_path, dtype=str) if os.path.exists(embedded_update_path) else np.zeros(EMBEDDED_UPDATE_VEC_LEN).astype(str)

def save_fig(plt: matplotlib.pyplot, fig_id: str, tight_layout=True, fig_extension="png", resolution=300) -> None:
    '''
    Given a plot object and save specs/info, saves the plt object's current fig
    to the globally defined plots folder path
    '''
    
    os.makedirs(PATH_TO_PLOTS, exist_ok=True)
    path = os.path.join(PATH_TO_PLOTS, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def move_file_to_if_fnx(
    path_to_folder_to_enumerate: str,
    path_to_other_folder: str,
    keep_fnx):
    '''
    Checks all the files in a the given folder and keeps them if they pass
    the given function check. Else, they are moved to the other given folder.
    '''

    for file in os.listdir(path_to_folder_to_enumerate):
        path_to_file = os.path.join(path_to_folder_to_enumerate, file)
        if not keep_fnx(path_to_file):
            shutil.move(path_to_file, path_to_other_folder)


def check_if_init_is_game_updates(file_name: str):
    '''
    There are files in a folder that are useless that do not have their
    first non-blank line as "# Game Updates"
    '''
    with open(file_name, "r") as f:
        blank_lines = True
        while blank_lines:
            line = f.readline()
            toks = line.split()
            num_toks = len(toks)
            if num_toks > 0:
                blank_lines=False
    
    return num_toks == 3 and toks[0]=="#" and toks[1]=="Game" and toks[2]=="updates"


def get_updates_from_certain_misc_updates_pattern(files_with_pattern) -> None:
    '''
    Found another common pattern amongst some files with some extractable updates that are different than
    those that we already got from above notebook cells.
    The below code parses this pattern, saving the updates.
    First, to more easily keep track of what I have procd and not, move those that I process over to
    a "has_been_processed" sub folder.
    '''
    for file_name in files_with_pattern:
        path_strt = os.path.join(PATH_TO_RAW_MISC_UPDATES, file_name)
        if os.path.exists(path_strt):
            lines = open(path_strt).readlines()
            i = 0
            while i < len(lines):
                toks = lines[i].split()
                if len(toks) == 3 and toks[0] == "Date" and toks[2] == "Changes":
                    i = i + 1
                    toks = lines[i].split()
                    while (toks[0] != "##" and toks[0] != '•'):
                        # get the date line as a folder path
                        d = toks[0]
                        path_to_day_folder = os.path.join(
                            PATH_TO_PROCESSED_UPDATES_BY_YEAR,
                            toks[2][2:4],
                            word_month_to_number_month(toks[1]),
                            d)
                        os.makedirs(path_to_day_folder, exist_ok=True)
                        # gather the lines below
                        per_day_update_ctr = len(os.listdir(path_to_day_folder))
                        i = i + 1
                        line_i = lines[i]
                        toks = line_i.split()
                        while len(toks) > 0:
                            print(path_to_day_folder)
                            with open(os.path.join(path_to_day_folder, f"{d}_{per_day_update_ctr}.txt"), 'w') as f:
                                f.write(line_i)
                            per_day_update_ctr = per_day_update_ctr + 1
                            i = i + 1
                            line_i = lines[i]
                            toks = line_i.split()

                        i = i + 1
                        toks = lines[i].split()

                    shutil.move(path_strt, os.path.join(PATH_TO_RAW_MISC_UPDATES_BEEN_PROCD, file_name))

                i = i + 1


def get_updates_from_other_certain_misc_updates_pattern(files_with_pattern) -> None:
    '''
    Another found pattern of updates to extract:
    ## Year

    day month
    update
    '''
    for file_name in files_with_pattern:
        path_strt = os.path.join(PATH_TO_RAW_MISC_UPDATES, file_name)
        print(path_strt)
        if os.path.exists(path_strt):
            lines = open(path_strt).readlines()
            i = 0
            while i < len(lines):
                toks = lines[i].split()
                if len(toks) == 2:
                    maybe_yr = toks[1]
                    if toks[0] == "##" and maybe_yr in RUNESCAPE_YEARS:
                        i = i + 1
                        toks = lines[i].split()
                        while len(toks) > 0:
                            d = toks[0]
                            path_to_day_folder = os.path.join(
                                PATH_TO_PROCESSED_UPDATES_BY_YEAR,
                                maybe_yr[2:4],
                                word_month_to_number_month(toks[1]),
                                d)
                            os.makedirs(path_to_day_folder, exist_ok=True)
                            print(path_to_day_folder)
                            per_day_update_ctr = len(os.listdir(path_to_day_folder))
                            i = i + 1
                            with open(os.path.join(path_to_day_folder, f"{d}_{per_day_update_ctr}.txt"), 'w') as f:
                                f.write(lines[i])

                            i = i + 1
                            toks = lines[i].split()

                i = i + 1

            shutil.move(path_strt, os.path.join(PATH_TO_RAW_MISC_UPDATES_BEEN_PROCD, file_name))


def convert_json_ms_unix_time_to_yyyy_m_d(unix_time: str):
    '''
    Converts a given unix time string (in millisecond format) into a year, month, day date object.
    Our unix times are in millisecond format.
    '''
    return datetime.datetime.utcfromtimestamp(float(unix_time)/1000).date()

def word_month_to_number_month(month: str) -> str:
    return WORD_MONTH_TO_NUMBER_MAP[month.lower()]

def save_table_as_updates(name_v_release_date) -> None:
    '''
    Takes a row from a quest table and saves it as info released on a certain day.
    Will append to the day's file if it exists already, otherwise, it makes a new one
    '''
    to_write = name_v_release_date["Title"].strip()
    date_toks = name_v_release_date["Date"].strip().split()
    path = os.path.join(
        PATH_TO_PROCESSED_UPDATES_BY_YEAR,
        date_toks[2][2:4],
        word_month_to_number_month(date_toks[1]),
        date_toks[0])
    if os.path.exists(path) and os.path.isdir(path):
        files = os.listdir(path)
        with open(os.path.join(path, files[0]), 'a') as f:
            f.write(f" and {to_write}")
    else:
        as_txt = f"{path}.txt"
        if os.path.exists(as_txt) and os.path.isfile(as_txt):
            with open(as_txt, 'a') as f:
                f.write(f" and {to_write}")
        else:
            with open(as_txt, 'w') as f:
                f.write(to_write)
    