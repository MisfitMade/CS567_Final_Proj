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

import pandas as pd
import numpy as np

PROJECT_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
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
PATH_TO_PLOTS = os.path.join(PROJECT_ROOT_DIR, "plots")


RUNESCAPE_YEARS = list(map(str, range(1998, 2023)))
MONTHS = list(map(str.lower, calendar.month_name[1:]))
WORD_MONTH_TO_NUMBER_MAP = {
    "january": "01", "february": "02", "march": "03", "april": "04", "may": "05", "june": "06",
    "july": "07", "august": "08", "september": "09", "october": "10", "november": "11", "december": "12"
} 
DAYS_OF_MONTH = list(map(str, range(1, 32)))
DAYS_OF_WEEK_ABRV = calendar.day_abbr[:]
NUMBER_OF_SECONDS_IN_A_UNIX_DAY = 86400

class MarketItem:
    def __init__(self, path_to_item_json):
        '''
        Given a path to a json market item, reads it in and makes a MarketItem out of it.
        '''
        item_data = json.load(open(path_to_item_json))
        item_keys = item_data.keys()
        item_vals = item_data.values()

        if len(item_keys) > 1 or len(item_vals) > 1:
            raise Exception(f"The json at {path_to_item_json} is malformed.")

        self.item_id = list(item_keys)[0]
        # this is a unix time -> [price, amount sold] dict
        self.time_to_info_dict = {}
        for unit_time_lis in list(item_vals)[0]:
            self.time_to_info_dict[unit_time_lis[0]] = unit_time_lis[1:]
        self.number_of_unit_times = len(self.time_to_info_dict)
    

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
        self.market_items = {}
        self.item_ids = []
        for market_item in market_is:
            self.item_ids.append(market_item.item_id)
            self.market_items[market_item.item_id] = market_item
        
        # number_of_infos_in_oldest_item is the number of [time, price, amount_sold] instances in the
        # item that has the most [time, price, amount_sold] instances, aka the longest runnning/oldest item.
        self.number_of_infos_in_oldest_item = np.array(list(map(lambda market_item: market_item.number_of_unit_times, market_is))).max()

        # markets_time_span is a list of the unix times from the item that has the most [time, price, amount_sold] instances.
        self.markets_time_span = sorted(list(self.get_items_that(
                lambda item: len(item.time_to_info_dict) == self.number_of_infos_in_oldest_item)[0].time_to_info_dict.keys()))

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

    def build_and_write_features_matrix(self):
        market_rep = pd.DataFrame({})

        # self.markets_time_span is sorted during Market object construction, so we can
        # iterate over it now and know that we are increasing in time.
        for ms_unix_time in self.markets_time_span:
            date = convert_json_ms_unix_time_to_yyyy_m_d(ms_unix_time)



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
            print("YAY")
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
    Converts a given unix time string into a year, month, day date object.
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
    