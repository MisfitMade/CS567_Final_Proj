import os
import calendar
import shutil
import matplotlib.pyplot

PROJECT_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
PATH_TO_RESOURCES = os.path.join(PROJECT_ROOT_DIR, "resources")
PATH_TO_MARKET_ITEM_DATA = os.path.join(PATH_TO_RESOURCES, "market_item_data")
PATH_TO_RAW_UPDATES = os.path.join(PATH_TO_RESOURCES, "Raw_Updates")
PATH_TO_RAW_BY_DAY_UPDATES = os.path.join(PATH_TO_RAW_UPDATES, "by_day_updates")
PATH_TO_RAW_BY_YEAR_UPDATES = os.path.join(PATH_TO_RAW_UPDATES, "by_year_updates")
PATH_TO_RAW_GAME_UPDATES = os.path.join(PATH_TO_RAW_UPDATES, "game_updates")
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
                    while (toks[0] != "##"):
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



def word_month_to_number_month(month: str) -> str:
    return WORD_MONTH_TO_NUMBER_MAP[month.lower()]