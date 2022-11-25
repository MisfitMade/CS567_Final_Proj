import html2text
import glob
import os

from FP567_Lib import *

# All files and directories ending with .txt and that don't begin with a dot:
print(glob.glob(f"{PATH_TO_RAW_UPDATES}\\*\\*.html"))

h = html2text.HTML2Text()
# Ignore converting links from HTML
h.ignore_links = True

for file in glob.glob(f"{PATH_TO_RAW_UPDATES}\\*\\*.html"):
    print(file)
    os.remove(file)



