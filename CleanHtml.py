import html2text
import glob

from FP567_Lib import *

# All files and directories ending with .txt and that don't begin with a dot:
print(glob.glob(f"{PATH_TO_RAW_UPDATES}\\*\\*.html"))

h = html2text.HTML2Text()
# Ignore converting links from HTML
h.ignore_links = True

for file in glob.glob(f"{PATH_TO_RAW_UPDATES}\\*\\*.html"):

    f = open(file, "r", encoding='utf-8')
    y = open(f.name+".clean", "w", encoding='utf-8')
    y.write(h.handle(f.read()))

