import html2text
import glob
import os

# All files and directories ending with .txt and that don't begin with a dot:
print(glob.glob("C:\\Users\\nicho\PycharmProjects\\CS567_Final_Proj_backup\\resources\\*\\*.html"))

h = html2text.HTML2Text()
# Ignore converting links from HTML
h.ignore_links = True

for file in glob.glob("C:\\Users\\nicho\PycharmProjects\\CS567_Final_Proj_backup\\resources\\*\\*.html"):
    print(file)
    os.remove(file)



