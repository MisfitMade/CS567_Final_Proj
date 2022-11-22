import html2text
import glob
# All files and directories ending with .txt and that don't begin with a dot:
print(glob.glob("C:\\Users\\nicho\PycharmProjects\\CS567_Final_Proj_backup\\resources\\*\\*.html"))

h = html2text.HTML2Text()
# Ignore converting links from HTML
h.ignore_links = True

for file in glob.glob("C:\\Users\\nicho\PycharmProjects\\CS567_Final_Proj_backup\\resources\\*\\*.html"):

    f = open(file, "r", encoding='utf-8')
    y = open(f.name+".clean", "w", encoding='utf-8')
    y.write(h.handle(f.read()))

