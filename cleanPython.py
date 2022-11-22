import json
import html2text
import glob
# Opening JSON file
import statistics
from scipy.stats import entropy

# returns JSON object as
# a dictionary
dictprice = {}
# dictamount = {}


for y in glob.glob("C:\\Users\\nicho\PycharmProjects\\CS567_Final_Proj_backup\\resources\\*\\*.json"):
    f = open(y)
    data = json.load(f)
    for i in data:
       price = {}
       for x in data[i]:
           price[x[0]]  = x[1]

       dictprice[i] = price
    f.close()


var =  list()
ent =  list()

bond = {}


dictpriceUpdated = {}

for key in dictprice.keys():

    dictpriceUpdated[key] = []
    for times in sorted(dictprice[key]):
        if dictprice['13190'].keys().__contains__(times):
            dictpriceUpdated[key].append([times, dictprice[key][times]/dictprice['13190'][times]])



dictprice.clear()





json_object = json.dumps(dictpriceUpdated)

# Writing to bondNormal.json
with open("sample.json", "w") as outfile:
    outfile.write(json_object)


# print(dictprice['13190'])


#
# for am in dictprice.keys():
#     remove =  min( len(dictprice[am]),bondsize)
    # var.append([am,statistics.variance(dictprice[am][-remove:-1])])
    # ent.append([am,entropy(dictprice[am][-remove:-1],base = 2)])

    # for am1 in dictprice.keys():
    #     min1 = min( len(dictprice[am]),len(dictprice[am1]))
        # mul.append([am,am1, entropy(dictprice[am][-min1:-1],qk=dictprice[am1][-min1:-1], base=2)])




print(sorted(var, key=lambda student: student[1]))
print(sorted(ent, key=lambda student: student[1]))
# print(sorted(mul, key=lambda student: student[2]))
# Closing file

