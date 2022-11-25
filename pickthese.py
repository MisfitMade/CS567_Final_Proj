import json
import html2text
import glob

from FP567_Lib import *

# Opening JSON file
f = open(f'{PATH_TO_MARKET_ITEM_DATA}\\all0a3b.json')
import statistics
from scipy.stats import entropy

# returns JSON object as
# a dictionary
dictprice = {}
dictamount = {}

data = json.load(f)
for y in glob.glob(f"{PATH_TO_MARKET_ITEM_DATA}\\*.json"):
    f = open(y)
    data = json.load(f)
    for i in data:
       price = {}
       quant = []
       for x in data[i]:
           price[x[0]]  = x[1]
           if len(x) == 3:
               quant.append([x[0],x[2]])

       dictamount[i] = quant
       dictprice[i] = price
    f.close()

dictpriceUpdated= {}


for key in dictprice.keys():

    dictpriceUpdated[key] = []
    for times in sorted(dictprice[key]):
        if dictprice['13190'].keys().__contains__(times):
            dictpriceUpdated[key].append( dictprice[key][times]/ dictprice['13190'][times]*2000000)
# /dictprice['13190'][times]*6993569
outlist = []
for items in dictpriceUpdated:
    outlist.append( [items,
                     statistics.variance(dictpriceUpdated[items]),
                     entropy(dictpriceUpdated[items],base = 2),
                     len(dictpriceUpdated[items]),
                     sum(dictpriceUpdated[items])/len(dictpriceUpdated[items])
                    ]
                    )

df = pd.DataFrame(outlist, columns=["id", "var", "entropy","amount" ,"avg"])
outlist1 = []
for items in dictpriceUpdated:

    outlist1.append( [items,
                     statistics.variance(dictpriceUpdated[items]),
                     entropy(dictpriceUpdated[items],base = 2),1
                     # (max(dictpriceUpdated[items])-min(dictpriceUpdated[items]))/min(dictpriceUpdated[items]),
                     # sum(dictpriceUpdated[items])/len(dictpriceUpdated[items])
                    ]
                    )
import pandas as pd
df = pd.DataFrame(outlist1, columns=["id", "var", "entropy", "avg"])





for x in  sorted(outlist, key=lambda student: student[1]):
    if x[1] < 1:
        print(x)
# print(sorted(outlist, key=lambda student: student[1]))
vard = {}
for yy in range(0, 17000):
    vard[yy] = 0
for x in  sorted(outlist, key=lambda student: student[1]):

    for yy in range(0,17000):
        if x[1] < yy:
            vard[yy]=vard[yy]+1


print(vard)

df1 = pd.DataFrame(outlist1, columns=["id", "var", "entropy"])




