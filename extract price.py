# importing datetime module
import datetime
import time
import json
from FP567_Lib import *
# assigned regular string date(year,month,date,hour,minute)
date_time = datetime.datetime(2015, 4,12 , 18,00)

unixtime=str(int(time.mktime(date_time.timetuple())))+'000'

# list={item:[[date,price],[date,price]]}
listvy = open('{PATH_TO_MARKET_ITEM_DATA}/all0a3b.json')
list1= json.load(listvy)

def index_2d(myList, v):
    for i, x in enumerate(myList):
        if v in x:
            return i,x.index(v)

for i in list1:

    price=[]
    index = index_2d(list1[i],int(unixtime))

    try:
        price.append(list(map(lambda x : x[1],list1[i][(index[0]-30)::(index[0]+30)])))

    except:
         break
print(price)





