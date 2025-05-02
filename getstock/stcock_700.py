import json
import re


def  code_700():
    a=[]
    with open("700.json",'r',encoding="utf8") as fp:
        json_data=json.load(fp)
        stock_list=json_data['stock']["_value"]
        print(stock_list)
        print(len(stock_list))

        for i in stock_list:
            j=re.split(r'(\d+)', i)
            k=j[1]+'.'+j[0]
            a.append(k)

    return a