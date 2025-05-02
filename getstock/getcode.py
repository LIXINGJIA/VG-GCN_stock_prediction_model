import os
import tushare as ts
import pandas as  pd
import datetime
token = 'your token'
# 可以登录文章首的链接注册获取
pro = ts.pro_api(token)
# dateToday = datetime.datetime.today().strftime('%Y%m%d')


#获取上海证券交易所的股票代码
def GetList():

        data = pro.stock_basic(exchange='SZSE', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
        # 上交所exchange='SSE'  交易所 SSE上交所 SZSE深交所 BSE北交所
        data.to_csv('stockcode.csv',encoding='utf-8')
        GetStockdetailed()
        # print(type(data))


def GetStockdetailed():
   stockcodedf=pd.read_csv("stockcode.csv",index_col=0,encoding='utf-8')
   stockcodedf=stockcodedf[stockcodedf['list_date']<=20190101]
   print(stockcodedf)
   stockcodedf.to_csv('newstockcode.csv',encoding='utf-8')







if  __name__ == '__main__':
    GetList()
    # GetStockdetailed()
