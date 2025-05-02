import tushare as ts
import pandas as  pd

token = 'your token'


pro = ts.pro_api(token)

#读取股票代码
stockcodedf=pd.read_csv("newstockcode.csv",index_col=0,encoding='utf-8')

#将股票代码转换成列表
stockcodeList=stockcodedf.ts_code.tolist()

# [2010-01-01,2020-01-01)1
# [2013-01-01,2020-01-01)2
# [2015-07-01,2020-01-01)3
# [2012-01-01,2020-01-01)4
# [2010-01-01,2020-01-01)5
# [2013-01-01,2020-01-01)6

# start_date='20190601'
# end_date='20221202'

# 1
# start_date='20100101'
# end_date='20200101'
# stockcodeList1=['600809.sh']

# 2
# start_date='20130101'
# end_date='20200101'
# stockcodeList1=['300330.sz']

# 3
# start_date='20150701'
# end_date='20200101'
# stockcodeList1=['603808.sh']
#4
# start_date='20120101'
# end_date='20200101'
# stockcodeList1=['002580.sz']


#5
# start_date='20100101'
# end_date='20200101'
# stockcodeList1=['601318.sh']


#6
# start_date='20130101'
# end_date='20200101'
# stockcodeList1=['603123.sh']


start_date='2019010'
end_date='20221231'
import  stcock_700
stockcodeList1=stcock_700.code_700()
#
#
# start_date='20190101'
# end_date='20221231'

# for ts_code  in  stockcodeList[:200]:
for ts_code  in  stockcodeList1[:200]:

    #日线行情


    stockdaily=pro.daily(ts_code=ts_code,start_date=start_date,end_date=end_date)

    #每日指标
    # df = pro.daily_basic(ts_code='', trade_date='20180726',
    #                      fields='ts_code,trade_date,turnover_rate,volume_ratio,pe,pb')

    stockdaily.sort_values(by='trade_date',inplace=True)
    stockdaily.drop(['pct_chg','change','pre_close','ts_code'],axis=1,inplace=True)
    stockdaily.rename(columns = {"trade_date": "date"},  inplace=True)
    stockdaily=stockdaily.set_index('date')

    stockdaily.to_csv('../data/'+ts_code+'.csv',encoding='utf-8')

