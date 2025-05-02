import numpy
import numpy as  np
import  os
import pandas as  pd
import torch.utils.data as Data
import  torch
import pickle
import  networkx  as nx
import visibility_graph
from multiprocessing import Pool


# fil_stock='600000.SH'

def make_visibility_graph(input_):
    file_, PI, T = input_
    vg_dir = os.path.join('../VG', PI)
    vg_file = os.path.join(vg_dir, '%s.pickle' % file_[:-4])
    df = pd.read_csv(os.path.join('../data', file_), index_col=0)
    df.set_index(df.index.astype('str'), inplace=True)
    vgs = {}
    for i in df.index:
        iloc = list(df.index).index(i)

        if df.loc[:i,:].shape[0] < T:
            continue
        # print("i是", i)
        time_series = df.iloc[iloc-T+1:iloc+1][PI]
        # print("time_series",time_series)
        if len(set(time_series.values)) == 1:
            continue
        net=visibility_graph.visibility_graph(time_series.values)
        a = np.array(nx.adjacency_matrix(net).todense())
        vgs[i] = a

    with open(vg_file, 'wb') as fp:
        pickle.dump(vgs, fp)

def price_graph(fil_stock):
    fil_stock=fil_stock+'.csv'
    vol_price = ['vol', 'amount', 'high', 'open', 'low', 'close']

    for PI in vol_price:
        vg_dir = os.path.join('../VG', PI)
        if not os.path.exists(vg_dir):
            os.makedirs(vg_dir)
        # pool = Pool()
        # pool.map(make_visibility_graph, [(f, PI, 20) for f in files])
        # pool.close()
        # pool.join()
        make_visibility_graph((fil_stock,PI,20))


def minmax_norm(df):
    return (df - df.min()) / ( df.max() - df.min())

def  dataset(fil_stock):
    df_main=pd.read_csv('../data/'+fil_stock+'.csv')
    sel_col = ['open','high','low','close','vol','amount']

    #判断是否有缺失值
    dataNull=np.sum( df_main.isnull())
    # print(dataNull)

    # 缺失值填充，使用上一个有效值
    df_main = df_main.fillna(method='ffill')
    np.sum(df_main.isnull())


#错误的涨跌计算代码
    # #计算股票涨跌 updown
    # df_main['updown']=df_main['close'].diff()
    # df_main.loc[df_main['updown']>0,'updown']=1
    # df_main.loc[df_main['updown']<=0,'updown']=0



# 计算股票涨跌updown
    lis=[]
    for i in  df_main.index[:-1]:
        a=df_main.loc[i+1]['close']-df_main.loc[i]['close']
        lis.append(a)
    lis.append(0)
    df_main['updown']=lis
    df_main=df_main.drop(df_main.tail(1).index)
    df_main.loc[df_main['updown']>0,'updown']=1
    df_main.loc[df_main['updown']<=0,'updown']=0
    df_main.to_csv('../data/'+fil_stock+'.csv',index=False)



    # 按照模型的接口组织数据
    y_true = df_main['updown'][19:]
    y_true = numpy.array(y_true)
    y_true = y_true.reshape(-1, 1)
    with open('y_true', 'wb') as fp:
        pickle.dump(y_true, fp)




def feature(file):
    file = file + '.csv'
    df_main = pd.read_csv(os.path.join('../data', file))

    # df_main = pd.read_csv('../data/600000.SH.csv')
    sel_col = ['open', 'high', 'low', 'close', 'vol', 'amount']
    datelist=list(df_main['date'])
    # print(datelist)
    f={}
    df_main.set_index('date', inplace=True)
    df_main = minmax_norm(df_main)
    # print(df_main)
    # print(df_main)
    for PI  in sel_col:
        fe_dir = os.path.join('feature', PI)
        fe_file = os.path.join(fe_dir,'%s.pickle' % file[:-4])
        for i in df_main.index:
            iloc = list(df_main.index).index(i)
            # print(i)
            # print(iloc)
            if df_main.loc[:i, :].shape[0] < 20:
                continue
            # print("i是", i)
            time_series = df_main.iloc[iloc - 20 + 1:iloc + 1][PI]
            i=str(i)
            f[i]=time_series.values

        # print(f)
        if not os.path.exists(fe_dir):
            os.makedirs(fe_dir)
        with open(fe_file, 'wb') as fp:
            pickle.dump(f, fp)






def adj_feature(PI, file):
    with open('feature/' + PI + '/' + file + '.pickle', 'rb') as fp:
        a_f = pickle.load(fp)
        # print("f", a_f)
    with open('../VG/' + PI + '/' + file + '.pickle', 'rb') as fp:
        a_vg = pickle.load(fp)
        # print('g', a_vg)

        # print(a_vg)
    df_main = pd.read_csv(os.path.join('../data', file + '.csv'))
    datelist = list(df_main['date'])
    # print(datelist)
    adj = []
    feat = []
    for i in datelist[19:]:
        i = str(i)
        adj.append(a_vg[i])
        feat.append(a_f[i])

    a_adj = np.asarray(adj)
    a_feature = np.asarray(feat)

    return a_feature, a_adj


# 训练数据处理
def  traindata(file):
    with open('y_true', 'rb') as fp:
        y_ture = pickle.load(fp)
    sel_col = ['open', 'high', 'low', 'close', 'vol', 'amount']


    (a_f,a_vg) = adj_feature("open",file)
    (b_f,b_vg) = adj_feature("high",file)
    (c_f,c_vg) = adj_feature("low",file)
    (d_f,d_vg) = adj_feature("close",file)
    (e_f,e_vg) = adj_feature("vol",file)
    (f_f,f_vg) = adj_feature("amount",file)


    # 这里按照7:3的比例划分训练集和测试集
    #
    # # print(type(a))
    train_size=int(np.round(y_ture.shape[0]*0.7))


    # #转换为tensor
    train_a_f=torch.from_numpy(a_f[:train_size]).type(torch.Tensor)
    train_a_adj=torch.from_numpy(a_vg[:train_size]).type(torch.Tensor)
    test_a_f=torch.from_numpy(a_f[train_size:]).type(torch.Tensor)
    test_a_adj=torch.from_numpy(a_vg[train_size:]).type(torch.Tensor)

    train_b_f = torch.from_numpy(b_f[:train_size]).type(torch.Tensor)
    train_b_adj = torch.from_numpy(b_vg[:train_size]).type(torch.Tensor)
    test_b_f = torch.from_numpy(b_f[train_size:]).type(torch.Tensor)
    test_b_adj = torch.from_numpy(b_vg[train_size:]).type(torch.Tensor)

    train_c_f = torch.from_numpy(c_f[:train_size]).type(torch.Tensor)
    train_c_adj = torch.from_numpy(c_vg[:train_size]).type(torch.Tensor)
    test_c_f = torch.from_numpy(c_f[train_size:]).type(torch.Tensor)
    test_c_adj = torch.from_numpy(c_vg[train_size:]).type(torch.Tensor)

    train_d_f = torch.from_numpy(d_f[:train_size]).type(torch.Tensor)
    train_d_adj = torch.from_numpy(d_vg[:train_size]).type(torch.Tensor)
    test_d_f = torch.from_numpy(d_f[train_size:]).type(torch.Tensor)
    test_d_adj = torch.from_numpy(d_vg[train_size:]).type(torch.Tensor)

    train_e_f = torch.from_numpy(e_f[:train_size]).type(torch.Tensor)
    train_e_adj = torch.from_numpy(e_vg[:train_size]).type(torch.Tensor)
    test_e_f = torch.from_numpy(e_f[train_size:]).type(torch.Tensor)
    test_e_adj = torch.from_numpy(e_vg[train_size:]).type(torch.Tensor)

    train_f_f = torch.from_numpy(f_f[:train_size]).type(torch.Tensor)
    train_f_adj = torch.from_numpy(f_vg[:train_size]).type(torch.Tensor)
    test_f_f = torch.from_numpy(f_f[train_size:]).type(torch.Tensor)
    test_f_adj = torch.from_numpy(f_vg[train_size:]).type(torch.Tensor)



    train_y = torch.from_numpy(y_ture[:train_size]).type(torch.Tensor)
    test_y=torch.from_numpy(y_ture[train_size:]).type(torch.Tensor)


    train=torch.utils.data.TensorDataset(train_a_f,train_a_adj,train_b_f,train_b_adj,train_c_f,train_c_adj,train_d_f,
                                         train_d_adj,train_e_f,train_e_adj,train_f_f,train_f_adj,train_y)


    test=torch.utils.data.TensorDataset(test_a_f,test_a_adj,test_b_f,test_b_adj,test_c_f,test_c_adj,test_d_f,test_d_adj,
                                        test_e_f,test_e_adj,test_f_f,test_f_adj,test_y)

    return train, test




if __name__ == '__main__':
    dataset('000897.sz')
    # feature()
    #
    # # dataset()
    # traindata()