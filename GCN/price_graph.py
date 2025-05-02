
import os
import sys
import json
import pickle
# import pyunicorn
import pandas as pd
# from pyunicorn import timeseries
import visibility_graph
# from mycode import vg  as  myvg
from multiprocessing import Pool
import  numpy as  np
import  networkx  as nx
from itertools import combinations

PWD = os.path.dirname(os.path.realpath(__file__))

# T 是20  PI 是'vol', 'amount', 'high', 'open', 'low', 'close
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
        # net = timeseries.visibility_graph.VisibilityGraph(time_series.values)
        net=visibility_graph.visibility_graph(time_series.values)
        a = np.array(nx.adjacency_matrix(net).todense())
        # a = nx.adjacency_matrix(net).todense()
        vgs[i] = a
        # print(i)
        # print("vgs", vgs)
        # print(vg_file)
        # for key, value  in vgs.items():
            # print("vgs", value.shape)
    with open(vg_file, 'wb') as fp:
        pickle.dump(vgs, fp)



if __name__ == '__main__':
    vol_price = ['vol', 'amount', 'high', 'open', 'low', 'close']
    # vol_price = ['vol']
    files = os.listdir('../data')
    for PI in vol_price:
        vg_dir = os.path.join('../VG', PI)
        if not os.path.exists(vg_dir):
            os.makedirs(vg_dir)
        pool = Pool()
        pool.map(make_visibility_graph, [(f, PI, 20) for f in files])
        pool.close()
        pool.join()
