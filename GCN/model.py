import torch.nn as nn
import torch
import torch.nn.functional as F
from pygcn import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.gc3 = GraphConvolution(1, 1)
        self.dropout = dropout
        self.fn = nn.Linear(20,1)

    def forward(self, x, adj):                     #forward是向前传播函数，最终得到网络向前传播的方式为：relu–>fropout–>gc2–>softmax
        x = (self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x=self.gc3(x,adj)
        x=x.reshape(20)
        # x=self.fn(x)
        # x = F.relu(x)

        # x=torch.sigmoid(x)
        return  x
        # return F.log_softmax(x, dim=1)

class finalpredict_model(torch.nn.Module):
    def __init__(self,nfeat, nhid, nclass, dropout):
        super(finalpredict_model,self).__init__()
        self.gcn=GCN(nfeat, nhid, nclass, dropout)
        self.fn1=nn.Linear(120,60)
        self.fn2=nn.Linear(60,1)

    def forward(self, a_feature,a_adj,b_feature,b_adj,c_feature,c_adj,d_feature,d_adj,e_feature,e_adj,f_feature,f_adj):
        a=self.gcn(a_feature,a_adj)
        b=self.gcn(b_feature,b_adj)
        c=self.gcn(c_feature,c_adj)
        d=self.gcn(d_feature,d_adj)
        e=self.gcn(e_feature,e_adj)
        f=self.gcn(f_feature,f_adj)
        x = torch.cat([a,b,c,d,e,f])
        x=self.fn1(x)
        x=self.fn2(x)
        x=torch.sigmoid(x)
        return x