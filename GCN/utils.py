import numpy as np
import scipy.sparse as sp
import torch

#特征独热码处理
def encode_onehot(labels):
    # 将所有的标签整合成一个不重复的列表
    classes = set(labels)     # set() 函数创建一个无序不重复元素集
    '''enumerate()函数生成序列，带有索引i和值c。
        这一句将string类型的label变为int类型的label，建立映射关系
        np.identity(len(classes)) 为创建一个classes的单位矩阵
        创建一个字典，索引为 label， 值为独热码向量（就是之前生成的矩阵中的某一行）'''
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    # 为所有的标签生成相应的独热码
    # map() 会根据提供的函数对指定序列做映射。
    # 这一句将string类型的label替换为int类型的label
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

#数据加载和处理
def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    #数据包含id，features和labels
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    #节点的特征，维度为 2708 * 1433，类型为 np.ndarray
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    #节点的标签，总共包括7个类别，类型为 np.ndarray
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    #节点的id
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    #每个id进行编号
    idx_map = {j: i for i, j in enumerate(idx)}
    #
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)   #map会根据提供的函数对指定序列做映射，flatten()就是降到一维
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

#特征归一化函数
def normalize(mx):
    """Row-normalize sparse matrix""" #对稀疏矩阵进行正则化处理
    rowsum = np.array(mx.sum(1))      #会得到一个（2708,1）的矩阵
    r_inv = np.power(rowsum, -1).flatten() #数组元素求-1次方
    # 在计算倒数的时候存在一个问题，如果原来的值为0，则其倒数为无穷大，因此需要对r_inv中无穷大的值进行修正，更改为0
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

# #精度计算函数
# def accuracy(output, labels):
#     # 使用type_as(tesnor)将张量转换为给定类型的张量
#     preds = output.max(1)[1].type_as(labels)
#     # 记录等于preds的label eq:equal
#     correct = preds.eq(labels).double()
#     correct = correct.sum()
#     return correct / len(labels)
#
# #将scipy稀疏矩阵转换为Torch稀疏张量
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    """
        numpy中的ndarray转化成pytorch中的tensor : torch.from_numpy()
        pytorch中的tensor转化成numpy中的ndarray : numpy()
        """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)