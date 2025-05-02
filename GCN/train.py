from __future__ import division
from __future__ import print_function
import numpy as np
import processdata
import torch
from model import finalpredict_model



def onetrain(file):

    dataset = processdata.dataset(fil_stock=file)
    dataset

    price_graph=processdata.price_graph(fil_stock=file)
    price_graph


    feature=processdata.feature(file=file)
    feature
    # 导入数据
    train_data,test_data = processdata.traindata(file=file)

    model=finalpredict_model(
                nfeat=1,
                nhid=1,
                nclass=1,
                dropout=0.2)
    #之前的 dropout=0.2

    #优化函数
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=0.001)  # 使用Adam优化算法  lr=0.001

    #损失函数
    criterion = torch.nn.BCELoss()
    # criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(100):
        for  a_feature, a_adj,b_feature, b_adj,c_feature, c_adj,d_feature, d_adj,e_feature, e_adj,f_feature, f_adj,train_y in train_data:

            a_feature = a_feature.reshape(20,1)
            b_feature = b_feature.reshape(20,1)
            c_feature = c_feature.reshape(20,1)
            d_feature = d_feature.reshape(20,1)
            e_feature = e_feature.reshape(20,1)
            f_feature = f_feature.reshape(20,1)

            # print('y',train_y)
            # print('y',train_y.shape)
            # print('a_f', a_feature)
            # print('a_f', a_feature.shape)
            # print('a_adj', a_adj)
            # print('a_adj', a_adj.shape)

            out = model(a_feature,a_adj,b_feature, b_adj,
                        c_feature, c_adj,d_feature, d_adj,e_feature, e_adj,f_feature, f_adj)
            # print('out', out)
            # print('out', out.shape)

            loss = criterion(out, train_y)
            optimiser.zero_grad()
            loss.backward()  # 计算梯度
            optimiser.step()
        if (epoch + 1) % 100 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))



    # 测试集
    model.eval()
    pred_proba, pred_01, labels = np.array([]), np.array([]), np.array([])
    for a_feature, a_adj,b_feature, b_adj,c_feature, c_adj,d_feature, d_adj,e_feature, e_adj,f_feature, f_adj,test_y in test_data:
        a_feature = a_feature.reshape(20, 1)
        b_feature = b_feature.reshape(20, 1)
        c_feature = c_feature.reshape(20, 1)
        d_feature = d_feature.reshape(20, 1)
        e_feature = e_feature.reshape(20, 1)
        f_feature = f_feature.reshape(20, 1)
        test_pred_y = model(a_feature, a_adj, b_feature, b_adj,
                    c_feature, c_adj, d_feature, d_adj, e_feature, e_adj, f_feature, f_adj)
        # print(criterion(test_pred_y, test_y))
        # print(y_test_pred.shape)
        # print(testy.shape)
        pred_proba = np.concatenate((pred_proba, test_pred_y.detach().numpy().reshape(-1)), axis=0)
        # print('pre_proba',pred_proba)
        pred_01 = np.array(list(map(lambda x: 1 if x >= 0.5 else 0, pred_proba)))
        # print('pred_01',pred_01)
        # print('test_y',test_y.shape)
        #
        labels = np.concatenate((labels, test_y.detach().numpy().reshape(-1)), axis=0)
        # print('labels',labels)

    acc = sum(map(lambda x, y: 1 if x == int(y) else 0, pred_01, np.array(labels, dtype=int))) * 1.0 / len(labels)
    print('acc=', acc)

    #
    # 计算混淆矩阵
    def compute_confusion_matrix(precited, expected):
        part = precited ^ expected  # 对结果进行分类，亦或使得判断正确的为0,判断错误的为1
        pcount = np.bincount(part)  # 分类结果统计，pcount[0]为0的个数，pcount[1]为1的个数
        tp_list = list(precited & expected)  # 将TP的计算结果转换为list
        fp_list = list(precited & ~expected)  # 将FP的计算结果转换为list
        tp = tp_list.count(1)  # 统计TP的个数
        fp = fp_list.count(1)  # 统计FP的个数
        tn = pcount[0] - tp  # 统计TN的个数
        fn = pcount[1] - fp  # 统计FN的个数
        return tp, fp, tn, fn

    # 计算常用指标
    def compute_indexes(tp, fp, tn, fn):
        accuracy = (tp + tn) / (tp + tn + fp + fn)  # 准确率
        try:

                precision = tp / (tp + fp)  # 精确率

                recall = tp / (tp + fn)  # 召回率
                F1 = (2 * precision * recall) / (precision + recall)  # F1
        except:

            return 0 ,0 , 0 , 0
        else:

            return accuracy, precision, recall, F1

    test_y=test_y.reshape(-1)
    precited = np.array(pred_01)
    expected = np.array(labels).astype(int)

    # precited = np.array([1,1,0,0,0,1,0,0,0,0,1,1,0,1,1,0])
    # expected = np.array([1,0,0,0,1,1,0,1,0,1,1,0,0,0,1,1])
    # print('precited', precited)
    # print('expected', expected)
    print(file)
    tp, fp, tn, fn = compute_confusion_matrix(precited, expected)
    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"TN: {tn}")
    print(f"FN: {fn}")
    accuracy, precision, recall, F1 = compute_indexes(tp, fp, tn, fn)
    print(f"Accuracy:  {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall:    {recall}")
    print(f"F1:        {F1}")

    return accuracy, precision, recall, F1

if __name__ == '__main__':
    onetrain()