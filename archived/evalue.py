from sklearn import metrics
import numpy as np
import pandas as pd
import pickle



def metrics_dict(path, name):
    result = pickle.load(open(path, 'rb'))
    dict_ = pickle.load(open('../result/class', 'rb'))
    # data = pd.read_excel('../dataset/train/train.xlsx')

    # {'angry': 0, 'happy': 1, 'neutral': 2, 'surprise': 3, 'sad': 4, 'fear': 5}
    idx = pickle.load(open('../result/class_swapped', 'rb'))#类别对应序号
    print(idx)
    metric = {}
    for i in result.keys():
        count = 0
        if i not in metric:
            metric[i] = dict()
        for j in result[i].keys():
            if i == j:
                metric[i]['TP'] = result[i][j]
            else:
                count += result[i][j]
        metric[i]['FN'] = count

    for i in idx.keys():
        count = 0
        for j in result.keys():
            if j != i:
                for k in result[j].keys():
                    if i == k:
                        count += result[j][k]
        metric[i]['FP'] = count

    for i in idx.keys():
        if 'TP' not in metric[i]:
            metric[i]['TP'] = 0
        if 'FN' not in metric[i]:
            metric[i]['FN'] = 0
        if 'FP' not in metric[i]:
            metric[i]['FP'] = 0

    pack_dict = [[] for i in range(len(metric))]
    id = 0
    for i in metric.keys():
        pack_dict[id].append(i)
        pack_dict[id].append(metric[i]['TP'])
        pack_dict[id].append(metric[i]['FP'])
        pack_dict[id].append(metric[i]['FN'])
        id += 1

    df = pd.DataFrame(pack_dict, columns=['emotion', 'TP', 'FP', 'FN'])
    df.to_excel('../result/scores/' + name + '.xlsx')

    return metric


def model_scores(path, name):
    dict_ = metrics_dict(path, name)
    # print(dict_)
    acc, macro_p, macro_r = 0, 0, 0
    sum1 = 0
    p = []
    r = []
    for i in dict_.keys():
        sum1 += dict_[i]['TP'] + dict_[i]['FN']
        acc += dict_[i]['TP']
        if dict_[i]['TP'] + dict_[i]['FP']:
            p.append(dict_[i]['TP'] / (dict_[i]['TP'] + dict_[i]['FP']))
        else:
            p.append(0)
        if dict_[i]['TP'] + dict_[i]['FN']:
            r.append(dict_[i]['TP'] / (dict_[i]['TP'] + dict_[i]['FN']))
        else:
            r.append(0)

    acc = round(acc / sum1, 3)
    macro_p = round(sum(p) / len(dict_), 3)
    macro_r = round(sum(r) / len(dict_), 3)

    return acc, macro_p, macro_r


def metrics_dict_b(path):
    result = pickle.load(open(path, 'rb'))
    dict_ = pickle.load(open('../result/class_', 'rb'))
    data = pd.read_excel('../dataset/train/b_train.xlsx')

    # {'angry': 0, 'happy': 1, 'neutral': 2, 'surprise': 3, 'sad': 4, 'fear': 5}
    # print(result)
    # idx = get_label_dict(data, list(data.columns)[-1])
    metric = {'TP': result[1][1], 'TN': result[0][0], 'FP': result[1][0], 'FN': result[0][1]}

    if 'TP' not in metric:
        metric['TP'] = 0
    if 'FN' not in metric:
        metric['FN'] = 0
    if 'FP' not in metric:
        metric['FP'] = 0
    if 'TN' not in metric:
        metric['TN'] = 0

    return metric


def model_scores_b(path):
    dict_ = metrics_dict_b(path)
    # print(dict_)
    acc, p, r, f1 = 0, 0, 0, 0
    sum1 = 0
    sum1 += dict_['TP'] + dict_['FN'] + dict_['TN'] + dict_['FP']
    acc += dict_['TP'] + dict_['TN']
    if dict_['TP'] + dict_['FP']:
        p = dict_['TP'] / (dict_['TP'] + dict_['FP'])
    if dict_['TP'] + dict_['FN']:
        r = dict_['TP'] / (dict_['TP'] + dict_['FN'])

    acc = round(acc / sum1, 3)
    f1 = round(2 * p * r / (p + r), 3)

    return acc, p, r, f1


if __name__ == '__main__':
    a1, mp1, mr1 = model_scores('../result/LeNet.matrix', 'LeNet')
    a2, mp2, mr2 = model_scores('../result/VGG.matrix', 'VGG')
    a3, mp3, mr3 = model_scores('../result/RESNET.matrix', 'RESNET')
    # a4, mp4, mr4 = model_scores('../result/LSTM.matrix', 'LSTM')

    scores_list = [[] for i in range(3)]
    scores_list[0].append('Lenet')
    scores_list[0].append(a1)
    scores_list[0].append(mp1)
    scores_list[0].append(mr1)
    scores_list[1].append('VGG')
    scores_list[1].append(a2)
    scores_list[1].append(mp2)
    scores_list[1].append(mr2)
    scores_list[2].append('RESNET')
    scores_list[2].append(a3)
    scores_list[2].append(mp3)
    scores_list[2].append(mr3)
    # scores_list[3].append('LSTM')
    # scores_list[3].append(a4)
    # scores_list[3].append(mp4)
    # scores_list[3].append(mr4)

    df = pd.DataFrame(scores_list, columns=['model', 'accuracy', 'macro_P', 'macro_R'])
    df.to_excel('../result/scores/eval_scores.xlsx')

    # a1, p1, r1, f1 = model_scores_b('../result/RNN_b.matrix')
    # a2, p2, r2, f2 = model_scores_b('../result/GRU_b.matrix')
    # a3, p3, r3, f3 = model_scores_b('../result/LSTM_b.matrix')
    #
    # scores_list = [[] for i in range(3)]
    # scores_list[0].append('RNN')
    # scores_list[0].append(a1)
    # scores_list[0].append(p1)
    # scores_list[0].append(r1)
    # scores_list[0].append(f1)
    # scores_list[1].append('GRU')
    # scores_list[1].append(a2)
    # scores_list[1].append(p2)
    # scores_list[1].append(r2)
    # scores_list[1].append(f2)
    # scores_list[2].append('LSTM')
    # scores_list[2].append(a3)
    # scores_list[2].append(p3)
    # scores_list[2].append(r3)
    # scores_list[2].append(f3)
    #
    # df = pd.DataFrame(scores_list, columns=['model', 'accuracy', 'Precision', 'Recall', 'F1-Measure'])
    # df.to_excel('../result/scores/eval_scores_b.xlsx')
