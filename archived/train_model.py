import torch
import torch.nn as nn
import numpy as np
import pandas as pd
# from data_get import get_list, get_data
# from create_loader import create_loader
import pickle
from tqdm import trange
# from sklearn.metrics import accuracy_score


if __name__ == '__main__':
    # torch.backends.cudnn.enabled = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 100
    # device = torch.device("cpu")
    n_epoch = 20
    lr = 0.001
    output_size = 6
    hidden_dim = 200
    n_layers = 2
    embedding_dim = 200
    model_name = 'RNN'
    opt_name = 'Adam'

    data0 = pd.read_excel("../dataset/train/train.xlsx")
    # test0 = pd.read_excel("../dataset/eval/eval.xlsx")
    # print(data.head())
    train_data, train_label = get_list(data0, 1, '文本', '情绪标签')
    train_data, len_, pro_dict = get_data(train_data)
    pickle.dump(pro_dict, open('./dict', "wb"), protocol=4)
    train_label = np.array(train_label)
    senti_dict = pickle.load(open('../result/class', 'rb'))

    train_loader, test_loader = create_loader(train_data, train_label, batch_size)

    criterion = nn.CrossEntropyLoss()

    # model = RNN_Net(output_size, hidden_dim, n_layers, embedding_dim, batch_size, device, len_)
    # model = LSTM_Net(output_size, hidden_dim, n_layers, embedding_dim, batch_size, device, len_)
    # model = GRU_Net(output_size, hidden_dim, n_layers, embedding_dim, batch_size, device, len_)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_ = {'loss': [], 'acc': []}

    with trange(n_epoch) as t:
        for epoch in t:
            t.set_description('训练进度')
            sum_loss = 0
            acc = 0
            h = model.init_hidden(batch_size)
            for i, data in enumerate(train_loader):
                h = tuple([e.data for e in h])
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                # print(labels.shape)
                optimizer.zero_grad()
                outputs = model(inputs).to(device)
                # print(outputs.shape)
                # outputs, h = model(inputs, h)

                acc += (outputs.argmax(dim=1) == labels.argmax(dim=1)).sum().item()

                # print(outputs[0].shape)
                # print(labels.shape)
                loss = criterion(outputs, labels.float())

                loss.backward()
                optimizer.step()
                sum_loss += loss.item()
            t.set_postfix(loss=sum_loss / (i+1), acc=acc / (i+1))
            # t.set_postfix(acc=acc / (i+1))

            # print('Epoch: {}/{}.............'.format(epoch + 1, n_epoch), end=' ')
            # print("Loss: {:.4f}".format(sum_loss / (i+1)))
            loss_['loss'].append(sum_loss / (i+1))
            loss_['acc'].append(acc / (i+1))

    model.eval()
    criterion = nn.CrossEntropyLoss()
    sum_loss = 0
    acc = 0
    classify = {}

    for i, data in enumerate(test_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        # outputs, h = model(inputs, h)
        # print(labels)
        acc += (outputs.argmax(dim=1) == labels.argmax(dim=1)).sum().item()
        for j in range(outputs.shape[0]):
            if outputs[j].argmax(dim=0) == labels[j].argmax(dim=0):
                if senti_dict[int(labels[j].argmax(dim=0))] not in classify:
                    classify[senti_dict[int(labels[j].argmax(dim=0))]] = {senti_dict[int(outputs[j].argmax(dim=0))]: 1}
                else:
                    if senti_dict[int(outputs[j].argmax(dim=0))] in classify[senti_dict[int(labels[j].argmax(dim=0))]]:
                        classify[senti_dict[int(labels[j].argmax(dim=0))]][
                            senti_dict[int(outputs[j].argmax(dim=0))]] += 1
                    else:
                        classify[senti_dict[int(labels[j].argmax(dim=0))]][
                            senti_dict[int(outputs[j].argmax(dim=0))]] = 1
            else:
                if senti_dict[int(labels[j].argmax(dim=0))] not in classify:
                    classify[senti_dict[int(labels[j].argmax(dim=0))]] = {senti_dict[int(outputs[j].argmax(dim=0))]: 1}
                else:
                    if senti_dict[int(outputs[j].argmax(dim=0))] in classify[senti_dict[int(labels[j].argmax(dim=0))]]:
                        classify[senti_dict[int(labels[j].argmax(dim=0))]][
                            senti_dict[int(outputs[j].argmax(dim=0))]] += 1
                    else:
                        classify[senti_dict[int(labels[j].argmax(dim=0))]][
                            senti_dict[int(outputs[j].argmax(dim=0))]] = 1
        # print(accuracy_score(out,lab))
        # print(labels.shape)
        loss = criterion(outputs, labels.float())
        sum_loss += loss.item()

    print('模型测试准确率为：', acc / (i + 1))
    print("Loss: {:.4f}".format(sum_loss / (i + 1)))
    print('结果字典：', classify)

    torch.save({'model': model.state_dict()}, '../result/' + model_name + '.pth')
    pickle.dump(loss_, open('../result/' + model_name + '.loss', "wb"), protocol=4)
    pickle.dump(classify, open('../result/' + model_name + '.matrix', "wb"), protocol=4)
    # torch.save({'model': model.state_dict()}, '../result/' + model_name + '_b.pth')
    # pickle.dump(loss_, open('../result/' + model_name + '_b.loss', "wb"), protocol=4)
    # pickle.dump(classify, open('../result/' + model_name + '_b.matrix', "wb"), protocol=4)

