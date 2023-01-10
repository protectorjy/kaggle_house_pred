from data_deal import *
from dev import *
from mxnet import gluon, nd, autograd
from mxnet.gluon import loss as gloss, data as gdata
import d2lzh as d2l

train_features, test_features, train_labels, test_data = load_data(
    "E:\\test_ai\\kaggle_house_pred\house-prices-advanced-regression-techniques")

loss = gloss.L2Loss()


def error(net, features, labels):
    clipped_preds = nd.clip(net(features), 1, float('inf'))
    rmse = nd.sqrt(2 * loss(clipped_preds.log(), labels.log()).mean())
    return rmse.asscalar()


def train(net, train_features, train_labels, test_features, test_labels, num_epochs, batch_size, lr, wd):
    train_l = []
    test_l = []
    trainer_w = gluon.Trainer(net.collect_params('.*weight'), 'adam', {'learning_rate': lr, 'wd': wd})
    trainer_b = gluon.Trainer(net.collect_params('.*bias'), 'adam', {'learning_rate': lr})
    train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features, train_labels), batch_size, shuffle=True)
    for epoch in range(num_epochs):
        for x, y in train_iter:
            with autograd.record():
                l = loss(net(x), y)
            l.backward()
            trainer_w.step(batch_size)
            trainer_b.step(batch_size)
        train_l.append(error(net, train_features, train_labels))
        if test_labels is not None:
            test_l.append(error(net, test_features, test_labels))
        print("epoch %d      train error:%f" % (epoch + 1, train_l[-1]))
    return train_l, test_l


k, num_epochs, lr, wd, batch_size = 5, 100, 3.5, 0, 10
train_model = train
train_l, dev_l = k_fold(k, train_model, train_features, train_labels, num_epochs, batch_size, lr, wd)
print('fold_num:%d   avg train rmse:%f   avg test rmse:%f' % (k, train_l / k, dev_l / k))


while (1):
    n = int(input("是否生成测试csv：（1是，2否）\n"))
    if n == 1:
        train_and_pred(train_model, train_features, test_features, train_labels, test_data, num_epochs, lr, wd,
                       batch_size)
        break
    elif n==2:
        print("未生成\n")
        break
    else:
        print("输入错误，请从新输入：")
