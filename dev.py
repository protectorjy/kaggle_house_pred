import pandas as pd
from mxnet import nd
import d2lzh as d2l
import matplotlib.pyplot as plt
from mxnet.gluon import nn
from mxnet import init


def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=1))
    return net


def get_i_data(k, i, x, y):
    assert k > 1
    size = x.shape[0] // k
    train_x, train_y = None, None
    for j in range(k):
        index = slice(j * size, (j + 1) * size)
        x_part, y_part = x[index, :], y[index]
        if j == i:
            x_dev, y_dev = x_part, y_part
        elif train_x is None:
            train_x, train_y = x_part, y_part
        else:
            train_x = nd.concat(train_x, x_part, dim=0)
            train_y = nd.concat(train_y, y_part, dim=0)
    return train_x, train_y, x_dev, y_dev


def k_fold(k, train, train_features, train_labels, num_epochs, batch_size, lr, wd):
    train_l_sum, dev_l_sum = 0.0, 0.0
    for i in range(k):
        net = get_net()
        data = get_i_data(k, i, train_features, train_labels)
        train_l, dev_l = train(net, *data, num_epochs, batch_size, lr, wd)
        train_l_sum += train_l[-1]
        dev_l_sum += dev_l[-1]
        if i == k:
            d2l.semilogy(range(1, num_epochs + 1), train_l, 'epochs', 'errors', range(1, num_epochs + 1), dev_l,
                         ['train', 'dev'])
            plt.show()
        print("fold %d     train rmse:%f   dev rmse:%f" % (i + 1, train_l[-1], dev_l[-1]))
    return train_l_sum, dev_l_sum


def train_and_pred(train,train_features,test_features,train_labels,test_data,num_epochs,lr,wd,batch_size):
    net = get_net()
    train_l , _ = train(net,train_features,train_labels,None,None,num_epochs,batch_size,lr
                        ,wd)
    print('train rmse: %f'%(train_l[-1]))
    d2l.semilogy(range(1,num_epochs+1),train_l,'epoch','rmse')
    preds = net(test_features).asnumpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1,-1)[0])
    submission = pd.concat([test_data['Id'],test_data['SalePrice']],axis = 1)
    submission.to_csv('submission.csv' , index=False)
