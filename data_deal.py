import pandas as pd
from mxnet import nd


def load(str):
    train_data = pd.read_csv(str + '//train.csv')
    test_data = pd.read_csv(str + '//test.csv')
    all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
    return train_data, test_data, all_features


def load_data(str):
    train_data, test_data, all_features = load(str)
    num_features = all_features.dtypes[all_features.dtypes != 'object'].index
    all_features[num_features] = all_features[num_features].apply(lambda x: (x - x.mean()) / (x.std()))
    all_features[num_features] = all_features[num_features].fillna(0)
    all_features_last = pd.get_dummies(all_features, dummy_na=True)
    m = train_data.shape[0]
    train_features = nd.array(all_features_last[:m].values)
    train_labels = nd.array(train_data.SalePrice.values).reshape((-1,1))
    test_features = nd.array(all_features_last[m:].values)

    return train_features, test_features, train_labels, test_data
