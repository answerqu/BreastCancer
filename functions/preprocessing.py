import random
import pandas as pd
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler


def normalization(X_train, y_train=None, X_test=None, y_test=None, test_non_symm=True):
    scaler = MinMaxScaler()
    X_train = deepcopy(X_train)
    if X_test is not None and y_test is not None and y_train is not None:
        X_train_norm = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
        if test_non_symm:
            X_test = X_test.loc[[ind for ind in X_test.index if ind.find('symm') == -1], :]
            y_test = y_test[X_test.index]
        X_test_norm = deepcopy(X_test)
        X_test_norm.loc[:, :] = scaler.transform(X_test)

        return X_train_norm, y_train, X_test_norm, y_test, scaler
    else:
        X_train_norm = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
        return X_train_norm, scaler


def variance(X):
    """
    """
    res = {}
    X_norm, scaler = normalization(X)
    for a in X_norm.columns:
        std = X_norm.loc[:, a].std()
        res[a] = std
    series = pd.Series(res)
    q = series.quantile(1 - 0.9544)
    series = series[series > series.quantile(1 - 0.9544)]

    return X.loc[:, series.index], q


def outliers(X):
    """
    """
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    low = Q1 - 1.5 * (Q3 - Q1)
    high = Q3 + 1.5 * (Q3 - Q1)

    res = {}
    for a in X.columns:
        sum = X.loc[:, a].apply(lambda x: 1 if x > high[a] or x < low[a] else 0).sum()
        res[a] = sum
    series = pd.Series(res)
    q = series.quantile(0.9544)
    series = series[series < series.quantile(0.9544)]

    return X.loc[:, series.index], q


def correlation(X, thr=0.95):
    """
    Производится отбор тех признаков, корелляция которых меньше определенного порога.
    
    Input:
        X_train (pd.DataFrame) - обучающая выборка
        y_train (pd.Series) - целевая перменная обучающей выборки
        thr (float) - модуль по порогу
    
    Output:
        X_train_dwn, X_test_dwn
    """
    corr = X.corr()
    cols = corr.columns.to_list()
    random.shuffle(cols)
    for a in cols:
        series = corr[a][cols]
        for b in series.index:
            if abs(series[b]) > thr and a != b:
                cols.remove(b)

    return X[cols]


def filt_features(X_train, X_test=None):
    cols_old = X_train.columns.tolist()

    new_cols = []
    for i in range(1, len(cols_old)):
        spl_0 = cols_old[i - 1].split('__')
        spl_1 = cols_old[i].split('__')
        if spl_0[0] == spl_1[0]:
            if spl_0[1] == 'left' and spl_1[1] == 'right':
                if len(spl_0) == 3:
                    new_cols.append(cols_old[i - 1])
                    new_cols.append(cols_old[i])
                if len(spl_0) == 4:
                    if spl_0[3] == spl_1[3]:
                        new_cols.append(cols_old[i - 1])
                        new_cols.append(cols_old[i])
                if len(spl_0) == 5:
                    if spl_0[3] == spl_1[3] and spl_0[4] == spl_1[4]:
                        new_cols.append(cols_old[i - 1])
                        new_cols.append(cols_old[i])
        if spl_1[1] == 'v1':
            new_cols.append(cols_old[i])
    if X_test is not None:
        return X_train.loc[:, new_cols], X_test.loc[:, new_cols]
    else:
        return X_train.loc[:, new_cols]
