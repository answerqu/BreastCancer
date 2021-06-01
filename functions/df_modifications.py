import pandas as pd
import numpy as np
import copy
from functions.functions import make_folds
from functions.hyperoptimisation import hp_model
from functions.preprocessing import filt_features
from sklearn.cluster import MiniBatchKMeans as mbkm
from random import shuffle
import random
import itertools
from tqdm import tqdm_notebook as tqdm


def symmetrization(df, two_classes=True):
    """
    Раздувает выборку симметричными данными.
    
    Input:
        df (pd.DataFrame) - полная таблица данных по больным
        two_classes (bool) - Если True, удваивается количество обоих классов, иначе тлько больных
        
    Output:
        df (pd.DataFrame) - увеличенная выборка
    """
    if two_classes:
        df1 = df.copy()
    else:
        df1 = df[df.pathology == 1].copy()
    cols = [col for col in df1.columns if col.find('left') != -1 or col.find('right') != -1]
    for i in range(len(cols)):
        if cols[i].find('right') != -1:
            df1.loc[:, cols[i - 1]], df1.loc[:, cols[i]] = df1.loc[:, cols[i]].copy(), df1.loc[:, cols[i - 1]].copy()
    df1.index = [ind + '_symm' for ind in df1.index]
    return pd.concat([df, df1])


def split_areas(df):
    df_new = pd.DataFrame()

    for loc, side in enumerate(['right', 'left']):
        cols = [col for col in df.columns if col.find(side) != -1]
        df_tmp = df.loc[:, cols]
        df_tmp.columns = [col.replace(f'__{side}__v1', '') for col in cols]
        df_tmp['pathology'] = df.pathology
        df_tmp['id'] = df.id
        df_tmp.index = df_tmp.index.map(lambda x: x + f'__{side}')
        df_new = pd.concat([df_new, df_tmp], axis=0)
    return df_new


def downsampling(X_train, y_train, type_of_downsampling='rand', id_column=None):
    """
    Снижение количества здоровых людей в выборке до количества больных.
    
    Input:
        X_train (pd.DataFrame) - обучающая выборка
        y_train (pd.Series) - целевая перменная обучающей выборки
        
    Output:
        X_train_dwn, y_train_dwn
    """

    if type_of_downsampling == 'rand':
        X_train_pos = X_train[y_train == 1]
        X_train_neg = X_train[y_train == 0]
        id = id_column[X_train_neg.index]
        set_id = list(id.unique())
        shuffle(set_id)
        id_number = 1
        while X_train_pos.shape[0] < X_train_neg.loc[id_column[X_train_neg.index].map(lambda x:
                                                                                      x not in set_id[
                                                                                               :id_number]).values,
                                     :].shape[0]:
            id_number += 1
        neg_under = X_train_neg.loc[id_column[X_train_neg.index].map(lambda x: x not in set_id[:id_number]).values, :]

        X_train_dwn = pd.concat([neg_under, X_train_pos], axis=0).sample(frac=1)
        y_train_dwn = y_train[X_train_dwn.index]
    if type_of_downsampling == 'outliers':
        # данным способом убирается слишком много сэмплов

        X_neg = X_train[y_train == 0]

        Q1 = X_neg.quantile(0.25)
        Q3 = X_neg.quantile(0.75)
        low = Q1 - 1.5 * (Q3 - Q1)
        high = Q3 + 1.5 * (Q3 - Q1)

        # low = X_neg.quantile(0.001)
        # high = X_neg.quantile(0.999)
        res = []
        for sample in X_neg.index:
            res.append(X_train.shape[1] - ((X_neg.loc[sample, :] > low) & (X_neg.loc[sample, :] < high)).sum())
        a = {X_neg.index[i]: res[i] for i in range(len(res))}
        a = sorted(a.items(), key=lambda x: x[1], reverse=True)
        ind = [a[i][0] for i in range(X_neg.shape[0] - y_train.sum())]
        X_train_dwn = X_train.drop(ind)
        y_train_dwn = y_train.drop(ind)

    if type_of_downsampling == 'proba':
        print('PROBA DOWNSAMPLING: ')
        y_predict = pd.Series(np.zeros(len(y_train.index)), index=y_train.index)
        X_fold, y_fold = make_folds(X_train, y_train, id_column=id_column, n=3)
        for i in range(len(X_fold)):
            X_train_curr, y_train_curr = pd.concat([X_fold[k] for k in range(len(X_fold)) if k != i]), pd.concat(
                [y_fold[k] for k in range(len(X_fold)) if k != i])
            X_test_curr, y_test_curr = X_fold[i].copy(), y_fold[i].copy()

            svm = hp_model('xgb', X_train_curr, y_train_curr, evals=15, id_column=id_column, class_w='balanced')
            y_predict[X_test_curr.index] = np.transpose(svm.predict_proba(X_test_curr))[0]

        ind = y_predict[y_train == 0].sort_values().index[:X_train[y_train == 0].shape[0] - int(y_train.sum())]

        print('PROBA DOWNSAMPLING IS DONE')
        X_train_dwn = X_train.drop(ind)
        y_train_dwn = y_train.drop(ind)
    return X_train_dwn, y_train_dwn


def categorize(X, bins=100, n_top=3):
    category_list = []
    X = copy.deepcopy(X)
    for i in range(X.shape[1]):
        a = pd.cut(X.iloc[:, i], bins=bins).map(lambda x: x.mid)

        maxes = a.value_counts().sort_values(ascending=False)[:n_top]
        if maxes.mean() > X.shape[0] // 10:
            X.iloc[:, i] = a.astype('float32')
            category_list.append(X.columns[i])
    

    X_new = copy.deepcopy(X[category_list])
    category_list = []
    for i in X_new.columns:
        a = pd.cut(X_new[i], bins=bins // 10).map(lambda x: x.mid)

        maxes = a.value_counts().sort_values(ascending=False)[:n_top]
        if maxes.mean() > X.shape[0] // 10:
            X[i] = a.astype('float32')
            category_list.append(i)
    return X, category_list


def aggregation(X_train, X_test=None, cat_features=[], num_features=[]):
    if sum([col.find('left') != -1 for col in X_train.columns]):
        for i in cat_features:
            for j in num_features:
                num_side = j.split('__')[1]
                cat_side = i.split('__')[1]
                feat1 = j.split('__')[0]
                feat2 = i.split('__')[0]
                if cat_side in ['left', 'right'] and num_side in ['left', 'right'] and cat_side == num_side:
                    name = f'mean__{num_side}__{feat1}__{feat2}'
                    X_train[name] = X_train[i].map(X_train.groupby(i)[j].mean())
                    X_test[name] = X_test[i].map(X_train.groupby(i)[j].mean())
                    name = f'max__{num_side}__{feat1}__{feat2}'
                    X_train[name] = X_train[i].map(X_train.groupby(i)[j].max())
                    X_test[name] = X_test[i].map(X_train.groupby(i)[j].max())
                    name = f'std__{num_side}__{feat1}__{feat2}'
                    X_train[name] = X_train[i].map(X_train.groupby(i)[j].std())
                    X_test[name] = X_test[i].map(X_train.groupby(i)[j].std())
                    name = f'min__{num_side}__{feat1}__{feat2}'
                    X_train[name] = X_train[i].map(X_train.groupby(i)[j].min())
                    X_test[name] = X_test[i].map(X_train.groupby(i)[j].min())
                if cat_side not in ['left', 'right'] and num_side in ['left', 'right']:
                    name = f'mean__{num_side}__{feat1}__{feat2}'
                    X_train[name] = X_train[i].map(X_train.groupby(i)[j].mean())
                    X_test[name] = X_test[i].map(X_train.groupby(i)[j].mean())
                    name = f'max__{num_side}__{feat1}__{feat2}'
                    X_train[name] = X_train[i].map(X_train.groupby(i)[j].max())
                    X_test[name] = X_test[i].map(X_train.groupby(i)[j].max())
                    name = f'std__{num_side}__{feat1}__{feat2}'
                    X_train[name] = X_train[i].map(X_train.groupby(i)[j].std())
                    X_test[name] = X_test[i].map(X_train.groupby(i)[j].std())
                    name = f'min__{num_side}__{feat1}__{feat2}'
                    X_train[name] = X_train[i].map(X_train.groupby(i)[j].min())
                    X_test[name] = X_test[i].map(X_train.groupby(i)[j].min())
                if cat_side not in ['left', 'right'] and num_side not in ['left', 'right']:
                    name = f'mean__{num_side}__{feat1}__{feat2}'
                    X_train[name] = X_train[i].map(X_train.groupby(i)[j].mean())
                    X_test[name] = X_test[i].map(X_train.groupby(i)[j].mean())
                    name = f'max__{num_side}__{feat1}__{feat2}'
                    X_train[name] = X_train[i].map(X_train.groupby(i)[j].max())
                    X_test[name] = X_test[i].map(X_train.groupby(i)[j].max())
                    name = f'std__{num_side}__{feat1}__{feat2}'
                    X_train[name] = X_train[i].map(X_train.groupby(i)[j].std())
                    X_test[name] = X_test[i].map(X_train.groupby(i)[j].std())
                    name = f'min__{num_side}__{feat1}__{feat2}'
                    X_train[name] = X_train[i].map(X_train.groupby(i)[j].min())
                    X_test[name] = X_test[i].map(X_train.groupby(i)[j].min())
    else:
        for i in cat_features:
            for j in num_features:
                name = f'mean__{i}__{j}'
                X_train[name] = X_train[i].map(X_train.groupby(i)[j].mean())
                if X_test is not None:
                    X_test[name] = X_test[i].map(X_train.groupby(i)[j].mean())
                name = f'max__{i}__{j}'
                X_train[name] = X_train[i].map(X_train.groupby(i)[j].max())
                if X_test is not None:
                    X_test[name] = X_test[i].map(X_train.groupby(i)[j].max())
                name = f'min__{i}__{j}'
                X_train[name] = X_train[i].map(X_train.groupby(i)[j].min())
                if X_test is not None:
                    X_test[name] = X_test[i].map(X_train.groupby(i)[j].min())
                name = f'std__{i}__{j}'
                X_train[name] = X_train[i].map(X_train.groupby(i)[j].std())
                if X_test is not None:
                    X_test[name] = X_test[i].map(X_train.groupby(i)[j].std())

    if X_test is not None:
        return X_train, X_test
    else:
        return X_train


def generation(X_train, y_train=None, X_test=None, y_test=None, X_val=None, y_val=None, num_features=[], group_min=3,
               group_max=6, clusters_min=2, clusters_max=5, clusters_step=1, features_frac=0.3, id_column=None,
               cv=False):
    if sum([col.find('left') != -1 for col in X_train.columns]):
        lefties = sorted([col for col in num_features if col.find('right') == -1])
        shuffle(lefties)
        for k in range(group_min, group_max):
            for i in range(k, len(lefties), k):
                features = lefties[i - k: i]
                for j in range(clusters_min, clusters_max, clusters_step):
                    name = f'kmeans_mb__left__{i}__{k}__{j}'
                    mb = mbkm(n_clusters=j)
                    mb.fit(X_train[features])
                    X_train[name] = mb.labels_
                    if X_test is not None:
                        X_test[name] = mb.predict(X_test[features])
                    if X_val is not None:
                        X_val[name] = mb.predict(X_val[features])
                    name = f'kmeans_mb__right__{i}__{k}__{j}'
                    ind = X_train.index.map(lambda x: x.replace('_symm', '') if x.find('symm') != -1 else x + '_symm')
                    X_train[name] = mb.predict(X_train.loc[ind, features])
                    if X_test is not None:
                        X_test[name] = mb.predict(X_test[features])
                    if X_val is not None:
                        ind = X_val.index.map(lambda x: x.replace('_symm', '') if x.find('symm') != -1 else x + '_symm')
                        X_val[name] = mb.predict(X_val.loc[ind, features])
    else:
        num_features = list(num_features)
        if cv:
            X_fold, y_fold = make_folds(X_train, y_train, id_column=id_column, n=3)
            for j in range(len(X_fold)):
                X_train_tmp, y_train_tmp = (pd.concat([X_fold[k] for k in range(len(X_fold)) if k != j]),
                                    pd.concat([y_fold[k] for k in range(len(X_fold)) if k != j]))
                X_test_tmp, y_test_tmp = X_fold[j].copy(), y_fold[j].copy()

                for group_size in tqdm(range(group_min, group_max)):
                    combinations = [a for a in list(enumerate(itertools.combinations(num_features, group_size)))
                                    if random.uniform(0, 1) < features_frac]
                    for i, features in combinations:
                        features = list(features)
                        for n_clusters in range(clusters_min, clusters_max, clusters_step):
                            name = f'kmeans__{i}__{group_size}__{n_clusters}'
                            X_train[name] = 0
                            mb = mbkm(n_clusters=n_clusters)
                            mb.fit(X_train_tmp[features])
                            X_train.loc[X_test_tmp.index, name] += mb.predict(X_test_tmp[features])
        else:
            for group_size in tqdm(range(group_min, group_max)):
                combinations = [a for a in list(enumerate(itertools.combinations(num_features, group_size)))
                                if random.uniform(0, 1) < features_frac]
                for i, features in combinations:
                    features = list(features)
                    for n_clusters in range(clusters_min, clusters_max, clusters_step):
                        name = f'kmeans__{i}__{group_size}__{n_clusters}'
                        X_train[name] = 0
                        mb = mbkm(n_clusters=n_clusters)
                        mb.fit(X_train[features])
                        X_train[name] += mb.predict(X_train[features])
                        try:
                            X_test[name] = 0
                            X_test[name] += mb.predict(X_test[features])
                        except:
                            pass
                        try:
                            X_val[name] = 0
                            X_val[name] += mb.predict(X_val[features])
                        except:
                            pass
    if X_test is not None:
        if X_val is not None:
            return X_train, X_test, X_val
        return X_val
    return X_train
