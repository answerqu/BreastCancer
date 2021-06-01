import pandas as pd
import numpy as np
import random
from sklearn.metrics import f1_score, confusion_matrix, recall_score, roc_auc_score, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.utils import class_weight
from functions.hyperoptimisation import hp_model
from copy import deepcopy


def split(X, y, test_size=0.27, id_column=None):
    """
    Аналог функции train_test_split из библиотеки sklearn. В данной реализации учитываются id пациентов.
    
    Input:
        X (pd.DataFrame) - датасет 
        y (pd.Series) - целевая переменная
        test_size (float) - процентное соотношение тестовой выборки к передаваемой
    
    Output:
        data_train, data_test, y_train, y_test
    """
    data_for_change = X.copy()
    data_for_change['id'] = id_column
    data_for_change = pd.concat([data_for_change, y], axis=1)
    data_train = pd.DataFrame([])
    data_test = pd.DataFrame([])
    p_gen = y.value_counts(normalize=True)[1]
    while data_test.shape[0] / X.shape[0] < test_size:
        data_for_leave = data_for_change.copy()
        p = data_for_change.pathology.value_counts(normalize=True)[1]
        if p > p_gen and (data_for_change[data_for_change.pathology == 1].shape[0] != 0):
            data_for_change = data_for_change[data_for_change.pathology == 1]
        else:
            data_for_change = data_for_change[data_for_change.pathology == 0]
        rand_id = random.choice(list(data_for_change.id))
        data_for_concat = data_for_change[data_for_change.id == rand_id]
        data_test = pd.concat([data_test, data_for_concat])
        data_for_leave.drop(data_for_concat.index, axis=0, inplace=True)
        data_for_change = data_for_leave.copy()
    data_train = data_for_change
    y_train = data_train.pathology
    y_test = data_test.pathology
    data_train.drop(['id', 'pathology'], axis=1, inplace=True)
    data_test.drop(['id', 'pathology'], axis=1, inplace=True)

    # ind = [i for i in data_test.index if i.find('symm') == -1]
    # data_test = data_test.loc[ind,:]
    # y_test = y_test[ind]
    return data_train, data_test, y_train, y_test


def make_folds(X, y, id_column, n=5):
    """
    """
    n -= 1
    X_fold = []
    y_fold = []
    X_train = X.copy()
    y_train = y.copy()
    for i in range(n):
        X_train, X_test, y_train, y_test = split(X_train, y_train, test_size=1 / (n + 1 - i), id_column=id_column)
        X_fold.append(X_test)
        y_fold.append(y_test)
    X_fold.append(X_train)
    y_fold.append(y_train)
    return X_fold, y_fold


def cross_val_split_score(model, X, y, metric, vf=False, id_column=None, oversampling=False, class_w=None, n_folds=6,
                          print_scores=False):
    """
    Аналог cross_val_score из библиотеки sklearn. В данной реализации учитываются id пациентов.
    
    Input:
        clf (ML classifier) - алгоритм бинарной классификации, по которому будем происходить подсчет метрики на кросс-валдиации
        X (pd.DataFrame) - датасет 
        y (pd.Series) - целевая переменная
        vf (bool) - булева переменная, необходимая для параметра validate_features в функции predict у некоторых алгоритмов
        id_column (pd.Series) - серия id по снимкам, необходимо передавать всегда
        oversampling (bool) - если True, синтетически увеличивается количество больных людей до количества здоровых на каждом фолде на основе функции SMOTE() библиотеки imblearn
        class_w (string or None): Если 'balanced', то соответствующая строка передается как параметр class_weight в алгоритмы МЛ, иначе None.
        
    Output:
        np.mean(res), np.max(res)-np.min(res), np.min(res)
            res - массив трех значений метрики по кросс-валидации (3-х фолдовой)
    """
    res = []
    # X1,X2,y1,y2 = split(X,y[X.index],test_size=0.33,id_column=id_column[X.index])
    # X1,X3,y1,y3 = split(X1,y1[X1.index],test_size=0.5,id_column=id_column[X1.index])

    X_fold, y_fold = make_folds(X, y, id_column, n=n_folds)
    model_tmp = deepcopy(model)
    for i in range(len(X_fold)):
        model = deepcopy(model_tmp)
        X_train, y_train = pd.concat([X_fold[k] for k in range(len(X_fold)) if k != i]), pd.concat(
            [y_fold[k] for k in range(len(X_fold)) if k != i])

        X_test, y_test = X_fold[i], y_fold[i]
        #ind = [i for i in X_test.index if i.find('symm') == -1]
        #X_test = X_test.loc[ind, :]
        #y_test = y_test[ind]

        # print(y_test.sum())
        if oversampling:
            sm = SMOTE()
            X_train, y_train = sm.fit_sample(X_train, y_train)
        if class_w is not None:
            class_weights = list(class_weight.compute_class_weight(class_w,
                                                                   np.unique(y_train),
                                                                   y_train))
            w_array = np.ones(y_train.shape[0], dtype='float')
            for i, val in enumerate(y_train):
                w_array[i] = class_weights[val - 1]
            model.fit(X_train, y_train, sample_weight=w_array)
        else:
            model.fit(X_train, y_train)
        if metric.__name__ == 'roc_auc_score':
            res.append(metric(y_test, model.predict_proba(X_test).T[1]))
        else:
            res.append(metric(y_test, model.predict(X_test)))

    res = np.array(res)
    if print_scores:
        print(res)
    if np.std(res) != 0:
        # print(res)
        return np.mean(res), np.max(res) - np.min(res), np.min(res)
    else:
        return 0., 0., 0.


def run_hp(X_train, y_train, X_test, y_test, id_column, metric=f1_score, oversampling=False, class_w=None, evals=50,
           n_folds=3, thr_diff=0.15, thr_min=0.5, print_scores=False):
    """
    Функция запускает автоматически оптимизацию 5-ти алгоритмов МЛ: логистическая регрессия, метод опорных векторов, случайный лес, XGBClassifier, LGBClassifier. \
После вычисления лучших параметров обучается модель на всей обучающей выборке и вычисляются результаты 4-х метрик (f1, sensivity, specificity,auc) на тестовой выборке.

    Input:
        X_train (pd.DataFrame) - обучающая выборка
        y_train (pd.Series) - целевая перменная обучающей выборки
        X_test (pd.DataFrame) - тестовая выборка
        y_test (pd.Series) - целевая перменная тестовой выборки
        id_column (pd.Series) - серия id по снимкам, необходимо передавать всегда
        oversampling (bool) - если True, синтетически увеличивается количество больных людей до количества здоровых на каждом фолде на основе функции SMOTE() библиотеки imblearn
        class_w (string or None): Если 'balanced', то соответствующая строка передается как параметр class_weight в алгоритмы МЛ, иначе None.
        evals (int) - количество итераций оптимизации. Вычисление проходит в следующей пропорции: 
            LogReg: 0.75*evals
            SVM: 1.*evals
            RF: 0.5*evals
            XGB: 1.*evals
            LGB: 1.*evals
    Output:
        res (pd.DataFrame) - таблица с результатами по тестовой выборке, где по строкам идут алгоритмы МЛ, по столбцам - метрики качества классификации.
    """
    lr = hp_model('lr', X_train, y_train, id_column=id_column[X_train.index], metric=metric, evals=evals,
                  oversampling=oversampling, class_w=class_w, n_folds=n_folds,
                  thr_diff=thr_diff, thr_min=thr_min, print_scores=print_scores)
    svm = hp_model('svm', X_train, y_train, id_column=id_column[X_train.index], metric=metric, evals=evals,
                   oversampling=oversampling, class_w=class_w, n_folds=n_folds,
                   thr_diff=thr_diff, thr_min=thr_min, print_scores=print_scores)
    rf = hp_model('rf', X_train, y_train, id_column=id_column[X_train.index], metric=metric, evals=evals,
                  oversampling=oversampling, class_w=class_w, n_folds=n_folds,
                  thr_diff=thr_diff, thr_min=thr_min, print_scores=print_scores)
    xgb = hp_model('xgb', X_train, y_train, id_column=id_column[X_train.index], metric=metric, evals=evals,
                   oversampling=oversampling, class_w=class_w, n_folds=n_folds,
                   thr_diff=thr_diff, thr_min=thr_min, print_scores=print_scores)
    lgb = hp_model('lgb', X_train, y_train, id_column=id_column[X_train.index], metric=metric, evals=evals,
                   oversampling=oversampling, class_w=class_w, n_folds=n_folds,
                   thr_diff=thr_diff, thr_min=thr_min, print_scores=print_scores)

    res = []
    res_predict = []
    models = []
    for clf in [lr, z, rf, xgb, lgb]:
        conf = confusion_matrix(y_test, clf.predict(X_test))
        a = {
            'f1': f1_score(y_test, clf.predict(X_test)),
            'sensivity': recall_score(y_test, clf.predict(X_test)),
            'specificity': conf[0, 0] / (conf[0, 0] + conf[0, 1]),
            'auc': roc_auc_score(y_test, clf.predict(X_test)),
            'accuracy': accuracy_score(y_test, clf.predict(X_test))
        }
        res.append(a)
        res_predict.append(clf.predict(X_test))
        models.append(clf)

    return pd.DataFrame(res, index=['lr', 'svm', 'rf', 'xgb', 'lgb']), res_predict, models
