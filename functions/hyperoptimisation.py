import sklearn
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from hyperopt import hp, tpe
from hyperopt.fmin import fmin
import numpy as np
import functions
from imblearn.over_sampling import SMOTE
from sklearn.utils import class_weight
from tqdm import tqdm


def hp_model(model, X, y, evals=100, max_iterations=500, metric=sklearn.metrics.f1_score, dict_concat={},
             id_column=None, oversampling=False, random_state=42, class_w=None, n_folds=3, thr_diff=0.15, thr_min=0.5,
             print_scores=False):
    """
    Функция вычисляет некоторый локальный максимум определенной метрики классификации, вычисляемой как среднее значение \
результата кросс-валидации и обучения некоторого алгоритма МЛ в пространстве гиперпараметров.
    
    Input:
        model (str): передается сокращенное название алгоритма, который нас интересует
            'lr': LogisticRegression
            'svm': SVC
            'rf': RandomForestClassifier
            'xgb': XGBClassifier
            'lgb': LGBClassifier
            'ctb': CatBoostClassifier
        X (pd.DataFrame) - датасет 
        y (pd.Series) - целевая переменная 
        evals (int) - количество итераций оптимизации
        max_iterations (int) - гиперпараметр iterations, n_iter и т.д., передаваемый в бустинговые модели и в случайный лес
        metric (sklearn.metrics.metric) - метрика, по которой будет идти минимизация метрики со знаком минус и поиск оптимальных гиперпараметров
        dict_concat (dictionary) - словарь, в который передается определенный набор гиперпараметров и их значений. Необходим, если нам нужно зафиксировать определенный набор гиперпараметров
        id_column (pd.Series) - серия id по снимкам, необходимо передавать всегда
        oversampling (bool) - если True, синтетически увеличивается количество больных людей до количества здоровых на каждом фолде на основе функции SMOTE() библиотеки imblearn
        class_w (string or None): Если 'balanced', то соответствующая строка передается как параметр class_weight в алгоритмы МЛ, иначе None.
        
    Output:
        res (ML model) - модель, обученная по всей переданной выборке с найденными оптимальными за evals итераций гиперпараметрами. Использовать для предсказания на тестовой выборке.
    """

    class_weights = list(class_weight.compute_class_weight(class_w,
                                                           np.unique(y),
                                                           y))
    w_array = np.ones(y.shape[0], dtype='float')

    for i, val in enumerate(y):
        w_array[i] = class_weights[val - 1]

    def hyperopt_lr_score(params):
        clf = LogisticRegression(**params)
        result = functions.functions.cross_val_split_score(clf, X, y, metric, vf=True, id_column=id_column,
                                                 oversampling=oversampling, n_folds=n_folds, print_scores=print_scores)
        if result[1] < thr_diff and result[1] != 0 and result[2] > thr_min:
            return -result[0]
        else:
            return 0.

    def hyperopt_svm_score(params):
        clf = SVC(**params)
        result = functions.functions.cross_val_split_score(clf, X, y, metric, vf=True, id_column=id_column,
                                                 oversampling=oversampling, n_folds=n_folds, print_scores=print_scores)
        if result[1] < thr_diff and result[1] != 0 and result[2] > thr_min:
            return -result[0]
        else:
            return 0.

    def hyperopt_rf_score(params):
        clf = RandomForestClassifier(**params)
        result = functions.functions.cross_val_split_score(clf, X, y, metric, vf=True, id_column=id_column,
                                                 oversampling=oversampling, n_folds=n_folds, print_scores=print_scores)
        if result[1] < thr_diff and result[1] != 0 and result[2] > thr_min:
            return -result[0]
        else:
            return 0.

    def hyperopt_xgb_score(params):
        clf = XGBClassifier(**params)

        result = functions.functions.cross_val_split_score(clf, X, y, metric, id_column=id_column, oversampling=oversampling,
                                                 class_w=class_w, n_folds=n_folds, print_scores=print_scores)
        if result[1] < thr_diff and result[1] != 0 and result[2] > thr_min:
            return -result[0]
        else:
            return 0.

    def hyperopt_lgb_score(params):
        clf = LGBMClassifier(**params)

        result = functions.functions.cross_val_split_score(clf, X, y, metric, id_column=id_column, oversampling=oversampling,
                                                 class_w=class_w, n_folds=n_folds, print_scores=print_scores)
        if result[1] < thr_diff and result[1] != 0 and result[2] > thr_min:
            return -result[0]
        else:
            return 0.

    def hyperopt_ctb_score(params):
        clf = CatBoostClassifier(**params)

        result = functions.functions.cross_val_split_score(clf, X, y, metric, vf=True, id_column=id_column,
                                                 oversampling=oversampling, class_w=class_w, n_folds=n_folds,
                                                 print_scores=print_scores)
        if result[1] < thr_diff and result[1] != 0 and result[2] > thr_min:
            return -result[0]
        else:
            return 0.

    if model == 'lr':
        space = {
            'C': hp.loguniform('C', -2, 4),
            'penalty': hp.choice('penalty', ['l1', 'l2']),
            'solver': 'liblinear',
            'random_state': random_state,
            'class_weight': class_w,
            'n_jobs': -1,
        }
        space.update(dict_concat)
        best = fmin(fn=hyperopt_lr_score, space=space, algo=tpe.suggest, max_evals=evals, show_progressbar=False)
        best['penalty'] = ['l1', 'l2'][best['penalty']]
        best['solver'] = 'liblinear'
        best.update(dict_concat)
        res = LogisticRegression(**best)
        if oversampling:
            sm = SMOTE()
            X_ovr, y_ovr = sm.fit_sample(X, y)
            res.fit(X_ovr, y_ovr)
        else:
            res.fit(X, y)

    if model == 'svm':
        space = {
            'C': hp.loguniform('C', -2, 4),
            'gamma': hp.loguniform('gamma', -5, 1),
            'kernel': 'rbf',
            'random_state': random_state,
            'class_weight': class_w,
        }
        space.update(dict_concat)
        best = fmin(fn=hyperopt_svm_score, space=space, algo=tpe.suggest, max_evals=evals, show_progressbar=False)
        best.update(dict_concat)
        res = SVC(**best)
        if oversampling:
            sm = SMOTE()
            X_ovr, y_ovr = sm.fit_sample(X, y)
            res.fit(X_ovr, y_ovr)
        else:
            res.fit(X, y)

    if model == 'rf':
        space = {
            'bootstrap': hp.choice('bootstrap', [True, False]),
            'max_depth': hp.choice('max_depth', np.arange(5, 50, 1)),
            'max_features': hp.choice('max_features', ['auto', 'sqrt']),
            'min_samples_leaf': hp.choice('min_samples_leaf', np.arange(1, 5, 1)),
            'min_samples_split': hp.choice('min_samples_split', np.arange(2, 10, 2)),
            'n_estimators': hp.choice('n_estimators', np.arange(10, max_iterations, 10)),
            'verbose': 0,
            'random_state': random_state,
            'class_weight': class_w,
            'n_jobs': -1,
        }
        space.update(dict_concat)
        best = fmin(fn=hyperopt_rf_score, space=space, algo=tpe.suggest, max_evals=evals, show_progressbar=False)
        best['bootstrap'] = [True, False][best['bootstrap']]
        best['max_depth'] = np.arange(5, 50, 1)[best['max_depth']]
        best['max_features'] = ['auto', 'sqrt'][best['max_features']]
        best['min_samples_leaf'] = np.arange(1, 5, 1)[best['min_samples_leaf']]
        best['min_samples_split'] = np.arange(2, 10, 2)[best['min_samples_split']]
        best['n_estimators'] = np.arange(10, max_iterations, 10)[best['n_estimators']]
        best.update(dict_concat)
        res = RandomForestClassifier(**best)
        if oversampling:
            sm = SMOTE()
            X_ovr, y_ovr = sm.fit_sample(X, y)
            res.fit(X_ovr, y_ovr)
        else:
            res.fit(X, y)

    if model == 'et':
        space = {
            'bootstrap': hp.choice('bootstrap', [True, False]),
            'max_depth': hp.choice('max_depth', np.arange(5, 50, 1)),
            'max_features': hp.choice('max_features', ['auto', 'sqrt']),
            'min_samples_leaf': hp.choice('min_samples_leaf', np.arange(1, 5, 1)),
            'min_samples_split': hp.choice('min_samples_split', np.arange(2, 10, 2)),
            'n_estimators': hp.choice('n_estimators', np.arange(10, max_iterations, 10)),
            'verbose': 0,
            'random_state': random_state,
            'class_weight': class_w,
        }
        space.update(dict_concat)
        best = fmin(fn=hyperopt_rf_score, space=space, algo=tpe.suggest, max_evals=evals)
        best['bootstrap'] = [True, False][best['bootstrap']]
        best['max_depth'] = np.arange(5, 50, 1)[best['max_depth']]
        best['max_features'] = ['auto', 'sqrt'][best['max_features']]
        best['min_samples_leaf'] = np.arange(1, 5, 1)[best['min_samples_leaf']]
        best['min_samples_split'] = np.arange(2, 10, 2)[best['min_samples_split']]
        best['n_estimators'] = np.arange(10, max_iterations, 10)[best['n_estimators']]
        best.update(dict_concat)
        res = RandomForestClassifier(**best)
        if oversampling:
            sm = SMOTE()
            X_ovr, y_ovr = sm.fit_sample(X, y)
            res.fit(X_ovr, y_ovr)
        else:
            res.fit(X, y)

    if model == 'xgb':
        space = {
            'max_depth': hp.choice('max_depth', np.arange(3, 13, 1)),
            'learning_rate': hp.uniform('learning_rate', 0.001, 0.1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
            'lambda': hp.uniform('lambda', 0.0, 1.0),
            'gamma': hp.uniform('gamma', 0.0, 1.0),
            'min_child_weight': hp.choice('min_child_weight', np.arange(1, 13, 1)),
            'subsample': hp.uniform('subsample', 0.3, 1.0),
            'n_estimators': hp.choice('n_estimators', np.arange(10, max_iterations, 10)),
            'nthread': -1,
            'verbosity': 0,
            # 'tree_method': 'gpu_hist',
            # 'predictor': 'gpu_predictor',
            'random_state': random_state,
        }
        space.update(dict_concat)
        best = fmin(fn=hyperopt_xgb_score, space=space, algo=tpe.suggest, max_evals=evals, show_progressbar=False)
        best['max_depth'] = np.arange(3, 13, 1)[best['max_depth']]
        best['min_child_weight'] = np.arange(1, 13, 1)[best['min_child_weight']]
        best['n_estimators'] = np.arange(10, max_iterations, 10)[best['n_estimators']]
        best.update(dict_concat)
        res = XGBClassifier(**best)
        if oversampling:
            sm = SMOTE()
            X_ovr, y_ovr = sm.fit_sample(X, y)
            res.fit(X_ovr, y_ovr)
        else:
            res.fit(X, y, sample_weight=w_array)

    if model == 'lgb':
        space = {
            'max_depth': hp.choice('max_depth', np.arange(3, 13, 1)),
            'learning_rate': hp.uniform('learning_rate', 0.001, 0.1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
            'num_leaves': hp.choice('num_leaves', np.arange(20, 201, 5)),
            'lambda': hp.uniform('lambda', 0.0, 1.0),
            'gamma': hp.uniform('gamma', 0.0, 1.0),
            'min_child_weight': hp.choice('min_child_weight', np.arange(1, 13, 1)),
            'subsample': hp.uniform('subsample', 0.3, 1.0),
            'n_estimators': hp.choice('n_estimators', np.arange(10, max_iterations, 10)),
            'random_state': random_state,
        }
        space.update(dict_concat)
        best = fmin(fn=hyperopt_lgb_score, space=space, algo=tpe.suggest, max_evals=evals, show_progressbar=False)
        best['max_depth'] = np.arange(3, 13, 1)[best['max_depth']]
        best['num_leaves'] = np.arange(20, 201, 5)[best['num_leaves']]
        best['min_child_weight'] = np.arange(1, 13, 1)[best['min_child_weight']]
        best['n_estimators'] = np.arange(10, max_iterations, 10)[best['n_estimators']]
        best.update(dict_concat)
        res = LGBMClassifier(**best)
        if oversampling:
            sm = SMOTE()
            X_ovr, y_ovr = sm.fit_sample(X, y)
            res.fit(X_ovr, y_ovr)
        else:
            res.fit(X, y, sample_weight=w_array)

    if model == 'ctb':
        space = {
            'max_depth': hp.choice('max_depth', np.arange(3, 13, 1)),
            'learning_rate': hp.uniform('learning_rate', 0.001, 0.1),
            'colsample_bylevel': hp.uniform('colsample_bylevel', 0.3, 1.0),
            'l2_leaf_reg': hp.uniform('l2_leaf_reg', 0.0, 1.0),
            'n_estimators': hp.choice('n_estimators', np.arange(10, max_iterations, 10)),
            'verbose': 0,
            'random_state': random_state,
        }
        space.update(dict_concat)
        best = fmin(fn=hyperopt_ctb_score, space=space, algo=tpe.suggest, max_evals=evals)
        best['max_depth'] = np.arange(3, 13, 1)[best['max_depth']]
        best['n_estimators'] = np.arange(10, max_iterations, 10)[best['n_estimators']]
        best.update(dict_concat)
        res = CatBoostClassifier(**best)
        if oversampling:
            sm = SMOTE()
            X_ovr, y_ovr = sm.fit_sample(X, y)
            res.fit(X_ovr, y_ovr)
        else:
            res.fit(X, y, sample_weight=w_array)

    return res
