from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from skfeature.function.sparse_learning_based import RFS as RFS_sk
from sklearn.linear_model import ARDRegression
from ReliefF import ReliefF as ReliefF_sk
from skfeature.function.statistical_based import CFS as CFS_sk
from skfeature.function.similarity_based import lap_score as lap_score_sk
#from fsfc.generic import NormalizedCut, GenericSPEC, NormalizedCut, WKMeans, MCFS
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, roc_auc_score
import pandas as pd
import numpy as np
from functions.preprocessing import filt_features
from sklearn.linear_model import LogisticRegression
from functions.preprocessing import normalization
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from boruta import BorutaPy
from sklearn.tree import DecisionTreeClassifier
from functions.functions import cross_val_split_score, split



def ttest(X_train, y_train, X_test=None, y_test=None, X_control=None, pvalue=0.05):
    """
    Производится отбор признаков t-тестом (критерий Стьюдента). Остаются только те, у которых РДУЗ < pvalue.

    Input:
        X_train (pd.DataFrame) - обучающая выборка
        y_train (pd.Series) - целевая перменная обучающей выборки
        X_test (pd.DataFrame) - тестовая выборка
        y_test (pd.Series) - целевая перменная тестовой выборки
        pvalue (float) - реально допустимый уровень значимости критерия

    Output:
        X_train_dwn, X_test_dwn
    """
    X_train_pos = X_train[y_train == 1]
    X_train_neg = X_train[y_train == 0]
    res = {}
    cols = []
    for a in X_train.columns:
        rvs1 = X_train_pos.loc[:, a].values
        rvs2 = X_train_neg.loc[:, a].values
        p = ttest_ind(rvs1, rvs2, equal_var=False)[1]
        res[a] = p
        if p < pvalue:
            cols.append(a)

    X_train_dwn = X_train.loc[:, cols]
    try:
        X_test_dwn = X_test.loc[:, cols]
        X_control_dwn = X_control.loc[:, cols]
        return X_train_dwn, X_test_dwn, X_control_dwn
    except:
        return X_train_dwn


def pca(X_train, y_train=None, X_test=None, y_test=None, X_control=None, n_components=0.95):
    """
    Производится понижение размерности методом главных компонент.

    Input:
        X_train (pd.DataFrame) - обучающая выборка
        X_test (pd.DataFrame) - тестовая выборка
        n_components (float or int) - количество главных компонент (int) или значение сохраненной информации (float)

    Output:
        X_train_dwn, X_test_dwn
    """
    X_train, y_train, X_test, y_test, scaler = normalization(X_train, y_train, X_test,y_test)
    X_control = pd.DataFrame(scaler.transform(X_control), columns=X_control.columns)
    pca_alg = PCA(n_components)

    X_train_dwn = pca_alg.fit_transform(X_train)
    X_train_dwn = pd.DataFrame(X_train_dwn, index=X_train.index)
    variance = pca_alg.explained_variance_
    res = {f'{i}': variance[i] for i in range(X_train_dwn.shape[1])}
    try:
        X_test_dwn = pca_alg.transform(X_test)
        X_test_dwn = pd.DataFrame(X_test_dwn, index=X_test.index)

        X_control_dwn = pca_alg.transform(X_control)
        X_control_dwn = pd.DataFrame(X_control_dwn, index=X_control.index)
        return X_train_dwn, X_test_dwn, X_control_dwn
    except:
        return X_train_dwn


def permut(X_train, y_train, X_test, y_test, X_control=None, permut_rounds=10,  id_column=None):

    X_train_1, X_val, y_train_1, y_val = split(X_train, y_train, id_column=id_column)
    model = DecisionTreeClassifier(class_weight='balanced').fit(X_train_1, y_train_1)
    baseline = roc_auc_score(y_val, model.predict_proba(X_val).T[1])

    imp = []
    for col in tqdm(X_train_1.columns):
        rounds_imp = []
        for it in range(permut_rounds):
            saved = X_val[col].copy()
            np.random.seed(it)

            X_val[col] = np.random.permutation(X_val[col])
            y_new = model.predict_proba(X_val).T[1]
            m = roc_auc_score(y_val, y_new)

            X_val[col] = saved

            rounds_imp.append(baseline - m)
        imp.append(np.mean(rounds_imp))

    imp = pd.Series(data=imp, index=X_train_1.columns)

    imp.sort_values(ascending=False, inplace=True)
    #print(imp)
    imp = imp[imp > 0]
    current_list = list(imp.index)
    return X_train[current_list], X_test[current_list], X_control[current_list]


def lr_selection(X_train, y_train, X_test=None, y_test=None, X_control=None):
    X_train_norm, _ = normalization(X_train)
    lr = LogisticRegression(class_weight='balanced', random_state=42)
    lr.fit(X_train_norm, y_train)
    coefs = (lr.coef_ > 0.5).T
    try:
        return X_train.loc[:, coefs], X_test.loc[:, coefs], X_control.loc[:, coefs]
    except:
        return X_train.loc[:, coefs]


def boruta_selection(X_train, y_train, X_test=None,y_test=None, X_control=None):
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose=0, max_iter=100)
    feat_selector.fit(X_train.values, y_train.values)
    try:
        return X_train.iloc[:, feat_selector.support_], X_test.iloc[:, feat_selector.support_], X_control.iloc[:, feat_selector.support_]
    except:
        return X_train.iloc[:, feat_selector.support_]


def greedy(X_train, y_train, X_test, y_test, X_control = None,
           id_column=None, max_cnt_non_improve=3, model_params={}):
    selected_features = []
    cnt_non_improve = 0
    cols = X_train.columns.tolist()
    last_best_score = 0
    feature_remains = X_train.columns.tolist()


    for a in tqdm(cols):
        cnt_of_repeats = 1
        for j in range(cnt_of_repeats):
            best_score = 0.5
            for b in feature_remains:
                current_features = selected_features.copy()
                symm_copy = b.replace('left', 'right') if b.find('left') != -1 else b.replace('right', 'left')
                new_features = list(set([b, symm_copy]))
                current_features += new_features
                clf = DecisionTreeClassifier(**model_params)
                score = cross_val_split_score(clf, X_train[current_features], y_train, metric=roc_auc_score,
                                              vf=True, id_column=id_column, oversampling=False,
                                              class_w='balanced', n_folds=5, print_scores=False)[0]
                #score = roc_auc_score(y_val, clf.predict(X_val[current_features]))
                if score > best_score:
                    best_new = new_features
                    best_score = score
            if best_score > 0.5 and len(feature_remains) != 0:
                selected_features += best_new
                for a in best_new:
                    try:
                        feature_remains.remove(a)
                    except:
                        pass

        #if np.random.choice([True, False], p=[0.75, 0.25]) and len(selected_features) > 1 and len(feature_remains) != 0:
        #    selected_features.remove(np.random.choice(selected_features))

        if len(selected_features) == 0:
            rand_feat = np.random.choice(feature_remains)
            symm_copy = rand_feat.replace('left', 'right') if rand_feat.find('left') != -1 else rand_feat.replace(
                'right', 'left')
            selected_features += list(set([b, symm_copy]))

        diff = best_score - last_best_score
        #print(selected_features, best_score, last_best_score)
        if diff < 0:
            cnt_non_improve += 1
        else:
            last_best_score = best_score
        if cnt_non_improve >= max_cnt_non_improve or best_score == 1.:
            break

    #print()

    #X_train = pd.concat([X_train, X_val])
    #y_train = pd.concat([y_train, y_val])

    return X_train[selected_features], X_test[selected_features], X_control[selected_features]

