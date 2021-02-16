from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler
from sklearn.pipeline import Pipeline

from scipy.sparse import load_npz, save_npz, hstack, csr_matrix
from collections import defaultdict
import pandas as pd
import datetime
import pywFM
import argparse
import numpy as np
import os

from .dataio import build_new_paths, prepare_folder, get_legend, get_strongest_folds, get_pseudostrong_folds
from .prepare_data import prepare_dataset
from .encode import df_to_sparse
import glob
import time
import json

# Location of libFM's compiled binary file
os.environ['LIBFM_PATH'] = os.path.join(os.path.dirname(__file__),
                                        'libfm/bin/')


all_features = ['users', 'items', 'skills', 'wins', 'fails', 'attempts', 'tw_kc', 'tw_items']


def factor_analysis_processing(dataset, model, d=0, metrics=['acc', 'auc', 'nll', 'rmse'], generalization='strongest',
                    min_interactions=15, remove_irrelevant_users=True, n_iter=5, verbose=False):

    assert model in ('afm', 'irt', 'pfa', 'lfa', 'mirt'), "Logistic regression model unknown."
    features = {'afm': ['skills', 'attempts'],
                'irt': ['users', 'items'],
                'mirt': ['users', 'items'],
                'pfa': ['skills', 'wins', 'fails'],
                'lfa': ['users', 'skills', 'wins', 'fails'],
                'pfae': ['index', 'skills', 'wins', 'fails']}

    df, q_mat = prepare_dataset(dataset=dataset,
                                min_interactions_per_user=min_interactions,
                                remove_irrelevant_users=remove_irrelevant_users,
                                verbose=verbose)

    # 2ND STEP: SPLITTING DATA
    if verbose:
        print("Splitting data...")

    if generalization == "strongest":
        folds = get_strongest_folds(df, axis="user_id", nb_folds=5)
    elif generalization == "pseudostrong":
        folds = get_pseudostrong_folds(df, perc_initial=.2, nb_folds=5)
    else:
        return TypeError("Unknown generalization scheme.")

    if verbose:
        print("Encoding data...")

    x = df_to_sparse(df, q_mat, features[model], verbose)
    y = x[:, 0].toarray().flatten()

    # FM parameters
    params = {
        'task': 'classification',
        'num_iter': n_iter,
        'rlog': True,
        'learning_method': 'mcmc',
        'k2': d
    }

    auc_list, acc_list, rmse_list, nll_list = [], [], [], []
    for test_ids in folds:
        train_ids = list(set(range(x.shape[0])) - set(test_ids))
        x_train = x[train_ids, 1:]
        y_train = y[train_ids]
        x_test = x[test_ids, 1:]
        y_test = y[test_ids]

        if verbose:
            print('fitting...')

        if d == 0:
            estimators = [
                ('maxabs', MaxAbsScaler()),
                ('lr', LogisticRegression(solver="saga", max_iter=n_iter, C=1.))
            ]
            pipe = Pipeline(estimators)
            pipe.fit(x_train, y_train)
            y_pred_test = pipe.predict_proba(x_test)[:, 1]
        else:
            transformer = MaxAbsScaler().fit(x_train)
            fm = pywFM.FM(**params)
            model = fm.run(transformer.transform(x_train), y_train,
                           transformer.transform(x_test), y_test)
            y_pred_test = np.array(model.predictions)

        temp_acc = accuracy_score(y_test, np.round(y_pred_test))
        temp_auc = roc_auc_score(pd.Series(y_test), pd.Series(y_pred_test))
        temp_nll = log_loss(y_test, y_pred_test)
        temp_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        acc_list.append(temp_acc)
        auc_list.append(temp_auc)
        nll_list.append(temp_nll)
        rmse_list.append(temp_rmse)

    results = {'acc': np.mean(acc_list), 'auc': np.mean(auc_list), 'nll': np.mean(nll_list), 'rmse': np.mean(rmse_list)}
    errors = {'acc': np.std(acc_list), 'auc': np.std(auc_list), 'nll': np.std(nll_list), 'rmse': np.std(rmse_list)}

    return results, errors
