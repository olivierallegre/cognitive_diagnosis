import pandas as pd
import argparse
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler
from .data_helper import convert_data
from .check_data import check_data
from .prepare_data import prepare_dataset
from .auc import compute_auc
from .rmse import compute_rmse
from pyBKT.generate import synthetic_data, random_model_uni
from pyBKT.fit import EM_fit, predict_onestep
from .dataio import get_strongest_folds

import glob
import time
import json

parser = argparse.ArgumentParser(description='Run BKT models on selected Kartable.')
parser.add_argument('--dataset', type=str, nargs='?', const=True, default='data')
parser.add_argument('--model', type=str, nargs='?', const=True, default='simple')
parser.add_argument('--level', type=str, nargs='?', const=True, default='all')
parser.add_argument('--course', type=str, nargs='?', const=True, default='all')
parser.add_argument('--behavior', type=str, nargs='?', const=True, default='all')
parser.add_argument('--n_time', type=int, nargs='?', default=0)
parser.add_argument('--axis', type=str, nargs='?', default='document_id')
parser.add_argument('--n_folds', type=int, nargs='?', default=5)
parser.add_argument('--d', type=int, nargs='?', default=0)
parser.add_argument('--verbose', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--min_interactions', type=int, nargs='?', default=5)
parser.add_argument('--remove_irrelevant_users', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--generalization', type=str, nargs='?', default='strongest')
parser.add_argument('--iter', type=int, nargs='?', default=5)
options = parser.parse_args()




def knowledge_tracing_processing(dataset, model, metrics=['acc', 'auc', 'nll', 'rmse'], generalization='strongest',
                    min_interactions=15, remove_irrelevant_users=True, n_iter=5, verbose=False):

    features = {'kt-bkt': {"return_df": False, "defaults": "kartable", "multilearn": False,
                           "multiguess": False, "multipair": False, "multiprior": False},
                'kt-ide': {"return_df": False, "defaults": "kartable", "multilearn": False,
                           "multiguess": True, "multipair": False, "multiprior": False},
                'kt-ile': {"return_df": False, "defaults": "kartable", "multilearn": True,
                           "multiguess": False, "multipair": False, "multiprior": False},
                'kt-ioe': {"return_df": False, "defaults": "kartable", "multilearn": False,
                           "multiguess": False, "multipair": True, "multiprior": False},
                'kt-pps': {"return_df": False, "defaults": "kartable", "multilearn": False,
                           "multiguess": False, "multipair": False, "multiprior": True}}

    df, q_mat = prepare_dataset(dataset=dataset,
                                min_interactions_per_user=min_interactions,
                                remove_irrelevant_users=remove_irrelevant_users,
                                verbose=verbose)

    # 2ND STEP: SPLITTING DATA
    if verbose:
        print("Splitting data...")

    is_working = False
    auc_values = []
    rmse_values = []
    cpt = 0
    while not is_working:
        cpt += 1
        is_working = True
        if generalization == "strongest":
            folds = get_strongest_folds(df, axis='user_id', nb_folds=5)

        if features[model]['multiguess']:
            for test_ids in folds:
                train_ids = list(set(list(df.index.values)) - set(test_ids))
                test_df = df[df.index.isin(test_ids)]
                train_df = df[df.index.isin(train_ids)]
                is_working = is_working and (set(pd.unique(test_df["item_id"])) == set(pd.unique(train_df["item_id"])))

    for test_ids in folds:
        train_ids = list(set(list(df.index.values)) - set(test_ids))

        train_df = df[df.index.isin(train_ids)]
        skills = pd.unique(train_df["document_id"])
        models = {}

        for skill in skills:
            data = convert_data(train_df, skill, **features[model])
            check_data(data)

            if "resource_names" in data:
                num_learns = len(data["resource_names"])
            else:
                num_learns = 1

            if "gs_names" in data:
                num_gs = len(data["gs_names"])
            else:
                num_gs = 1

            best_likelihood = float("-inf")
            for i in range(n_iter):
                fitmodel = random_model_uni.random_model_uni(num_learns,
                                                             num_gs)  # include this line to randomly set initial param values
                (fitmodel, log_likelihoods) = EM_fit.EM_fit(fitmodel, data)
                if (log_likelihoods[-1] > best_likelihood):
                    best_likelihood = log_likelihoods[-1]
                    best_model = fitmodel
            models[skill] = best_model

        # 3. Vérifier la justesse de la prédiction pour le jeu de test
        test_df = df[df.index.isin(test_ids)]
        skills = pd.unique(test_df["document_id"])
        all_true, all_pred = [], []
        for skill in skills:
            test_data = convert_data(test_df, skill, **features[model])
            # run model predictions from training data on test data
            (correct_predictions, state_predictions) = predict_onestep.run(models[skill], test_data)
            flat_true_values = np.zeros((len(test_data["data"][0]),), dtype=np.intc)

            for i in range(len(test_data["data"])):
                for j in range(len(test_data["data"][i])):
                    if test_data["data"][i][j] != 0:
                        flat_true_values[j] = test_data["data"][i][j]
            flat_true_values = flat_true_values.tolist()
            all_true.extend(flat_true_values)
            all_pred.extend(correct_predictions)

        rmse, auc = 0, 0
        try:
            rmse = compute_rmse(all_true, all_pred)
        except ZeroDivisionError:
            print("RMSE ZeroDivisionError")
            pass
        try:
            auc = compute_auc(all_true, all_pred)
        except ValueError:
            print(all_true, all_pred)
            print("AUC ValueError")
            pass

        auc_values.append(auc)
        rmse_values.append(rmse)

    results = {'auc': np.mean(auc_values), 'rmse': np.mean(rmse_values)}
    errors = {'auc': np.std(auc_values), 'rmse': np.std(rmse_values)}

    return results, errors

