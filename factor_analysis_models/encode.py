import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from collections import defaultdict, Counter
from scipy import sparse
import argparse
import os


feature_translation = {'users': 'user_id', 'skills': 'document_id', 'items':'item_id'}


def compute_previous_attempts(df, idx, item_list):
    row = df.loc[idx]
    item_id = int(row['item_id'])
    user_id = int(row['user_id'])
    sub_df = df.loc[:idx - 1]
    return len(sub_df[(sub_df['user_id'] == user_id) & (sub_df['item_id'] == item_id)].index)


def df_to_sparse(df, Q_mat, active_features, verbose=False):
    """Build sparse features dataset from dense dataset and q-matrix.
    :param df -- dense dataset, output from one function from prepare_data.py (pandas DataFrame)
    :param Q_mat -- q-matrix, output from one function from prepare_data.py (sparse array)
    :param active_features -- features used to build the dataset (list of strings)
    :param tw -- useful when script is *not* called from command line.
    :param verbose -- if True, print information on the encoding process (bool)
    :return sparse_df -- sparse dataset. The 5 first columns of sparse_df are just the same columns as in df.
    Notes:
    * tw_kc and tw_items respectively encode time windows features instead of regular counter features
      at the skill and at the item level for wins and attempts, as decribed in our paper. As a consequence,
      these arguments can only be used along with the wins and/or attempts arguments. With tw_kc, one column
      per time window x skill is encoded, whereas with tw_items, one column per time window is encoded (it is
      assumed that items share the same time window biases).
    """

    df_legend = {x: idx for idx, x in enumerate(list(df.columns))}

    # Transform q-matrix into dictionary
    dict_q_mat = {i: set() for i in range(Q_mat.shape[0])}
    for elt in np.argwhere(Q_mat == 1):
        dict_q_mat[elt[0]].add(elt[1])

    # Initialization of all features
    X = {}
    X['correctness'] = sparse.csr_matrix(np.empty((0, Q_mat.shape[1])))
    X['df'] = np.empty((0, len(df.columns)))  # Keep track of the original dataset

    if "attempts" in active_features:
        X["attempts"] = sparse.csr_matrix(np.empty((0, Q_mat.shape[1])))
    if "wins" in active_features:
        X["wins"] = sparse.csr_matrix(np.empty((0, Q_mat.shape[1])))
    if "fails" in active_features:
        X["fails"] = sparse.csr_matrix(np.empty((0, Q_mat.shape[1])))

    for stud_id in df["user_id"].unique():
        df_stud = df[df["user_id"] == stud_id].copy()
        # df_stud.sort_values(by="item_id", inplace=True)  # Sort values
        df_stud = np.array(df_stud)
        X['df'] = np.vstack((X['df'], df_stud))

        if 'attempts' in active_features:
            skills_temp = Q_mat[df_stud[:, df_legend[feature_translation['items']]].astype(int)].copy()
            attempts = np.multiply(np.cumsum(np.vstack((np.zeros(skills_temp.shape[1]), skills_temp)), 0)[:-1],
                                   skills_temp)
            X['attempts'] = sparse.vstack([X['attempts'], sparse.csr_matrix(attempts.astype(np.float))])

        if "wins" in active_features:
            skills_temp = Q_mat[df_stud[:, df_legend[feature_translation['items']]].astype(int)].copy()
            wins = np.multiply(np.cumsum(np.multiply(np.vstack((np.zeros(skills_temp.shape[1]), skills_temp)),
                                                     np.hstack((np.array([0]), df_stud[:, -1])).reshape(-1, 1)),
                                         0)[:-1],
                               skills_temp)
            X['wins'] = sparse.vstack([X['wins'], sparse.csr_matrix(wins.astype(np.float))])

        if "fails" in active_features:
            skills_temp = Q_mat[df_stud[:, df_legend[feature_translation['items']]].astype(int)].copy()
            fails = np.multiply(np.cumsum(np.multiply(np.vstack((np.zeros(skills_temp.shape[1]), skills_temp)),
                                                      np.hstack((np.array([0]), 1 - df_stud[:, -1])).reshape(-1, 1)),
                                          0)[:-1],
                                skills_temp)
            X["fails"] = sparse.vstack([X["fails"], sparse.csr_matrix(fails.astype(np.float))])

    onehot = OneHotEncoder()
    if 'users' in active_features:
        X['users'] = onehot.fit_transform(X["df"][:, df_legend[feature_translation['users']]].reshape(-1, 1))
        if verbose:
            print("Users encoded.")

    if 'items' in active_features:
        X['items'] = onehot.fit_transform(X["df"][:, df_legend[feature_translation['items']]].reshape(-1, 1))
        if verbose:
            print("Items encoded.")

    # If skills are considered to be alone in documents
    if 'skills' in active_features:
        X['skills'] = onehot.fit_transform(X["df"][:, df_legend[feature_translation['skills']]].reshape(-1, 1))
        if verbose:
            print("Skills encoded.")

    X['correctness'] = [[x] for x in X["df"][:, -1]]
    sparse_df = sparse.hstack([X['correctness']]+[X[agent] for agent in active_features]).tocsr()
    return sparse_df
