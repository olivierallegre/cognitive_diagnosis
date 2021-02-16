import numpy as np
import pandas as pd
from scipy import sparse
import argparse
import os
import json


level_str_to_int = {'CE1': 4, 'CE2': 5}
course_str_to_int = {'Mathématiques': 1, 'mathématiques': 1, 'Maths': 1, 'maths': 1,
                     'Français': 3, 'français': 3, 'fr': 3, 'french': 3}
behavior_str_to_int = {'procedural': 'exercice_technique',
                       'declarative': 'exercice_de_connaissance'}


def remove_non_student_users(df):
    return df[df["account_type"] == "STUDENT"]


def remove_kartable_users(df):
    return df[~df['username'].str.contains("kartable")]


def filter_by_level(df, level):
    assert level in level_str_to_int.keys(), 'Level id unknown.'
    return df[df['level_id'] == level_str_to_int[level]]


def filter_by_course(df, course):
    assert course in course_str_to_int.keys(), 'Course id unknown.'
    return df[df['course_id'] == course_str_to_int[course]]


def filter_by_behavior(df, behavior):
    assert behavior in behavior_str_to_int.keys(), 'Behavior unknown.'
    return df[df['behavior'] == behavior_str_to_int[behavior]]


def prepare_dataset(dataset, min_interactions_per_user, remove_irrelevant_users, verbose=False):
    """
    Preprocess Kartable datasets for item response theory.
    :param dataset: datasert
    :param remove_irrelevant_users:
    :param min_interactions_per_user: minimum number of interactions per student
    :param course: course name, 'all' if all courses
    :param level: level name, 'all' if all levels
    :param behavior: behavior name, 'all' if all behaviors
    :param verbose: verbose
    :return df: preprocessed ASSISTments dataset (pandas DataFrame)
    :return Q_mat: corresponding q-matrix (item-skill relationships sparse array)
    """
    if not isinstance(dataset, pd.DataFrame):
        df = pd.read_csv(f"./data/{dataset}.csv", encoding="latin1", index_col=False)
    else:
        df = dataset

    initial_shape = df.shape[0]

    if verbose:
        print(f"Opened Kartable dataset. Output: {initial_shape} samples.")

    df = df.dropna()  # Clean non complete data
    if verbose:
        print(f"Removed {initial_shape - df.shape[0]} not complete rows.")
        initial_shape = df.shape[0]

    df = df[df['correct'].isin([0, 1])]  # Remove potential continuous outcomes
    if verbose:
        print(f"Removed {df.shape[0] - initial_shape} samples with non-binary outcomes.")
        initial_shape = df.shape[0]

    if remove_irrelevant_users:
        df = remove_non_student_users(df)
        df = remove_kartable_users(df)
        if verbose:
            print(f'Removed {initial_shape - df.shape[0]} samples from non student users.')
            initial_shape = df.shape[0]

    df['correct'] = df['correct'].astype(np.int32)  # Cast outcome as int32
    df = df.groupby("user_id").filter(lambda x: len(x) >= min_interactions_per_user)
    if verbose:
        print(f'Removed {initial_shape - df.shape[0]} samples (users with less '
              f'than {min_interactions_per_user} interactions).')
        initial_shape = df.shape[0]

    df = df.groupby("item_id").filter(lambda x: len(x) >= 100)
    if verbose:
        print(f'Removed {initial_shape - df.shape[0]} samples (less '
              f'than {min_interactions_per_user} interactions per item).')
        initial_shape = df.shape[0]


    df["item_id"] = np.unique(df["item_id"], return_inverse=True)[1]
    df["document_id"] = np.unique(df["document_id"], return_inverse=True)[1]
    df["user_id"] = np.unique(df["user_id"], return_inverse=True)[1]

    # Build q-matrix
    kc_list = df["document_id"].unique()
    item_list = df["item_id"].unique()

    q_mat = np.zeros((len(item_list), len(kc_list)))
    item_skill = np.array(df[["item_id", "document_id"]])
    for i in range(len(item_skill)):
        q_mat[int(item_skill[i, 0]), int(item_skill[i, 1])] = 1

    if verbose:
        print(f"Computed q-matrix. Shape: {q_mat.shape}.")

    # df.sort_values(by="timestamp", inplace=True)
    df.reset_index(inplace=True, drop=True)
    print("Data preprocessing done. Final output: {} samples.".format((df.shape[0])))

    # Save data
    return df, q_mat
