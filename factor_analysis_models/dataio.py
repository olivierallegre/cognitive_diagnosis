import os
import datetime
import numpy as np
from sklearn.model_selection import KFold


def prepare_folder(path):
    """Create folder from path."""
    if not os.path.isdir(path):
        os.makedirs(path)


def build_new_paths(DATASET_NAME):
    """Create dataset folder path name."""
    DATA_FOLDER = './data'
    CSV_FOLDER = os.path.join(DATA_FOLDER, DATASET_NAME)
    return CSV_FOLDER


def get_legend(experiment_args):
    """Generate legend for an experiment.
    Argument:
    experiment_args -- experiment arguments (from ArgumentParser)
    Outputs:
    short -- short legend (str)
    full -- full legend (str)
    latex -- latex legend (str)
    active -- list of active variables
    """
    dim = experiment_args['d']
    short = ''
    full = ''
    agents = ['users', 'items', 'skills', 'wins', 'fails', 'attempts']
    active = []
    for agent in agents:
        if experiment_args.get(agent):
            short += agent[0]
            active.append(agent)
    if experiment_args.get('tw_kc'):
        short += 't1'
        active.append("tw_kc")
    elif experiment_args.get('tw_items'):
        short += 't2'
        active.append("tw_items")
    short += '_'  # add embedding dimension after underscore
    short += str(dim)
    prefix = ''
    if set(active) == {'users', 'items'} and dim == 0:
        prefix = 'IRT: '
    elif set(active) == {'users', 'items'} and dim > 0:
        prefix = 'MIRTb: '
    elif set(active) == {'skills', 'attempts'}:
        prefix = 'AFM: '
    elif set(active) == {'skills', 'wins', 'fails'}:
        prefix = 'PFA: '
    elif set(active) == {'users', 'items', 'skills', 'wins', 'attempts', 'tw_kc'}:
        prefix = 'DAS3H: '
    elif set(active) == {'users', 'items', 'wins', 'attempts', 'tw_items'}:
        prefix = 'DASH: '
    full = prefix + ', '.join(active) + f' d = {dim}'
    latex = prefix + ', '.join(active)
    return short, full, latex, active


def get_strongest_folds(full, axis="user_id", nb_folds=5):
    all_elements = full[axis].unique()
    kfold = KFold(nb_folds, shuffle=True)
    folds = []
    for i, (train, test) in enumerate(kfold.split(all_elements)):
        list_of_test_ids = []
        for element_id in test:
            list_of_test_ids += list(full.query(f'{axis} == {element_id}').index)
        folds.append(np.array(list_of_test_ids))
    return folds



def get_pseudostrong_folds(full, dataset_name, perc_initial=.2, nb_folds=5):
    all_users = full["user_id"].unique()
    kfold = KFold(nb_folds, shuffle=True)
    for i, (train, test) in enumerate(kfold.split(all_users)):
        path = "./data/" + dataset_name + "/pseudostrong/folds"
        prepare_folder(path)
        list_of_test_ids = []
        for user_id in test:
            fold = full.query('user_id == {}'.format(user_id)).sort_values('timestamp').index
            list_of_test_ids += list(fold[round(perc_initial * len(fold)):])
        np.save(path + '/test_fold{}.npy'.format(i), np.array(list_of_test_ids))


