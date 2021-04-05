import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
from stellargraph import StellarGraph
from xgboost import XGBClassifier
from collections import OrderedDict
import logging
import argparse
import sys


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('{}.log'.format(name))
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def split_data(df_features_class, n_training_tsteps):
    labelled_data = df_features_class.loc[(df_features_class['label'] == 0) | (df_features_class['label'] == 1)]
    return list(labelled_data[labelled_data['time step'] <= n_training_tsteps].index), \
           list(labelled_data[labelled_data['time step'] > n_training_tsteps].index)


def get_elliptic_data():
    # Importing classes.csv file, renaming the columns to (id, label)
    df_classes = pd.read_csv('elliptic_bitcoin_dataset/elliptic_txs_classes.csv')
    df_classes.columns = ['id', 'label']
    # Renaming labels to (2,0,1)
    df_classes['label'] = df_classes['label'].map({'unknown': 2, '2': 0, '1': 1})

    # Importing edgelist.csv file, renaming the coulmns to (source, target).
    # Must be source and target in order to be used in stellargraph
    df_edgelist = pd.read_csv('../elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv')
    df_edgelist.columns = ['source', 'target']

    # Importing features.csv file, renaming features as well. first to (id, time step) then trans_feat_ for the next 93 feathres and agg_feat_ for the rest
    df_features = pd.read_csv('../elliptic_bitcoin_dataset/elliptic_txs_features.csv', header=None)
    tx_features = [f'trans_feat_{i}' for i in range(93)]
    agg_features = [f'agg_feat_{i}' for i in range(72)]
    df_features.columns = ['id', 'time step'] + tx_features + agg_features

    df_classes_id = df_classes
    df_classes_id['label'] = df_classes_id['label']
    df_classes_id = df_classes_id.set_index('id')

    df_features_new = df_features.copy()
    df_features_new['GCN-label'] = df_classes_id['label'].map({2: 0, 0: -1, 1: 1}).values

    df_features_class = pd.merge(df_features_new, df_classes, left_on='id', right_on='id', how='left')
    df_features_class = df_features_class.set_index('id')

    df_features_new = df_features_class[['time step'] + tx_features + agg_features + ['GCN-label']]

    return df_features, df_classes_id, tx_features, agg_features, df_features_class, df_features_new, df_edgelist


def get_graph_with_label_elliptic(df_features, df_edgelist, train_subjects, val_subjects,
                                  support_subset_index=None, test=False):
    df_features = df_features.copy()
    if support_subset_index is None:
        support_subset_index = []
    if test:
        df_features.loc[(~df_features.index.isin(train_subjects.index)) &
                        (~df_features.index.isin(val_subjects.index)) &
                        (~df_features.index.isin(support_subset_index)), 'GCN-label'] = 0
    else:
        df_features.loc[(~df_features.index.isin(train_subjects.index)), 'GCN-label'] = 0
    G = StellarGraph(df_features, df_edgelist)
    return G


def get_graph_without_label_elliptic(df_classes_id, df_features, df_edgelist):
    G = StellarGraph(df_features, df_edgelist)
    node_subjects = df_classes_id[df_classes_id.label != 0]
    s = pd.Series(data=node_subjects.label.values, index=node_subjects.index)
    s.index.name = None
    return G, s


def get_graph(features, edges, node_subjects, train_subjects, val_subjects, support_subset=None, test=False):
    cols = ['feat{}'.format(i) for i in range(len(features[0]))]
    features = pd.DataFrame(features, columns=cols, index=node_subjects.index)
    s = pd.Series(data=node_subjects.values, index=node_subjects.index)
    dummies = pd.get_dummies(s)
    scaler = StandardScaler()
    scaler.fit(dummies)
    dummies = pd.DataFrame(scaler.transform(dummies), columns=dummies.columns, index=dummies.index)
    if test:
        for node_id in node_subjects.index:  # make sure we don't use information from test nodes
            if (node_id not in support_subset) and (node_id not in train_subjects.index) \
                    and (node_id not in val_subjects.index):
                for col in dummies.columns:
                    dummies.loc[node_id, col] = 0
    else:
        for node_id in node_subjects.index:  # make sure we only use information from train nodes
            if node_id not in train_subjects.index:
                for col in dummies.columns:
                    dummies.loc[node_id, col] = 0
    new_features = pd.concat([features, dummies], axis=1)
    edges = pd.DataFrame({'source': [e[0] for e in edges], 'target': [e[1] for e in edges]})
    G = StellarGraph(new_features, edges)
    n_cols = len(new_features.columns)
    n_dummies = len(dummies.columns)
    return G, n_cols, n_dummies


def encoding(x):
    empty = []
    for idx, i in enumerate(x):
        if i == [0]:
            empty.append(np.array([1, 0]))
        else:
            empty.append(np.array([0, 1]))
    return np.array(empty)


def output_results(y_test, y_pred, logger):
    logger.info('Log Loss= {}'.format(log_loss(y_test, y_pred)))
    logger.info('Illicit F1= {}'.format(f1_score(y_test, y_pred, average='binary', pos_label=1)))
    logger.info('Illicit AUC= {}'.format(roc_auc_score(y_test, y_pred)))
    logger.info('Illicit Recall= {}'.format(recall_score(y_test, y_pred, average='binary', pos_label=1)))
    logger.info('Illicit Precision= {}'.format(precision_score(y_test, y_pred, average='binary', pos_label=1)))
    logger.info('Micro F1= {}'.format(f1_score(y_test, y_pred, average='micro', pos_label=1)))
    return


def dfs_for_standard_models(df_features_class, tx_colnames, agg_colnames, n_training_tsteps):
    data = df_features_class.loc[(df_features_class['label'] == 0) | (df_features_class['label'] == 1)]  # only labelled nodes
    data.reset_index(inplace=True, drop=True)
    data = data.loc[:, data.columns != 'id']

    train = data.loc[(data['time step'] <= n_training_tsteps), ['time step'] + tx_colnames + agg_colnames + ['label']]
    test = data.loc[(data['time step'] > n_training_tsteps), ['time step'] + tx_colnames + agg_colnames + ['label']]

    X_train = train[['time step'] + tx_colnames + agg_colnames].reset_index(drop=True)
    y_train = train['label']

    X_test = test[['time step'] + tx_colnames + agg_colnames].reset_index(drop=True)
    y_test = test['label']

    return X_train, X_test, y_train, y_test


def get_len_after_shutdown(df_features_class, dark_market_shutdown_tstep):
    df_labelled = df_features_class.loc[
        (df_features_class['label'] == 0) | (df_features_class['label'] == 1)]  # only labelled nodes
    return len(df_labelled[df_labelled['time step'] >= dark_market_shutdown_tstep])


def performance_gcn(all_gcn_results, len_after_shutdown):
    y_true = all_gcn_results[0][0]
    gcn_results_no_label = OrderedDict({'precision': [], 'recall': [], 'f1': [], 'f1_shutdown': [], 'accuracy': []})
    gcn_results = OrderedDict({'precision': [], 'recall': [], 'f1': [], 'f1_shutdown': [], 'accuracy': []})
    for res in all_gcn_results:
        y_gcn_no_label, y_gcn = res[1], res[2]
        performance_no_label = performance(y_true, y_gcn_no_label, len_after_shutdown)
        performance_label = performance(y_true, y_gcn, len_after_shutdown)
        for k, v in performance_no_label.items():
            gcn_results_no_label[k].append(v)
        for k, v in performance_label.items():
            gcn_results[k].append(v)
    return gcn_results_no_label, gcn_results


def performance_standard_models(X_train, y_train, X_test, y_test, len_after_shutdown, nruns, method):

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_std = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_std = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    results = OrderedDict({'precision': [], 'recall': [], 'f1': [], 'f1_shutdown': [], 'accuracy': []})
    for run in range(nruns):
        if method == 'LR':
            model = LogisticRegression(random_state=run, max_iter=10000)
        elif method == 'RF':
            model = RandomForestClassifier(random_state=run, n_estimators=50, max_features=50)
        elif method == 'XG':
            model = XGBClassifier(random_state=run)
        else:
            sys.exit('Method invalid')
        model.fit(X_train_std, y_train)
        y_pred = model.predict(X_test_std)
        result = performance(y_test, y_pred, len_after_shutdown)
        for k, v in result.items():
            results[k].append(v)

    results_averaged = OrderedDict({'precision': None, 'recall': None, 'f1': None, 'f1_shutdown': None, 'accuracy': None})
    for k in results_averaged.keys():
        mean = np.mean(results[k])
        standard_deviation = np.std(results[k])
        results_averaged[k] = '{} {} {}'.format(mean, u'\u00B1', standard_deviation)

    return results_averaged, results


def performance(y_test, y_pred, len_after_shutdown):
    f1 = f1_score(y_test, y_pred, average='binary', pos_label=1)
    f1_shutdown = f1_score(y_test[-len_after_shutdown:], y_pred[-len_after_shutdown:], average='binary', pos_label=1)
    recall = recall_score(y_test, y_pred, average='binary', pos_label=1)
    precision = precision_score(y_test, y_pred, average='binary', pos_label=1)
    accuracy = accuracy_score(y_test, y_pred)
    result = {'precision': precision, 'recall': recall, 'f1': f1, 'f1_shutdown': f1_shutdown, 'accuracy': accuracy}
    return result


def init_vars_transductive(dataset, ns, nr):
    bias = False
    learning_rate = 0.01
    dropout = 0.5
    patience = 10
    n_epochs = 1000
    n_oversampling = 0
    if dataset == 'citeseer':
        layer_sizes = [16, 16]
        training_fraction = 0.0364
        num_random_states, num_runs = 10, 100
        max_val = 0.90
    elif dataset == 'cora':
        layer_sizes = [16, 16]
        training_fraction = 0.052
        num_random_states, num_runs = 10, 100
        max_val = 0.90
    elif dataset == 'pubmed':
        layer_sizes = [16, 16]
        training_fraction = 0.00306
        num_random_states, num_runs = 10, 20
        max_val = 0.90
    elif dataset == 'elliptic':
        layer_sizes = [100, 100]
        training_fraction = 0.1
        num_random_states, num_runs = 2, 5
        max_val = 0.80
        dropout = 0.5
        patience = 30
    else:
        sys.exit('dataset invalid')
    # overwrite default values from command line
    if ns is not None:
        num_random_states = ns
    if nr is not None:
        num_runs = nr
    return bias, learning_rate, dropout, patience, n_epochs, n_oversampling, layer_sizes, training_fraction,\
           num_random_states, num_runs, max_val


def parse_command_line_args_transductive():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ds', '--dataset')
    parser.add_argument('-ns', '--num_random_states', default=None)
    parser.add_argument('-nr', '--num_runs', default=None)
    args = vars(parser.parse_args())
    return args['dataset'], int(args['num_random_states']), int(args['num_runs'])


def parse_command_line_args_inductive():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ns', '--num_random_states', default=None)
    parser.add_argument('-nr1', '--num_runs_1', default=None)
    parser.add_argument('-nr2', '--num_runs_2', default=None)
    args = vars(parser.parse_args())
    return int(args['num_random_states']), int(args['num_runs_1']), int(args['num_runs_2'])
