import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from stellargraph.layer import GCN, LabelGCN
from scipy import sparse
from stellargraph.core.utils import GCN_Aadj_feats_op
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph import StellarGraph
from tensorflow.keras import layers, optimizers, losses, Model
from sklearn import preprocessing
import pandas as pd
from numpy.random import seed
from sklearn.metrics import *
from collections import OrderedDict
import utils


def init_results_dict():
    return {'precision': [], 'recall': [], 'f1': [], 'f1_shutdown': [], 'accuracy': []}


def run_job(logger, layer_sizes, bias, learning_rate, dropout, n_epochs, n_training_tsteps, n_oversampling):

    df_features, df_classes_id, tx_colnames, agg_colnames, df_features_class, \
        df_features_with_label, df_edgelist = utils.get_elliptic_data()

    df_features_no_label = df_features_with_label.loc[:, df_features_with_label.columns != 'GCN-label']

    #all graph
    G, _ = utils.get_graph_without_label_elliptic(df_classes_id, df_features_no_label, df_edgelist)

    # Sub-graph for training
    train_ids = df_features[df_features['time step'] <= n_training_tsteps].id
    # Target ids
    train_target_ids, test_target_ids = utils.split_data(df_features_class, n_training_tsteps)

    #subgraph for training
    G1 = G.subgraph(train_ids)
    node_subjects = df_classes_id[df_classes_id.label != 2]
    node_subjects = pd.Series(data=node_subjects.label.values, index=node_subjects.index)
    node_subjects.index.name = None

    train_subjects = node_subjects[node_subjects.index.isin(train_target_ids)]
    test_subjects = node_subjects[node_subjects.index.isin(test_target_ids)]
    target_encoding = preprocessing.LabelBinarizer()

    #oversample training nodes for illicit
    illicit = train_subjects.loc[lambda x: x == 1]
    train_subjects_no_oversampling = train_subjects.copy()
    labels = [train_subjects]
    for _ in range(n_oversampling):
        labels += [illicit]
    train_subjects = pd.concat(labels, axis=0)

    train_targets_no_oversampling = target_encoding.fit_transform(train_subjects_no_oversampling)
    train_targets_no_oversampling = utils.encoding(train_targets_no_oversampling)
    train_targets = target_encoding.fit_transform(train_subjects)
    train_targets = utils.encoding(train_targets)
    test_targets = target_encoding.transform(test_subjects)
    test_targets = utils.encoding(test_targets)

    generator = FullBatchNodeGenerator(G1, method="gcn")
    train_gen_no_oversampling = generator.flow(train_subjects_no_oversampling.index, train_targets_no_oversampling)
    train_gen = generator.flow(train_subjects.index, train_targets)

    gcn = GCN(
        layer_sizes=layer_sizes, activations=['relu'] * len(layer_sizes), generator=generator, dropout=dropout, bias=bias
    )
    x_inp, x_out = gcn.in_out_tensors()
    predictions = layers.Dense(units=train_targets.shape[1], activation='softmax')(x_out)

    model1 = Model(inputs=x_inp, outputs=predictions)
    model1.compile(
        optimizer=optimizers.Adam(lr=learning_rate),
        loss=losses.categorical_crossentropy,
        metrics=['acc'],
    )
    model1.fit(
        train_gen,
        epochs=n_epochs,
        verbose=2,
        shuffle=False,
        callbacks=None
    )

    embedding_model_no_label = Model(inputs=x_inp, outputs=x_out)
    embeds_training_no_label = embedding_model_no_label.predict(train_gen_no_oversampling)
    embeds_training_no_label = embeds_training_no_label.squeeze(0)
    embeds_testing_no_label = []
    y_gcn_no_label = []

    weights = model1.get_weights()

    for tstep in range(35, 50):
        print('tstep: {}'.format(tstep))
        test_ids_tstep = df_features[(df_features['time step'] == tstep) & (df_features.id.isin(test_target_ids))]['id'].values
        # Sub-graph includes testing + training
        nodes_ids = df_features[df_features['time step'] <= tstep].id  # all graph up until this time step
        graph_sampled = G.subgraph(nodes_ids)
        Adj = graph_sampled.to_adjacency_matrix()
        node_features = graph_sampled.node_features()
        _, Adj_norm = GCN_Aadj_feats_op(node_features, Adj)
        df_features_subgraph = df_features_no_label[df_features_no_label.index.isin(nodes_ids)]
        h1 = sparse.csr_matrix.dot(Adj_norm, df_features_subgraph)
        h1 = np.dot(h1, weights[0])
        h1 = K.relu(h1)
        h2 = sparse.csr_matrix.dot(Adj_norm, h1)
        h2 = np.dot(h2, weights[1])
        h2 = K.relu(h2)
        y = np.dot(h2, weights[2])
        y = K.softmax(y + weights[3]).numpy()
        test_id_ilocs = graph_sampled.node_ids_to_ilocs(test_ids_tstep)
        for test_id_iloc in test_id_ilocs:
            emb_test_id = h2[test_id_iloc].numpy()
            embeds_testing_no_label.append(emb_test_id)
            y_gcn_no_label.append(y[test_id_iloc].round()[1])

    # ----- Label-GCN -----
    G = utils.get_graph_with_label_elliptic(df_features_with_label, df_edgelist, train_subjects, pd.Series())

    #subgraph for training
    G2 = G.subgraph(train_ids)

    generator = FullBatchNodeGenerator(G2, method='gcn')
    train_gen_no_oversampling = generator.flow(train_subjects_no_oversampling.index, train_targets_no_oversampling)
    train_gen = generator.flow(train_subjects.index, train_targets)

    gcn2 = LabelGCN(layer_sizes=layer_sizes, activations=['relu'] * len(layer_sizes), generator=generator, dropout=dropout,
                    label_idxs=[-1], bias=bias)

    x_inp, x_out = gcn2.in_out_tensors()  # inp, out aren't defined yet in their values
    predictions2 = layers.Dense(units=train_targets.shape[1], activation='softmax')(x_out)

    model2 = Model(inputs=x_inp, outputs=predictions2)
    model2.compile(
        optimizer=optimizers.Adam(lr=learning_rate),
        loss=losses.categorical_crossentropy,
        metrics=['acc'],
    )
    model2.fit(
        train_gen,
        epochs=n_epochs,
        verbose=2,
        shuffle=False,  # this should be False, since shuffling data means shuffling the whole graph
    )

    weights = model2.get_weights()
    embedding_model = Model(inputs=x_inp, outputs=x_out)
    embeds_training = embedding_model.predict(train_gen_no_oversampling)
    embeds_training = embeds_training.squeeze(0)
    embeds_testing = []
    graph_full = StellarGraph(df_features_with_label, df_edgelist)
    y_gcn = []
    for tstep in range(35, 50):
        test_ids_tstep = df_features[(df_features['time step'] == tstep) & (df_features.id.isin(test_target_ids))]['id'].values
        # Sub-graph includes testing + training
        nodes_ids = df_features[df_features['time step'] <= tstep].id  # all graph up until this time step
        graph_sampled = graph_full.subgraph(nodes_ids)
        Adj = graph_sampled.to_adjacency_matrix()
        node_features = graph_sampled.node_features()
        _, Adj_norm = GCN_Aadj_feats_op(node_features, Adj)
        df_features_subgraph = df_features_with_label[df_features_with_label.index.isin(nodes_ids)]
        for i, test_id in enumerate(test_ids_tstep):
            print('tstep: {}, i: {}'.format(tstep, i))
            df = df_features_subgraph.copy()
            df.loc[test_id, 'GCN-label'] = 0
            df = df.to_numpy()
            h1 = sparse.csr_matrix.dot(Adj_norm, df)
            h1 = np.dot(h1, weights[0])
            h1 = K.relu(h1)
            h2 = sparse.csr_matrix.dot(Adj_norm, h1)
            h2 = np.dot(h2, weights[1])
            h2 = K.relu(h2)
            y = np.dot(h2, weights[2])
            y = K.softmax(y + weights[3]).numpy()
            test_id_iloc = graph_sampled.node_ids_to_ilocs([test_id])[0]
            y_gcn.append(y[test_id_iloc].round()[1])
            emb_test_id = h2[test_id_iloc].numpy()
            embeds_testing.append(emb_test_id)

    y_true = [float(yy.round()[1]) for yy in test_targets]

    accuracy1 = accuracy_score(y_true, y_gcn_no_label)
    accuracy2 = accuracy_score(y_true, y_gcn)

    logger.info('accuracy w/o label feature: {}'.format(accuracy1))
    logger.info('accuracy w/  label feature: {}'.format(accuracy2))

    return [y_true, y_gcn_no_label, y_gcn,
            embeds_training_no_label, embeds_testing_no_label,
            embeds_training, embeds_testing]


if __name__ == '__main__':
    dataset = 'elliptic'
    nstates, nruns1, nruns2 = utils.parse_command_line_args_inductive()
    seed(1)
    validation = True
    bias = False
    dropout = 0.5
    n_illicit_oversampling = 6
    n_training_tsteps = 34
    learning_rate = 0.001
    layer_sizes = [100, 100]
    n_epochs = 1000
    num_random_states = 10
    num_runs_standard_models1 = 100
    num_runs_standard_models2 = 10
    # overwrite default values from command line
    if nstates is not None:
        num_random_states = nstates
    if nruns1 is not None:
        num_runs_standard_models1 = nruns1
    if nruns2 is not None:
        num_runs_standard_models2 = nruns2

    dark_market_shutdown_tstep = 43

    logger = utils.get_logger(dataset + '_inductive')
    logger.info('layer_size: {}, learning rate: {}, dropout: {}, n_oversampling: {}, num_random_states: {}'
                .format(layer_sizes, learning_rate, dropout, n_illicit_oversampling, num_random_states))
    logger_results = utils.get_logger(dataset + '_inductive_results')
    seed = 2021
    random_state = np.random.RandomState(seed)
    tf_seeds = random_state.randint(0, 1000000, num_random_states)
    count = 0
    all_gcn_results = []
    for i in range(num_random_states):
        logger_results.info('random state: {}'.format(i))
        tf.random.set_seed(tf_seeds[i])
        count += 1
        tf.keras.backend.clear_session()
        res = run_job(logger, layer_sizes, bias, learning_rate, dropout, n_epochs, n_training_tsteps, n_illicit_oversampling)
        all_gcn_results.append(res)
    logger_results.info('*' * 50)
    logger_results.info('layer_size: {}, learning rate: {}, dropout: {}, n_oversampling: {}, num_random_states: {}'
                        .format(layer_sizes, learning_rate, dropout, n_illicit_oversampling, num_random_states))

    df_features, df_classes_id, tx_colnames, agg_colnames, df_features_class, df_features_with_label, \
        df_edgelist = utils.get_elliptic_data()
    len_after_shutdown = utils.get_len_after_shutdown(df_features_class, dark_market_shutdown_tstep)

    gcn_results_no_label, gcn_results_label = utils.performance_gcn(all_gcn_results, len_after_shutdown)
    for title, gcn_averaged in zip(['w/o label', 'w/ label'], [gcn_results_no_label, gcn_results_label]):
        for k in gcn_averaged.keys():
            mean = np.mean(gcn_averaged[k])
            standard_deviation = np.std(gcn_averaged[k])
            logger_results.info('{} {}: {} {} {}'.format(k, title, str(mean), u'\u00B1', standard_deviation))
    logger_results.info('------------------')

    print('Evaluating standard models')
    X_train, X_test, y_train, y_test = utils.dfs_for_standard_models(df_features_class, tx_colnames,
                                                                     agg_colnames, n_training_tsteps)

    mean_lr, _ = utils.performance_standard_models(X_train, y_train, X_test, y_test, len_after_shutdown,
                                                   nruns=num_runs_standard_models1, method='LR')
    for k, v in mean_lr.items():
        logger_results.info('LR - {}: {}'.format(k, v))

    mean_rf, _ = utils.performance_standard_models(X_train, y_train, X_test, y_test, len_after_shutdown,
                                                   nruns=num_runs_standard_models1, method='RF')
    for k, v in mean_rf.items():
        logger_results.info('RF - {}: {}'.format(k, v))

    mean_xg, _ = utils.performance_standard_models(X_train, y_train, X_test, y_test, len_after_shutdown,
                                                   nruns=num_runs_standard_models1, method='XG')
    for k, v in mean_xg.items():
        logger_results.info('XGBOOST - {}: {}'.format(k, v))

    all_results = OrderedDict({'LR_no_label_embs': init_results_dict(), 'LR_embs': init_results_dict(),
                   'RF_no_label_embs': init_results_dict(), 'RF_embs': init_results_dict(),
                   'XGBOOST_no_label_embs': init_results_dict(), 'XGBOOST_embs': init_results_dict()})
    print('Evaluating metrics with embeddings')
    for i, res in enumerate(all_gcn_results):
        X_train_no_label = pd.concat([X_train, pd.DataFrame(data=res[3])], axis=1)
        X_test_no_label = pd.concat([X_test, pd.DataFrame(data=res[4])], axis=1)
        X_train_label = pd.concat([X_train, pd.DataFrame(data=res[5])], axis=1)
        X_test_label = pd.concat([X_test, pd.DataFrame(data=res[6])], axis=1)

        _, all_lr_no_label = utils.performance_standard_models(X_train_no_label, y_train, X_test_no_label, y_test,
                                                           len_after_shutdown, nruns=num_runs_standard_models2, method='LR')
        _, all_lr = utils.performance_standard_models(X_train_label, y_train, X_test_label, y_test,
                                                      len_after_shutdown, nruns=num_runs_standard_models2, method='LR')
        _, all_rf_no_label = utils.performance_standard_models(X_train_no_label, y_train, X_test_no_label, y_test,
                                                           len_after_shutdown, nruns=num_runs_standard_models2, method='RF')
        _, all_rf = utils.performance_standard_models(X_train_label, y_train, X_test_label, y_test,
                                                      len_after_shutdown, nruns=num_runs_standard_models2, method='RF')
        _, all_xg_no_label = utils.performance_standard_models(X_train_no_label, y_train, X_test_no_label, y_test,
                                                           len_after_shutdown, nruns=num_runs_standard_models2, method='XG')
        _, all_xg = utils.performance_standard_models(X_train_label, y_train, X_test_label, y_test,
                                                      len_after_shutdown, nruns=num_runs_standard_models2, method='XG')
        this = {'LR_no_label_embs': all_lr_no_label, 'LR_embs': all_lr,
                'RF_no_label_embs': all_rf_no_label, 'RF_embs': all_rf,
                'XGBOOST_no_label_embs': all_xg_no_label, 'XGBOOST_embs': all_xg}

        for method in all_results.keys():
            for metric in all_results[method].keys():
                all_results[method][metric].extend(this[method][metric])
    for method in all_results.keys():
        for k, v in all_results[method].items():
            mean = np.mean(v)
            standard_deviation = np.std(v)
            logger_results.info('{} - {}: {} {} {}'.format(method, k, str(mean), u'\u00B1', standard_deviation))


