import numpy as np
import utils
import tensorflow as tf
from tensorflow.keras import backend as K
import stellargraph as sg
from stellargraph.layer import GCN, LabelGCN
from scipy import sparse
from stellargraph.core.utils import GCN_Aadj_feats_op
from stellargraph.mapper import FullBatchNodeGenerator
from tensorflow.keras import layers, optimizers, losses, Model
from sklearn import preprocessing, model_selection
import sys
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from numpy.random import seed
from sklearn.metrics import *


def run_job(count, logger, dataset, training_fraction, validation_fraction, support_fraction,
            layer_sizes, bias, learning_rate, dropout, fracs, patience, n_epochs, n_oversampling=-1):

    if dataset == 'cora':
        dataset = sg.datasets.Cora()
        G1, node_subjects = dataset.load()
    elif dataset == 'citeseer':
        dataset = sg.datasets.CiteSeer()
        G1, node_subjects = dataset.load()
    elif dataset == 'pubmed':
        dataset = sg.datasets.PubMedDiabetes()
        G1, node_subjects = dataset.load()
    elif dataset == 'elliptic':
        df_features, df_classes_id, tx_colnames, agg_colnames, df_features_class, \
            df_features_with_label, df_edgelist = utils.get_elliptic_data()
        df_features_no_label = df_features_with_label.loc[:, df_features_with_label.columns != 'GCN-label']
        G1, node_subjects = utils.get_graph_without_label_elliptic(df_classes_id, df_features_no_label, df_edgelist)
    else:
        sys.exit('dataset invalid - aborting')

    train_size = int(training_fraction * len(node_subjects))
    validation_size = int(validation_fraction * len(node_subjects))
    support_size = int(support_fraction * len(node_subjects))

    es_callback = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1)
    train_subjects, test_subjects = model_selection.train_test_split(
        node_subjects, train_size=train_size, test_size=None, stratify=node_subjects
    )
    val_subjects, test_subjects = model_selection.train_test_split(
        test_subjects, train_size=validation_size, test_size=None, stratify=test_subjects
    )
    support_subjects, test_subjects = model_selection.train_test_split(
        test_subjects, train_size=support_size, test_size=None, stratify=test_subjects
    )
    target_encoding = preprocessing.LabelBinarizer()
    train_targets = target_encoding.fit_transform(train_subjects)
    val_targets = target_encoding.transform(val_subjects)
    test_targets = target_encoding.transform(test_subjects)

    if dataset == 'elliptic':
        illicit = train_subjects.loc[lambda x: x == 1]
        labels = [train_subjects]
        for _ in range(n_oversampling - 1):
            labels += [illicit]
        train_subjects = pd.concat(labels, axis=0)
        train_targets = target_encoding.fit_transform(train_subjects)
        train_targets = utils.encoding(train_targets)
        val_targets = utils.encoding(val_targets)
        test_targets = utils.encoding(test_targets)

    generator = FullBatchNodeGenerator(G1, method='gcn')
    val_gen = generator.flow(val_subjects.index, val_targets)
    train_gen = generator.flow(train_subjects.index, train_targets)
    test_gen = generator.flow(test_subjects.index, test_targets)

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
        validation_data=val_gen,
        verbose=2,
        shuffle=False,
        callbacks=[es_callback]
    )

    test_predictions = model1.predict(test_gen)
    y_true = [np.argmax(y) for y in test_targets]
    y_pred1 = [np.argmax(y) for y in test_predictions[0]]
    accuracy1 = accuracy_score(y_true, y_pred1)

    # ----- Label-GCN -----
    if dataset == 'elliptic':
        G2 = utils.get_graph_with_label_elliptic(df_features_with_label, df_edgelist, train_subjects, val_subjects)
    else:
        G2, n_cols, n_dummies = utils.get_graph(G1.node_features(), G1.edges(), node_subjects, train_subjects, val_subjects)
    generator = FullBatchNodeGenerator(G2, method='gcn')
    train_gen = generator.flow(train_subjects.index, train_targets)
    val_gen = generator.flow(val_subjects.index, val_targets)

    if dataset == 'elliptic':
        gcn2 = LabelGCN(layer_sizes=layer_sizes, activations=['relu'] * len(layer_sizes), generator=generator, dropout=dropout,
                        label_idxs=[-1], bias=bias)
    else:
        gcn2 = LabelGCN(layer_sizes=layer_sizes, activations=['relu'] * len(layer_sizes), generator=generator, dropout=dropout,
                        label_idxs=[i for i in range(n_cols - n_dummies, n_cols)], bias=bias)
    x_inp, x_out = gcn2.in_out_tensors()
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
        validation_data=val_gen,
        verbose=2,
        shuffle=False,
        callbacks=[es_callback]
    )

    weights = model2.get_weights()
    res = []
    for frac in fracs:
        support_subset_index = np.random.choice(support_subjects.index, int(frac * len(support_subjects.index)),
                                                replace=False)
        if dataset == 'elliptic':
            G3 = utils.get_graph_with_label_elliptic(df_features_with_label, df_edgelist,
                                                     train_subjects, val_subjects, support_subset_index, test=True)
        else:
            G3, _, _ = utils.get_graph(G1.node_features(), G1.edges(), node_subjects,
                                       train_subjects, val_subjects, support_subset_index, test=True)
        Adj = G3.to_adjacency_matrix()
        node_features = G3.node_features()
        _, Adj_norm = GCN_Aadj_feats_op(node_features, Adj)
        h1 = sparse.csr_matrix.dot(Adj_norm, node_features)
        h1 = np.dot(h1, weights[0])
        h1 = K.relu(h1)
        h2 = sparse.csr_matrix.dot(Adj_norm, h1)
        h2 = np.dot(h2, weights[1])
        h2 = K.relu(h2)
        out = np.dot(h2, weights[2])
        out = K.softmax(out + weights[3]).numpy()
        y_pred = []
        for test_id in test_subjects.index:
            test_id_iloc = G3.node_ids_to_ilocs([test_id])
            y_pred.append(out[test_id_iloc])

        y_pred2 = [np.argmax(y) for y in y_pred]
        accuracy2 = accuracy_score(y_true, y_pred2)

        logger.info('frac: {}, count: {}, accuracy w/o label feature: {}'.format(frac, count, accuracy1))
        logger.info('frac: {}, count: {}, accuracy w/  label feature: {}'.format(frac, count, accuracy2))
        if dataset != 'elliptic':
            res.append([accuracy1, accuracy2])
        else:
            recall1 = recall_score(y_true, y_pred1, average='binary', pos_label=1)
            precision1 = precision_score(y_true, y_pred1, average='binary', pos_label=1)
            f11 = f1_score(y_true, y_pred1, average='binary', pos_label=1)
            recall2 = recall_score(y_true, y_pred2, average='binary', pos_label=1)
            precision2 = precision_score(y_true, y_pred2, average='binary', pos_label=1)
            f12 = f1_score(y_true, y_pred2, average='binary', pos_label=1)
            res.append([accuracy1, accuracy2, precision1, precision2, recall1, recall2, f11, f12])
    return res


def run_jobs(dataset, num_random_states, num_runs, logger, training_fraction, validation_fraction,
             support_fraction, layer_sizes, bias, learning_rate, dropout, fracs, patience, n_epochs, n_oversampling):

    random_state = np.random.RandomState(2021)
    tf_seeds = random_state.randint(0, 1000000, num_random_states)

    results = []
    count = 0
    for i in range(num_random_states):
        tf.random.set_seed(tf_seeds[i])
        for j in range(num_runs):
            count += 1
            tf.keras.backend.clear_session()
            if dataset == 'elliptic':
                results.append(run_job(count, logger, dataset, training_fraction, validation_fraction,
                                       support_fraction, layer_sizes, bias, learning_rate, dropout,
                                       fracs, patience, n_epochs, n_oversampling))
            else:
                results.append(run_job(count, logger, dataset, training_fraction, validation_fraction,
                                       support_fraction, layer_sizes, bias, learning_rate, dropout,
                                       fracs, patience, n_epochs))
    return results


def main():
    np.random.seed(1)
    dataset, nstates, nruns = utils.parse_command_line_args_transductive()

    bias, learning_rate, dropout, patience, n_epochs, n_oversampling, layer_sizes, training_fraction, \
        num_random_states, num_runs, max_val = utils.init_vars_transductive(dataset, nstates, nruns)

    validation_fraction = training_fraction
    min_val = training_fraction + validation_fraction
    support_fraction = max_val - validation_fraction - training_fraction

    if dataset == 'elliptic':
        fracs = [(s - validation_fraction - training_fraction) / support_fraction for s in [min_val, 0.5, 0.6, 0.7, max_val]]
    else:
        fracs = [(s - validation_fraction - training_fraction) / support_fraction for s in [min_val, 0.15, 0.3, 0.6, max_val]]

    logger = utils.get_logger(dataset)
    logger.info('layer_size: {}, learning rate: {}, dropout: {}, min_val: {}, max_val: {}, '
                'n_oversampling: {}, patience: {}, num_random_states: {}, num_runs: {}, training_fraction: {}'
                .format(layer_sizes, learning_rate, dropout, min_val, max_val,
                        n_oversampling, patience, num_random_states, num_runs, training_fraction))
    logger_results = utils.get_logger(dataset + '_results')

    results = run_jobs(dataset, num_random_states, num_runs, logger, training_fraction, validation_fraction,
             support_fraction, layer_sizes, bias, learning_rate, dropout, fracs, patience, n_epochs, n_oversampling)

    logger_results.info('layer_size: {}, learning rate: {}, dropout: {}, min_val: {}, max_val: {}, '
                        'n_oversampling: {}, patience: {}, num_random_states: {}, num_runs: {}, training_fraction: {}'
                        .format(layer_sizes, learning_rate, dropout, min_val, max_val,
                                n_oversampling, patience, num_random_states, num_runs, training_fraction))
    for ifrac, frac in enumerate(fracs):
        result_frac = []
        for result in results:
            result_frac.append(result[ifrac])
        if ifrac == 0:  # result w/o label doesn't change with size of support set -> only output when ifrac == 0
            mean_accuracy1, std_accuracy1 = np.mean([r[0] for r in result_frac]), np.std([r[0] for r in result_frac])
            logger_results.info('Accuracy w/o label: {} {} {}'.format(str(mean_accuracy1), u'\u00B1', std_accuracy1))
        mean_accuracy2, std_accuracy2 = np.mean([r[1] for r in result_frac]), np.std([r[1] for r in result_frac])
        if dataset == 'elliptic':
            if ifrac == 0:  # result w/o label doesn't change with size of support set -> only output when ifrac == 0
                mean_precision1, std_precision1 = np.mean([r[2] for r in result_frac]), np.std([r[2] for r in result_frac])
                mean_recall1, std_recall1 = np.mean([r[4] for r in result_frac]), np.std([r[4] for r in result_frac])
                mean_f11, std_f11 = np.mean([r[6] for r in result_frac]), np.std([r[6] for r in result_frac])
                logger_results.info('Precision w/o label: {} {} {}'.format(str(mean_precision1), u'\u00B1', std_precision1))
                logger_results.info('Recall w/o label: {} {} {}'.format(str(mean_recall1), u'\u00B1', std_recall1))
                logger_results.info('F1 score w/o label: {} {} {}'.format(str(mean_f11), u'\u00B1', std_f11))
            mean_precision2, std_precision2 = np.mean([r[3] for r in result_frac]), np.std([r[3] for r in result_frac])
            mean_recall2, std_recall2 = np.mean([r[5] for r in result_frac]), np.std([r[5] for r in result_frac])
            mean_f12, std_f12 = np.mean([r[7] for r in result_frac]), np.std([r[7] for r in result_frac])
            logger_results.info('frac: {}, Accuracy w/ label: {} {} {}'.format(frac, str(mean_accuracy2), u'\u00B1', std_accuracy2))
            logger_results.info('frac: {}, Precision w/ label: {} {} {}'.format(frac, str(mean_precision2), u'\u00B1', std_precision2))
            logger_results.info('frac: {}, Recall w/ label: {} {} {}'.format(frac, str(mean_recall2), u'\u00B1', std_recall2))
            logger_results.info('frac: {}, F1 score w/ label: {} {} {}'.format(frac, str(mean_f12), u'\u00B1', std_f12))
        else:
            logger_results.info('frac: {}, Accuracy w/ label: {} {} {}'.format(frac, str(mean_accuracy2), u'\u00B1', std_accuracy2))
    logger_results.info('------------------')


if __name__ == '__main__':
    main()
