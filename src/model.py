# -*- coding: utf-8 -*-

# Author: Nathan Mori <nathanmori@gmail.com>

import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import seaborn
from eda.eda import *
from eda.feat_engr import FeatureEngineer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    confusion_matrix
from datetime import datetime
from sys import argv
import csv


if __name__ == '__main__':

    time_str = get_time_str()
    path = get_path()

    shard = 'shard' in argv
    df_train, df_test = get_data(shard=shard, drop_category=True)

    # Convert dataframes to arrays
    y_train = df_train.pop('Priority').values

    # Split train data into two samples - one to fit the model, and
    df_fit, df_eval, y_fit, y_eval = train_test_split(df_train, y_train)

    # Pipe FeatureEngineer and RandomForestClassifier together
    pipe = Pipeline([('feng', FeatureEngineer()),
                     ('rfc', RandomForestClassifier())])

    # Search for best parameters for model on fit data
    param_grid = {'feng__dummy_PdDistrict': [False],
                  'feng__include_Mean': [False],
                  'feng__include_Intersection': [False],
                  'rfc__criterion': ['gini', 'entropy'],
                  'rfc__max_features': ['auto', None],
                  'rfc__max_depth': [None, 10],
                  'rfc__min_samples_split': [2, 5, 10],
                  'rfc__n_jobs': [-1],
                  'rfc__class_weight': ['balanced']}
    grid = GridSearchCV(pipe, param_grid=param_grid, n_jobs=-1)
    grid.fit(df_fit, y_fit)

    # Evaluate model on eval data
    eval_preds = grid.predict(df_eval)
    accuracy = accuracy_score(y_eval, eval_preds)
    precisions = precision_score(y_eval, eval_preds, average=None)
    recalls = recall_score(y_eval, eval_preds, average=None)
    confmat = confusion_matrix(y_eval, eval_preds, labels=[1, 2, 3, 4])
    # C[i, j] = # obs with true label = i, pred label = j

    # Save best model score and info
    results_file = ('%s/output/shard_%s_model_results.txt' % (path, time_str)
                    if shard else
                    '%s/output/%s_model_results.txt' % (path, time_str))
    with open(results_file, 'wb') as f:

        f.write('\nGRID SCORE\n----------\n  %f\n' % grid.best_score_)

        f.write('\nACCURACY\n--------\n  Total: %f\n' % accuracy)
        f.write('\nPRECISION\n---------\n')
        for priority, precision in enumerate(precisions, start=1):
            f.write('  Priority %d: %f\n' % (priority, precision))
        f.write('\nRECALL\n------\n')
        for priority, recall in enumerate(recalls, start=1):
            f.write('  Priority %d: %f\n' % (priority, recall))
        f.write('\nCONFUSION MATRIX\n----------------\n')
        f.write('                   prediction\n')
        f.write('         1         2         3         4\n')
        for i, (char, line) in enumerate(zip('true', confmat), start=1):
            f.write('  %s  %d  ' % (char, i))
            for elem in line:
                f.write(str(elem).ljust(10))
            f.write('\n')

        f.write('\nPARAMS\n------\n')
        for param, best_value in grid.best_params_.iteritems():
            f.write('  %s: %s\n' % (param, str(best_value)))

        f.write('\nFEATURE IMPORTANCES\n-------------------\n')
        f.write('  Feature'.ljust(30))
        f.write('Importance'.ljust(15))
        f.write('Normalized\n')
        features = \
            grid.best_estimator_.named_steps['feng'].transform(df_fit).columns
        importances = \
            grid.best_estimator_.named_steps['rfc'].feature_importances_
        sorted_args = np.argsort(importances)[::-1]
        imps = importances[sorted_args]
        for feature, importance \
            in zip(features[sorted_args],
                   importances[sorted_args]):
            f.write(('  %s:' % feature).ljust(30) +
                    ('%f' % importance).ljust(15) +
                    '%f\n' % (importance / imps[0]))

    # Plot confusion matrix
    ticks = np.linspace(1, 4, num=4)
    fig, ax = plt.subplots(figsize=(12, 12))
    plt_confmat = np.zeros((5, 5))
    plt_confmat[1:, 1:] = confmat
    plt.imshow(plt_confmat, interpolation='none')
    plt.colorbar()
    plt.xticks(ticks, fontsize=20)
    plt.yticks(ticks, fontsize=20)
    ax.set_xlabel('Predicted Priority', fontsize=20)
    ax.set_ylabel('True Priority', fontsize=20)
    ax.set_xlim((.5, 4.5))
    ax.set_ylim((.5, 4.5))
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.invert_yaxis()
    fig.suptitle('Confusion Matrix', fontsize=30)
    plt.savefig('%s/output/shard_%s_confusion_matrix' % (path, time_str)
                if shard else
                '%s/output/%s_confusion_matrix' % (path, time_str))
    if 'show' in argv:
        plt.show()
    else:
        plt.close()

    # Plot feature importances
    fig = plt.figure(figsize=(12, 12))
    n_feats = len(features)
    x_ind = np.arange(n_feats)
    plt.barh(x_ind, imps[::-1]/imps[0], height=.8, align='center')
    plt.ylim(x_ind.min() - .5, x_ind.max() + .5)
    plt.yticks(x_ind, features[::-1], fontsize=20)
    plt.xticks(np.arange(0, 1.1, 0.2), fontsize=20)
    plt.title('Feature Importances',
              fontsize=30)
    plt.gcf().tight_layout()
    plt.savefig('%s/output/shard_%s_feature_importances' % (path, time_str)
                if shard else
                '%s/output/%s_feature_importances' % (path, time_str))
    if 'show' in argv:
        plt.show()
    else:
        plt.close()

    # Use best model to predict test data and save predictions to .csv
    df_test['Prediction'] = grid.predict(df_test)
    fname = ('%s/output/shard_%s_test_preds.csv' % (path, time_str)
             if shard else
             '%s/output/%s_test_preds.csv' % (path, time_str))
    df_test.to_csv(fname,
                   columns=['Prediction'],
                   header=True,
                   index_label='Id',
                   encoding='utf-8')
