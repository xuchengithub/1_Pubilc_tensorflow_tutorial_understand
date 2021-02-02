# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 09:40:56 2019

@author: tome
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
from IPython.display import clear_output
from matplotlib import pyplot as plt

#%%#####################################データを読み込み##########################################
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')
#%%#####################################データを読み込み##########################################
#!pip install -q tf-nightly-2.0-preview
#使うかどうか実験
import tensorflow as tf
tf.random.set_seed(123)

#%%#####################################データを観察##########################################

dftrain.head()

dftrain.describe()

dftrain.shape[0], dfeval.shape[0]

dftrain.age.hist(bins=20)
plt.show()

dftrain.sex.value_counts().plot(kind='barh')
plt.show()

dftrain['class'].value_counts().plot(kind='barh')
plt.show()

dftrain['embark_town'].value_counts().plot(kind='barh')
plt.show()

pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')
plt.show()
#%%#####################################データを変更して##########################################

fc = tf.feature_column
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

def one_hot_cat_column(feature_name, vocab):
  return tf.feature_column.indicator_column(
      tf.feature_column.categorical_column_with_vocabulary_list(feature_name,
                                                 vocab))
feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  # Need to one-hot encode categorical features.
  vocabulary = dftrain[feature_name].unique()
  feature_columns.append(one_hot_cat_column(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name,
                                           dtype=tf.float32))
#%%#####################################ニューラルネットワークに入力できるデータを作成##########################################
example = dict(dftrain.head(1))
class_fc = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list('class', ('First', 'Second', 'Third')))

#%%#####################################ニューラルネットワークに入力できるデータを観察##########################################
print('Feature value: "{}"'.format(example['class'].iloc[0]))
print('One-hot encoded: ', tf.keras.layers.DenseFeatures([class_fc])(example).numpy())
tf.keras.layers.DenseFeatures(feature_columns)(example).numpy()

#%%#####################################訓練に関する設定#############################################################
NUM_EXAMPLES = len(y_train)
def make_input_fn(X, y, n_epochs=None, shuffle=True):
  def input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
    if shuffle:
      dataset = dataset.shuffle(NUM_EXAMPLES)
 
    dataset = dataset.repeat(n_epochs)
    dataset = dataset.batch(NUM_EXAMPLES)
    return dataset
  return input_fn

# 训练与评估的输入函数。
train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, shuffle=False, n_epochs=1)

linear_est = tf.estimator.LinearClassifier(feature_columns)

#%%#####################################訓練#############################################################
linear_est.train(train_input_fn, max_steps=100)

#%%#####################################評価#############################################################
result = linear_est.evaluate(eval_input_fn)
clear_output()
print(pd.Series(result))
#%%#####################################Boosted Treesモデル#############################################################

n_batches = 1
est = tf.estimator.BoostedTreesClassifier(feature_columns,
                                          n_batches_per_layer=n_batches)

# 一旦建立了指定数量的树，模型将停止训练，
# 而不是基于训练步数。
est.train(train_input_fn, max_steps=100)

# 评估。
result = est.evaluate(eval_input_fn)
clear_output()
print(pd.Series(result))
#%%#####################################Boosted Treesモデル#############################################################

pred_dicts = list(est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])

probs.plot(kind='hist', bins=20, title='predicted probabilities')
plt.show()

from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_eval, probs)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.xlim(0,)
plt.ylim(0,)
plt.show()