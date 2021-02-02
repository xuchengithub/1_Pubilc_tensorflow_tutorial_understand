# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 11:17:50 2019

@author: tome
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import functools

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
#データのアドレス
TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"
############################アドレスから"train.csv"の名前のファイルをダウンロードして、読み込み########################################
train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)

# 有効数字を３にする。#デフォルトは８である。
np.set_printoptions(precision=3, suppress=True) 

#データを観察して、パラメーターである単語を記録
CSV_COLUMNS = ['survived', 'sex', 'age', 'n_siblings_spouses', 'parch', 'fare', 'class', 'deck', 'embark_town', 'alone']
 
dataset = tf.data.experimental.make_csv_dataset(..., column_names=CSV_COLUMNS,...)
#1つデータは行であることを示す。
dataset = tf.data.experimental.make_csv_dataset(
  ...,
  select_columns = columns_to_use, 
  ...)
#ラベルを示す。
LABEL_COLUMN = 'survived'
LABELS = [0, 1]  

def get_dataset(file_path):
  dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=12, #見やすいため
      label_name=LABEL_COLUMN,
      na_value="?",
      num_epochs=1,
      ignore_errors=True)
  return dataset

raw_train_data = get_dataset(train_file_path)
raw_test_data = get_dataset(test_file_path)
#最初のデータ
examples, labels = next(iter(raw_train_data))
print("EXAMPLES: \n", examples, "\n")
print("LABELS: \n", labels)
############################################データ前処理###############################################################
##########################################定義分類列のラベルの処理#########################################################
CATEGORIES = {
    'sex': ['male', 'female'],
    'class' : ['First', 'Second', 'Third'],
    'deck' : ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    'embark_town' : ['Cherbourg', 'Southhampton', 'Queenstown'],
    'alone' : ['y', 'n']
}

#分類列の情報をcategorical_columnsに加えて
categorical_columns = []
for feature, vocab in CATEGORIES.items():
  cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
        key=feature, vocabulary_list=vocab)
  categorical_columns.append(tf.feature_column.indicator_column(cat_col))

#
categorical_columns
##########################################定義数字列のラベルの処理#########################################################
#数字列のラベルを標準化
def process_continuous_data(mean, data): 
  data = tf.cast(data, tf.float32) * 1/(2*mean)
  return tf.reshape(data, [-1, 1])
#事前計算
MEANS = {
    'age' : 29.631308,
    'n_siblings_spouses' : 0.545455,
    'parch' : 0.379585,
    'fare' : 34.385399
}

numerical_columns = []

for feature in MEANS.keys():
  num_col = tf.feature_column.numeric_column(feature, normalizer_fn=functools.partial(process_continuous_data, MEANS[feature]))
  numerical_columns.append(num_col)
  
MEANS = {
    'age' : 29.631308,
    'n_siblings_spouses' : 0.545455,
    'parch' : 0.379585,
    'fare' : 34.385399
}

numerical_columns = []

for feature in MEANS.keys():
  num_col = tf.feature_column.numeric_column(feature, normalizer_fn=functools.partial(process_continuous_data, MEANS[feature]))
  numerical_columns.append(num_col)
  
#
numerical_columns
##############################################全体処理#################################
preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns+numerical_columns)
##############################################モデルの構築#################################
model = tf.keras.Sequential([
  preprocessing_layer,
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])
##############################################結果#################################
train_data = raw_train_data.shuffle(500)
test_data = raw_test_data

model.fit(train_data, epochs=20)

test_loss, test_accuracy = model.evaluate(test_data)

print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))

predictions = model.predict(test_data)

# 結果を示す。
for prediction, survived in zip(predictions[:10], list(test_data)[0][1][:10]):
  print("Predicted survival: {:.2%}".format(prediction[0]),
        " | Actual outcome: ",
        ("SURVIVED" if bool(survived) else "DIED"))