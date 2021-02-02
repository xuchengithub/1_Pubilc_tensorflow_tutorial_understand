# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 14:16:11 2019

@author: tome
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import pandas as pd

#%%###############################  バラメータとラベル    #################################
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

#%%###############################  アドレス　　　　　   #################################
train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

#%%###############################  　データをダウンロード　　　　   #################################
train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

#%%###############################  　データ処理　　　   #################################
#処理前データ
train.head()

train_y = train.pop('Species')
test_y = test.pop('Species')
#処理後データ
#SPECIESをデータの中から削除
train.head()

#%%###############################  　　　datasetの例　   #################################
def input_evaluation_set():
    features = {'SepalLength': np.array([6.4, 5.0]),
                'SepalWidth':  np.array([2.8, 2.3]),
                'PetalLength': np.array([5.6, 3.3]),
                'PetalWidth':  np.array([2.2, 1.0])}
    labels = np.array([2, 1])
    return features, labels

#%%###############################  　　　dataset整える　   #################################
def input_fn(features, labels, training=True, batch_size=256):
    """An input function for training or evaluating"""
    # データを整える
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # テスト時、データを混乱して、コピーする。
    if training:
        dataset = dataset.shuffle(1000).repeat()
    
    return dataset.batch(batch_size)

#%%###############################  　　　モデルの構築   ################################# 
# 第一層の構築
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
     
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # 隠れる層は30 和 10.
    hidden_units=[30, 10],
    # クラスは三つ
    n_classes=3)

#%%###############################  　　　モデルの訓練   #################################
classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True),
    steps=5000)

eval_result = classifier.evaluate(
    input_fn=lambda: input_fn(test, test_y, training=False))
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

#%%###############################  　　　モデルの予測   #################################
expected = ['Setosa', 'Versicolor', 'Virginica']
predict_x = {
    'SepalLength': [5.1, 5.9, 6.9],
    'SepalWidth': [3.3, 3.0, 3.1],
    'PetalLength': [1.7, 4.2, 5.4],
    'PetalWidth': [0.5, 1.5, 2.1],
}
#入力の訓練データのラベルを取り消し
def input_fn(features, batch_size=256):
    """An input function for prediction."""
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

predictions = classifier.predict(
    input_fn=lambda: input_fn(predict_x))

for pred_dict, expec in zip(predictions, expected):
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('Prediction is "{}" ({:.1f}%), expected "{}"'.format(
        SPECIES[class_id], 100 * probability, expec))