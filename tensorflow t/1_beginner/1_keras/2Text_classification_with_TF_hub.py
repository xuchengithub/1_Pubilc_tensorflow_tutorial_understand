# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 15:09:46 2019
@author: tome
"""
#学習ポイントはembeddingモデルの使用によって、データをベクトル化
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

#%%#####################################   データダウンロード    #################################################################
#データが少ない場合、データを６０％と４０％に分ける。
train_validation_split = tfds.Split.TRAIN.subsplit([6, 4])
#tfdsはTensorFlowオンラインデータ
(train_data, validation_data), test_data = tfds.load(
    name="imdb_reviews", 
    #データを分ける
    split=(train_validation_split, tfds.Split.TEST),
    #as_supervisedは`bool`のデータを出力する。Trueの場合は（input, label）として、出力する。Falseの場合はtupleで出力する。
    #asupurvised=Falseの場合、tfds.loadは辞書<mage、lable>をダウンロードして、この辞書はパラメータとしてformatmaxmaple()に伝えます。
    as_supervised=True)
#10個ごとに、train_dataを取り出す、iterはシーズマリー、nextは次のです。
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
train_examples_batch
train_labels_batch

#%%###############################   モデル構築    #######################################################################
#構築した文字embeddingモデルをダウンロードする。embeddingでデータをベクトル化
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[], 
                           #stringデータが欲しい、trainableを０、１データにする。
                           dtype=tf.string, trainable=True)
#データの.",不要
hub_layer(train_examples_batch[:3])

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
#結果は０１のため、sigmoidを使用
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.summary()

#%%#####################################   訓練ステップ設定    #################################################################
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#%%#####################################   訓練        ###############################################################################
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=20,
                    validation_data=validation_data.batch(512),
                    verbose=1)
#verbose: 0 or 1. Verbosity mode.プログレスバー
results = model.evaluate(test_data.batch(512), verbose=2)
#zip() 関数はmodel.metrics_namesとresultsを合わせて出力する。
for name, value in zip(model.metrics_names, results):
    #プリント['loss', 'accuracy']
    print("%s: %.3f" % (name, value))