# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 11:01:33 2019

@author: tome
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
from tensorflow import keras
tf.__version__

#%%############################データダウンロード###############################################################################
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

#%%############################１０００個データを読み込み###############################################################################
#１０００個のデータを取り出す
train_labels = train_labels[:1000]
test_labels = test_labels[:1000]
#０－１に変化
#shape=-1いわゆる，1列になる
train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

#%%############################モデルの構築###############################################################################
# 短いシーケンシャルモデルを返す関数
def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
  ])
  
  model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  
  return model

#基本的なモデルのインスタンスを作る
#モデルを見る
model = create_model()
model.summary()

#%%############################モデルを保存設定###############################################################################
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# チェックポイントコールバックを作る　　　モデル保存用
#%%############################モデルを保存用のコード,モデル訓練##############################################################################
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 save_weights_only=True,
                                                 verbose=1)

model = create_model()

model.fit(train_images, train_labels,  epochs = 10, 
          validation_data = (test_images,test_labels),
          callbacks = [cp_callback])  # 訓練にコールバックを渡す

# オプティマイザの状態保存についての警告が表示されるかもしれません。
# これらの警告は（このノートブックで発生する同様な警告を含めて）
# 古い用法を非推奨にするためのもので、無視して構いません。

#%%############################ロード用のモデルを作成，###############################################################################
model = create_model()

#%%############################モデルを保存用のコード###############################################################################
#モデル評価，結果はよくないことを示す．
loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))
#モデルの重みを入力
model.load_weights(checkpoint_path)
#モデル評価
loss,acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

#%%#新しいcallbacks
############################モデルにパラメータを入力###############################################################################

# ファイル名に(`str.format`を使って)エポック数を埋め込みます
#checkpoint_pathは保存の場所
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # 重みを5エポックごとに保存しますverboseを１のとき，プログラムバーで表示する．
    period=5)

model = create_model()
model.fit(train_images, train_labels,
          epochs = 50, callbacks = [cp_callback],
          validation_data = (test_images,test_labels),
          verbose=0)

latest = tf.train.latest_checkpoint(checkpoint_dir)
latest

model = create_model()
model.load_weights(latest)
loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

#%%#新しいcallbacks
############################手動でモデル構築###############################################################################

# 重みの保存
model.save_weights('./checkpoints/my_checkpoint')

# 重みの復元
model = create_model()
#重みを読み込む
model.load_weights('./checkpoints/my_checkpoint')
#モデル評価
loss,acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

#%%############################全モデル保存###############################################################################
model = create_model()

model.fit(train_images, train_labels, epochs=5)

# モデル全体を１つのHDF5ファイルに保存します。
model.save('my_model.h5')

#%%############################全モデル構築###############################################################################
# 重みとオプティマイザを含む全く同じモデルを再作成
new_model = keras.models.load_model('my_model.h5')
new_model.summary()

loss, acc = new_model.evaluate(test_images,  test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

#%%############################全モデル構築###############################################################################
model = create_model()

model.fit(train_images, train_labels, epochs=5)

import time
saved_model_path = "./saved_models/{}".format(int(time.time()))

tf.keras.experimental.export_saved_model(model, saved_model_path)
saved_model_path

new_model = tf.keras.experimental.load_from_saved_model(saved_model_path)
new_model.summary()

model.predict(test_images).shape

# モデルを評価する前にコンパイルする必要があります。
# モデルをデプロイするだけであればこのステップは不要です。

new_model.compile(optimizer=model.optimizer,  # ロードしてあったオプティマイザを保持
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# モデルを評価します。
loss, acc = new_model.evaluate(test_images,  test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))