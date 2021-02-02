# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 14:44:21 2019

@author: tome
"""
#TensorFlow 2 のパッケージをダウンロードしてインストールします。プログラムに TensorFlow をインポートします。
#コード１に説明しました。
from __future__ import absolute_import, division, print_function, unicode_literals#...
# TensorFlow and tf.keras
import tensorflow as tf#...
#コード１に説明しました。
from tensorflow import keras#...
# Helper libraries
import numpy as np#...
#figureするため
import matplotlib.pyplot as plt#...
#コード１に説明しました。
print(tf.__version__)#...
#コード１に説明しました。

#%%#####################################   データダウンロード    #################################################################
#MNIST データセットをロードして準備します。コード１にコマンドしました
#ここでは、60,000枚の画像を訓練に、10,000枚の画像を、ネットワークが学習した画像分類の正確性を評価するのに使います。TensorFlowを使うと、下記のようにFashion MNISTのデータを簡単にインポートし、ロードすることが出来ます。
fashion_mnist = keras.datasets.fashion_mnist#...
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()#...
#クラスのラベルの名前
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',#...
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#%%#####################################   データ観察    #################################################################
#60000枚图像を訓練データとする。#(60000, 28, 28)
train_images.shape
#長さ
len(train_labels)
train_labels
#10000枚图像をテストデータとする。
test_images.shape
len(test_labels)

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
#網なし
plt.grid(False)
plt.show()

#%%#####################################   データ前処理   #################################################################
#float64に変換
train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

#%%#####################################   モデルの構築   #################################################################
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    #ラベルが2個以上の場合はsoftmax
    keras.layers.Dense(10, activation='softmax')
])
    
#%%#####################################   訓練ステップ設定    #################################################################    
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#%%#####################################   訓練   #################################################################   
model.fit(train_images, train_labels, epochs=10)

#%%#####################################   結果訓練表示   ################################################################# 
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

#%%#####################################   結果予測   ################################################################# 
predictions = model.predict(test_images)
predictions[0]
#確率が一番多いやつを推定値とする。
np.argmax(predictions[0])
test_labels[0]#真値

#%%#####################################   結果を目で確かめる   ################################################################# 
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  #予測正しい場合は青、間違う場合は赤
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
  
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()


i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

#テストイメージを使って、予測し、目で確認、青は正しい、赤は間違い
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

# 最後に、訓練済みモデルを使って1枚の画像に対する予測を行います
img = test_images[1]

print(img.shape)

#tf.keras モデルは、サンプルの中のバッチ（batch）あるいは「集まり」について予測を行うように作られています。そのため、1枚の画像を使う場合でも、リスト化する必要があります。.
img = (np.expand_dims(img,0))

print(img.shape)
#予測を行います。
predictions_single = model.predict(img)

print(predictions_single)

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

np.argmax(predictions_single[0])