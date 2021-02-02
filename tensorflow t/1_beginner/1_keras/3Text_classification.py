# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 16:46:11 2019

@author: tome
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
print(tf.__version__)
#tensorflowのバッジョを示す。
imdb = keras.datasets.imdb

#%%#####################################   データダウンロード    #################################################################
#num_wordsはあまり使えない単語を削除
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

#%%#####################################   データ観察　　　    #################################################################
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
print(train_data[0])

#%%#####################################   データを文字に変換    #################################################################
#1つのデータの長さは違うことが分かりました。
len(train_data[0]), len(train_data[1])
#整数を単語にインデックスマッピングする辞書
word_index = imdb.get_word_index()
#[1, 14, 22, 16, 43.....
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

decode_review(train_data[0])

#%%#####################################   データ前処理    #################################################################
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)
#pad_sequences映像の評価の文字数の長さは同じでなければならない
#maxlen: 整数，映像の文字数は２５６にする。
#padding: string，'pre' もしくは 'post' ，文字列の前もしくは後ろに単語を加える。
#value: 浮点数，加える単語
test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)
#長さを計算、同じになる。
len(train_data[0]), len(train_data[1])
#変換後の評価を見る。
print(train_data[0])

#%%#####################################   モデル構築   #################################################################
#映像評価の単語数
vocab_size = 10000
model = keras.Sequential()

#16次元の値を出力します.#vocab_size最大値、Embeddingは正整数をベクトルにする。
model.add(keras.layers.Embedding(vocab_size, 16))
#次元減少、長さを求めるような感じ
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.summary()

#%%#####################################   訓練ステップ設定    #################################################################
model.compile(optimizer='adam',
              loss='binary_crossentropy',#int
              metrics=['accuracy'])

#%%##################################### trainからtrain set とtrain dev setに分けて     #################################################################
#訓練モデルの信頼性,train dev setとtrain setの分布が違う。原因はtrain dev setデータにtrain setにいないデータがあること。
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

#%%#####################################   訓練    #########################################################################
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,#20 best 
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

#%%######################################  最終結果を評価 ########################################################
results = model.evaluate(test_data,  test_labels, verbose=2)
print(results)

#%%######################################  overloadを評価するため、accurayとlossの評価図を作成 ########################################################
history_dict = history.history
history_dict.keys()

import matplotlib.pyplot as plt

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()