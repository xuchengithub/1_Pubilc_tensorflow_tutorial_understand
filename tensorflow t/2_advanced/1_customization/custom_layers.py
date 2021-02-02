# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 13:02:58 2019

@author: tome
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
#ｇｔｐ使うかどうか
print(tf.test.is_gpu_available())
# tf.keras.layers パッケージの中では、レイヤーはオブジェクトです。
# レイヤーを構築するためにすることは、単にオブジェクトを作成するだけです。
# ほとんどのレイヤーでは、最初の引数が出力の次元あるいはチャネル数を表します。
layer = tf.keras.layers.Dense(100)
# 入力の次元数は多くの場合不要となっています。それは、レイヤーが最初に使われる際に
# 推定可能だからです。ただし、引数として渡すことで手動で指定することも可能です。
# これは複雑なモデルを構築する場合に役に立つでしょう。
layer = tf.keras.layers.Dense(10, input_shape=(None, 5))
# レイヤーを使うには、単純にcallします。
layer(tf.zeros([10, 5]))
# レイヤーにはたくさんの便利なメソッドがあります。例えば、`layer.variables`を使って
# レイヤーのすべての変数を調べることができます。訓練可能な変数は、 `layer.trainable_variables`
# でわかります。この例では、全結合レイヤーには重みとバイアスの変数があります。

layer.variables
# これらの変数には便利なアクセサを使ってアクセス可能です。layer.kernel, layer.bias
class MyDenseLayer(tf.keras.layers.Layer):
  def __init__(self, num_outputs):
    super(MyDenseLayer, self).__init__()
    self.num_outputs = num_outputs

  def build(self, input_shape):
    self.kernel = self.add_variable("kernel",
                                    shape=[int(input_shape[-1]),
                                           self.num_outputs])

  def call(self, input):
    return tf.matmul(input, self.kernel)

layer = MyDenseLayer(10)
_ = layer(tf.zeros([10, 5])) # Calling the layer `.builds` it.
print([var.name for var in layer.trainable_variables])

class ResnetIdentityBlock(tf.keras.Model):
  def __init__(self, kernel_size, filters):
    super(ResnetIdentityBlock, self).__init__(name='')
    filters1, filters2, filters3 = filters

    self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
    self.bn2a = tf.keras.layers.BatchNormalization()

    self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
    self.bn2b = tf.keras.layers.BatchNormalization()

    self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
    self.bn2c = tf.keras.layers.BatchNormalization()

  def call(self, input_tensor, training=False):
    x = self.conv2a(input_tensor)
    x = self.bn2a(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2b(x)
    x = self.bn2b(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2c(x)
    x = self.bn2c(x, training=training)

    x += input_tensor
    return tf.nn.relu(x)


block = ResnetIdentityBlock(1, [1, 2, 3])

_ = block(tf.zeros([1, 2, 3, 3])) 

block.layers

len(block.variables)

block.summary()

my_seq = tf.keras.Sequential([tf.keras.layers.Conv2D(1, (1, 1),
                                                    input_shape=(
                                                        None, None, 3)),
                             tf.keras.layers.BatchNormalization(),
                             tf.keras.layers.Conv2D(2, 1,
                                                    padding='same'),
                             tf.keras.layers.BatchNormalization(),
                             tf.keras.layers.Conv2D(3, (1, 1)),
                             tf.keras.layers.BatchNormalization()])
my_seq(tf.zeros([1, 2, 3, 3]))

my_seq.summary()