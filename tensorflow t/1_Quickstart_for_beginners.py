#TensorFlow 2 のパッケージをダウンロードしてインストールします。プログラムに TensorFlow をインポートします。
#_future_                   python2のコードをpython3で使用できるように,いきなり3系に移行するのはちょっと心配…という場合、まず2 系 + __future__ モジュールを使うのが良さそう。__future__ モジュールは Python 2 系用のモジュール。Python 3 系に実装されている Python 2 系 と互換性の無い機能をPython 2 系で使用できるようにする。
#absolute_import            相対インポートではなく、絶対インポート優先になる。
#division                   打ち切り除算:2系では割り算の演算子 / は小数点以下切り捨てだったが、3系では小数点以下をそのまま扱える。
#print_function             3系では print は文から関数 print()になった。 
#unicode_literals           Python 3 系では文字列は基本的にすべてユニコードとして扱う。日本語等の扱いが相当楽になる。u'日本語'と書いていたものは全て'日本語'と書けば良い。逆に以前の通常の文字列を取り扱いたい場合b'日本語'のようにb''の形で書く(byte列表現)。
from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlowの名前をｔｆとする。
import tensorflow as tf

#%%#####################################   データダウンロード    #################################################################
#MNIST データセットをロードして準備します。サンプルを整数から浮動小数点数に変換します。
#データを読み出す　　データの場所は#C:\Users\tome\Anaconda3\Lib\site-packages\keras\datasets\mnist\\
mnist = tf.keras.datasets.mnist

#データを読み、トレーニングデータと実験データに分けて、各データは訓練データとラベルデータに分ける。
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#%%#####################################   データ前処理    #################################################################
#一般的な画像データ（uint8）からfloat64に変換する。ニューラルネットワークに入力するため。
#しないと、学習率を大きくなる必要性がある。
x_train, x_test = x_train / 255.0, x_test / 255.0

#%%#####################################   モデル構築    #################################################################
#層を積み重ねてtf.keras.Sequentialモデルを構築します。訓練のためにオプティマイザと損失関数を選びます。
model = tf.keras.models.Sequential([
        
    #28*28の画像データを784*1のデータに変換
    tf.keras.layers.Flatten(input_shape=(28, 28)),
  
    #第1層128個ニューラル、活性化関数relu
    tf.keras.layers.Dense(128, activation='relu'),
  
    #OVER　LOADINDを防ぐため、第2層の２０％のニューラルの値をDropoutsする
    tf.keras.layers.Dropout(0.2),
  
    #第３層10個ニューラル、活性化関数softmax　確率を取り出す
    tf.keras.layers.Dense(10, activation='softmax')
])

#%%#####################################   訓練ステップ設定    #################################################################
#optimizer：勾配降下estimation优化器、普通はadamを使う
model.compile(optimizer='adam',
              
              #loss：目的関数：　　現状と目標の差別
              loss='sparse_categorical_crossentropy',
              
              #metrics：評価関数、正確率
              metrics=['accuracy'])

#%%#####################################   訓練   #################################################################
#モデルを訓練してから評価します。モデル最適化、epochsは回数
model.fit(x_train, y_train, epochs=3)

#%%#####################################   予測と結果評価   #################################################################
 
#%%#モデル評価　　#verboseは==============================です。一般には２をする。
model.evaluate(x_test,  y_test, verbose=2)