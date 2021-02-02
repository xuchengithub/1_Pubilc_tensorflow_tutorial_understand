#TensorFlow 2 のパッケージをダウンロードしてインストールします。プログラムに TensorFlow をインポートします。
from __future__ import absolute_import, division, print_function, unicode_literals
#コード１にコマンドしました
import tensorflow as tf
#コード１にコマンドしました。
#Conv2Dは畳み込み層
from tensorflow.keras.layers import Dense, Flatten, Conv2D
#コード１にコマンドしました
from tensorflow.keras import Model

#%%#####################################   データダウンロード    #################################################################
#MNIST データセットをロードして準備します。コード１にコマンドしました
mnist = tf.keras.datasets.mnist
#コード１にコマンドしました
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#%%#####################################   データ前処理    #################################################################
#コード１にコマンドしました
x_train, x_test = x_train / 255.0, x_test / 255.0

#チャンネルを加える、Conv2Dの入力は４次元のデータ
#Conv2Dを使うため、x_trainは４次元しなければならない。１チャンネルは画像の番号、２チャンネルはｘ軸、３チャンネルはｙ軸、４チャンネルはchannel、grayの場合１、RGB等の場合は２

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

#データセットをシャッフルし、バッチ化するために tf.data を使います
#arrayをtensorに変換、毎回32個データを取り出す、
#shuffle 機械学習においてデータを混乱させる必要があります。違うデータで、訓練効率向上#shuffleを大きさはデータの大きさより大きいに設定する必要,
#しかし、データが60000個、shuffleを10000にする場合、最初の10000個データからデータを取って、ランダムに一つデータを取り出した。そして、ランダムに取り出したデータの代わりに、10001番のデータを入れて、補充する。
#shuffleとbatchの順番は大事、順番が変わると、batchをshuffleする。
train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)
#arrayをtensorに変換、毎回32個データを取り出す、テストデータであるため、データを混乱させる必要がない。
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


#%%#####################################   モデル構築    #################################################################
class MyModel(Model):#MyModelは派生クラス、Modelは基本クラス
    def __init__(self):#派生クラスのコンストラクタ関数
        super(MyModel, self).__init__()#基本クラスのコンストラクタ関数
        #channalは３２、32個フィルタ、サイズ３＊３
        self.conv1 = Conv2D(32, 3, activation='relu')
        #コード１にコマンドしました
        self.flatten = Flatten()
        #コード１にコマンドしました
        self.d1 = Dense(128, activation='relu')
        #コード１にコマンドしました
        self.d2 = Dense(10, activation='softmax')
        
        #オリジナルのレイヤーでマスキングをサポートしない限り，第1引数である入力テンソルがcallに渡されることに気を付けてください．
    def call(self, x):
        #定義した各関数を使う
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)
#コード１にコマンドしました
model = MyModel()
#目的関数：　　現状と目標の差別

#%%#####################################   訓練ステップ設定    #################################################################
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
#勾配降下
optimizer = tf.keras.optimizers.Adam()
#評価関数、lossとaccuracy　　　　平均lossと正確率の計算
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
    #with 内部の計算を記録#GradientTape とは勾配を求めるためのクラスです。精度の良い予測器をつくるには、適当に選んだパラメータで予想と、実際の結果を比較して、差が出来るだけ少なくなるようにパラメータを調整して行きます。このときにパラメータをどれだけ増やしたり減らしたりすれば良いのかを決めるのが勾配です。
    with tf.GradientTape() as tape:
        #modelで、結果を予測
        predictions = model(images)
        #予測と真値の差を計算
        loss = loss_object(labels, predictions)
        #微分計算、xはloss、重みは model.trainable_variables
    gradients = tape.gradient(loss, model.trainable_variables)
    #勾配降下 optimizerで重みを更新
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)
  
@tf.function
def test_step(images, labels):
    #モデルで予測
    predictions = model(images)
    #loss計算
    t_loss = loss_object(labels, predictions)
    #平均loss
    test_loss(t_loss)
    #正確率
    test_accuracy(labels, predictions)
  
EPOCHS = 5

#%%#####################################   訓練   #################################################################
for epoch in range(EPOCHS):
    for images, labels in train_ds:
        train_step(images, labels)
    
    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)
        
#%%#####################################   予測と結果評価   #################################################################
    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print (template.format(epoch+1,
                         train_loss.result(),
                         train_accuracy.result()*100,
                         test_loss.result(),
                         test_accuracy.result()*100)) 