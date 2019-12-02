---
excerpt: "(1) tensorflow"
header:
  overlay_image: /assets/images/2computers.jpg
  overlay_filter: 0.5 # same as adding an opacity of 0.5 to a black background
  caption: "Photo credit: [**Unsplash**]"
  actions:
    - label: "Reference"
      url: "https://www.tensorflow.org/"
title: "[Tensorflow] Deep learning"
date: 2019-11-29 00:00:00 -0400
categories: deep_learning
tags: deep_learning 
gallery1:
  - url: /assets/images/fashion.JPG
    image_path: assets/images/fashion.JPG
    alt: "placeholder image"
gallery2:
  - url: /assets/images/fashion25.JPG
    image_path: assets/images/fashion25.JPG
    alt: "placeholder image"
gallery3:
  - url: /assets/images/denseroc.JPG
    image_path: assets/images/denseroc.JPG
    alt: "placeholder image"   
gallery4:
  - url: /assets/images/fashion_predict.JPG
    image_path: assets/images/fashion_predict.JPG
    alt: "placeholder image"       
---

---
**본 과정은 제가 파이썬 수업을 들으면서 배운 내용을 복습하는 과정에서 적어본 것입니다.<br> 틀린 부분이 있다면 댓글에 남겨주시면 고치도록 하겠습니다.<br> 확실하지 않은 내용은 '(?)'을 함께 적었으니 그 내용을 아신다면 댓글에 남겨주시면 감사하겠습니다.** 
{: .notice--warning}
--- 

# Deep learning  

- 평균적으로 unit 개수/ layer가 많을수록 성능은 좋습니다. 

## Tensorflow 

텐서플로우는 딥러닝을 위해 구글에서 제공하는 프레임워크입니다. 다시 말해, 어느 누구나 사용할수 있는 머신러닝 오픈소스 라이브러리입니다.
텐서 기반으로 artificial 뉴럴 네트워크 계열 프로그래밍 하는 것에 최적화되어 있는 라이브러리입니다.

Tensor = Multidimensional Arrays = Data (다차원 배열)

텐서는 가속기 메모리(GPU, TPU와 같은)에서 사용할 수 있습니다.
텐서는 불변성(immutable)을 가집니다.

텐서플로우 이용하면 처음부터 데이터를 넘파이가 아닌 텐서로 만들어 놓는게 좋습니다.


{% capture notice-2 %}
tensor는 assign을 통해서 값을 바꿀 수 있다. (tesnfor)상수 개념이 있으면 코딩하기가 더 귀찮지만, 디버깅하기가 좋고 값을 일부러 체크할 필요가 없다. 

numpy와 같이 mutable은 언제 바뀌는지 모르기 때문에 디버깅하기가 힘들다. 

tensor board: 구조를 잡아놓고 compile 를 시키면 빠르게 계산할 수 있다. 

명령형 패러다임 방식/ eagerly executing method 
tensorflow2에서 이 방식으로 바뀌었다. 전에는 C처럼 compile 해줘야만 보였다.
{% endcapture %}

<div class="notice">{{ notice-2 | markdownify }}</div>


```python
import tensorflow as tf
tf.__version__
# '2.0.0'
# 쿠다 지원 그래픽 카드 
# 쿠다 버전/ 설치/ mac에서 사용 불가 
```

{% capture notice-2 %}
### [Tensorflow 구현하는 4가지 방식]

- keras 3가지

1. 누가 만들어놨던 클래스 그대로 가져오는 방식 (초보자용)

2. 함수형 패러다임 이용하는 방식 

3. 상속해서 남이 만들어 놓은 클래스 가져오는 방식 (전문가용)

- tensorflow 2가지 -> 1가지 

    estimator 쓰는 것은 더 이상 사용하지 말라고해서 결국 tensorflow 1가지만 배울 것이다.

1. Tensor 그자체로 처음부터 다 하는 방식 
{% endcapture %}

<div class="notice">{{ notice-2 | markdownify }}</div>


{% capture notice-2 %}
**keras:** 원래는 tensorflow 위한 애가 아니었습니다. 처음에는 티아노라는 애가 있었는데, 쓰기가 어려웠습니다. 
그래서 처음에 tensorflow의 쉬운 부분은 keras가 쓰도록 했지만, 이제는 서로 강하게 통합되었습니다. 
딥러닝 개념만 이해하고 있으면 상호적으로 배우기가 쉽습니다. 
{% endcapture %}

<div class="notice">{{ notice-2 | markdownify }}</div>



### (예제1) 숫자 mnist

- 데이터 

```python
mnist = tf.keras.datasets.mnist

# Normalization
x_train, x_test = x_train/255.0, x_test / 255.0

# 0과 1사이에서 찾는게 빠르기 때문에 이걸 하면 더 빠르게 값을 찾을 수 있다. 
```

#### Tensorflow Model 

만드는 법 2가지 (Model1, Model2)

- Model1 

```python
from tensorflow.keras.models import Sequential # 대문자: 클래스 
model = Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu', name='1'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10,activation = 'softmax', name='2')
], name='one') # 이름을 지정할 수 있다. 

model.compile(optimizer='adam',
             loss= 'sparse_categorical_crossentropy',
             metrics=['accuracy'])

# loss는 데이터에 따라 다르게 정한다. 
# gradient decent: optimizer 를 이용하여 global error를 찾도록 도와준다.
```

- Model2

```python
model = Sequential()
# flatten은 코드 편의상 나오는 레이어이다. 
# 남에것 가져올 때 input_layer를 안만드는 대신, input_Shape를 만들어줘야한다. 
# input_shape/input_dimension 이렇게 쓰는방법 2가지
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
# Dropout도 편의상 레이어 (앞에거 Dense에서 랜덤하게 유닛을 선택하여 0으로 만드는 것)
model.add(tf.keras.layers.Dropout(0.2))
# 맞추기 위해서 하는 것 
model.add(tf.keras.layers.Dense(10,activation='softmax'))

# 여기서는 hidden layer 1개 

model.summary()
'''
Model: "sequential_5"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten_13 (Flatten)         (None, 784)               0         
_________________________________________________________________
1 (Dense)                    (None, 128)               100480    
_________________________________________________________________
dropout_12 (Dropout)         (None, 128)               0         
_________________________________________________________________
2 (Dense)                    (None, 10)                1290      
=================================================================
Total params: 101,770
Trainable params: 101,770
Non-trainable params: 0
_________________________________________________________________
'''
```


#### Compile 

- 손실 함수(Loss function)-훈련 하는 동안 모델의 오차를 측정합니다. 모델의 학습이 올바른 방향으로 향하도록 이 함수를 최소화해야 합니다. <br>
    출력변수가 categorical 형태일 경우 변수를 
    Onehot encoding을 하면 categorical_crossentropy를 쓰고, oneHot encoding을 하지 않으면
    sparse_categorical_crossentropy를 쓴다.

- 옵티마이저(Optimizer)-데이터와 손실 함수를 바탕으로 모델의 업데이트 방법을 결정합니다. (learning rate,gradient) 

- 지표(Metrics)-훈련 단계와 테스트 단계를 모니터링하기 위해 사용합니다. 다음 예에서는 올바르게 분류된 이미지의 비율인 정확도를 사용합니다.

 
  
```python
model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
```

#### fit / train_on_batch

- fit: epochs/batch_size

- train_on_batch: batch_size


```python
## fit 
model.fit(x_train, y_train, epochs=5)
# batch_size: 한번에 학습하는 것이 아닌 몇개 나누어 학습을 통해 갱신
# epochs: 전체 데이터를 5번 학습시킨다는 것 
'''
Train on 60000 samples
Epoch 1/5
60000/60000 [==============================] - 8s 127us/sample - loss: 0.0640 - accuracy: 0.9798
Epoch 2/5
60000/60000 [==============================] - 8s 126us/sample - loss: 0.0582 - accuracy: 0.9810
Epoch 3/5
60000/60000 [==============================] - 10s 171us/sample - loss: 0.0535 - accuracy: 0.9826
Epoch 4/5
60000/60000 [==============================] - 8s 139us/sample - loss: 0.0479 - accuracy: 0.9841
Epoch 5/5
60000/60000 [==============================] - 7s 123us/sample - loss: 0.0429 - accuracy: 0.9857
<tensorflow.python.keras.callbacks.History at 0x2157a40aa58>
'''
# 명령형 패러다임 방식/ eagerly executing method 

# tensorflow2에서 이 방식으로 바뀌었다. 전에는 C처럼 compile 해줘야만 보였다. 
model.evaluate(x_test,y_test, verbose=2)
10000/1 - 1s - loss: 0.0407 - accuracy: 0.9762
[0.07903090946055018, 0.9762]


## train_on_batch
model.train_on_batch(x_train, y_train)
# loss/ accuracy 
# [2.4186661, 0.0872]
```

### (예제2) Fashion Mnist 

- 데이터 불러오기

```python
fashion = keras.datasets.fashion_mnist
(train_im, train_labels),(test_im, test_labels) = fashion.load_data()

plt.imshow(train_im[0])
plt.colorbar()
```

{% include gallery id="gallery1" caption="fashion_mnist" %}


```python
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_im.shape
# (60000, 28, 28)

# normalization
train_im = train_im/255.0
test_im = test_im/255.0

# 데이터 확인하기 
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_im[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
```
{% include gallery id="gallery2" caption="fashion5*5" %}


#### Model 구조 만들기

{% capture notice-2 %}
### activation function
데이터에 따라 쓰는 activation 함수가 다르다. 
- binary classification: sigmoid
- multi classification: softmax
- regression: actiavtion function 필요 없다.
{% endcapture %}

<div class="notice">{{ notice-2 | markdownify }}</div>

```python
model = keras.Sequential()
# 구조만들기
model.add(keras.layers.Flatten(input_shape=(28,28)))
model.add(keras.layers.Dense(128, activation = 'relu'))
# classification 개수 맞춰준 것 
model.add(keras.layers.Dense(10, activation = 'softmax'))

model.summary()
'''
Model: "sequential_6"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten_6 (Flatten)          (None, 784)               0         
_________________________________________________________________
dense_11 (Dense)             (None, 128)               100480    
_________________________________________________________________
dense_12 (Dense)             (None, 10)                1290      
=================================================================
Total params: 101,770
Trainable params: 101,770
Non-trainable params: 0
_________________________________________________________________
'''
```


#### Compile 

```python
# compile : loss, optimizer, metrics 
model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy']
             )
# categorical_crossentropy (보통)
# sparse_categorical_crossentropy : (one-hot encoding 안된 데이터)
```

#### fit

```python
# 모델 훈련 
a = model.fit(train_im, train_labels, epochs=5)

# validation_data -> validation 셋을 지정해줄 수 있다. 
# batch_size: 한번에 학습하는 것이 아닌 몇개 나누어 학습을 통해 갱신
# epochs: 전체 데이터를 5번 학습시킨다는 것 
'''
Train on 60000 samples
Epoch 1/5
60000/60000 [==============================] - 8s 137us/sample - loss: 0.4991 - accuracy: 0.8236
Epoch 2/5
60000/60000 [==============================] - 8s 132us/sample - loss: 0.3749 - accuracy: 0.8656
Epoch 3/5
60000/60000 [==============================] - 8s 138us/sample - loss: 0.3370 - accuracy: 0.8778
Epoch 4/5
60000/60000 [==============================] - 8s 132us/sample - loss: 0.3142 - accuracy: 0.8857
Epoch 5/5
60000/60000 [==============================] - 8s 129us/sample - loss: 0.2959 - accuracy: 0.8916
'''

## history
a.history
'''
{'loss': [0.3191453230281671,
  0.3028219168563684,
  0.29111538934310277,
  0.2813006566405296,
  0.27239179743429026],
 'accuracy': [0.88545, 0.8901833, 0.89538336, 0.89825, 0.9009167]}
'''
# 값 변화한 결과가 히스토리에 저장이 된다. 
# dictionary 형태 
# 그래프 그릴 수 있다.

## Dense 128 ROC
plt.plot(a.history['loss'], a.history['accuracy'])
plt.xlabel('loss')
plt.ylabel('accuracy')
```
{% include gallery id="gallery3" caption="roc" %}


```python
test_loss, test_acc = model.evaluate(test_im, test_labels, verbose=2)
print('\n테스트 정확도:', test_acc)
'''
10000/1 - 1s - loss: 0.3180 - accuracy: 0.8624

테스트 정확도: 0.8624
'''


## train_on_batch
model.train_on_batch(train_im, train_labels)
# [0.2740793, 0.8984333]

## predictions
predictions = model.predict(test_im)
predictions[0]
'''
array([1.2476424e-06, 2.6267611e-07, 2.6416228e-07, 9.3259587e-08,
       7.9673703e-07, 5.7914983e-03, 8.8731558e-06, 2.2964407e-02,
       2.0012419e-06, 9.7123057e-01], dtype=float32)
'''

## 예측
np.argmax(predictions[0])
# 9 
## 실제값
test_labels[0]
# 9
```

#### 예측한 내용 그래프 그리기

```python
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.xticks([])
    plt.yticks([])
    
    plt.imshow(img, cmap=plt.cm.binary)
    
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color='blue'
    else:
        color ='red'
        
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                        100*np.max(predictions_array),
                                        class_names[true_label]),
              color=color)
    
def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array)
    predicted_label = np.argmax(predictions_array)
    
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_labels, test_im)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, test_labels)
plt.show()
```
{% include gallery id="gallery4" caption="fashion" %}

### One-Hot encoding 

{% capture notice-2 %}
### 원핫 인코딩 

장점: 크기 학습시킬 때 영향이 없다. 

단점: 차원이 10배 확장된다. 성능이 안좋을 수 도 있다. 그리고 차원 확장으로 시간이 더 걸리는 확률이 더 크다.

라벨 데이터를 원핫으로 돌렸다. 라벨일 때는 굳이 원핫 인코딩 안해도된다. 속도에서 큰 차이가 없다. 

<->

숫자값으로 되어있는 인코딩을 라벨 인코딩이라고 한다.
타켓에서는 라벨 인코딩을 원핫 인코딩으로 바꿀 필요가 없다. 

하지만 학습데이터일 경우는 고민해야한다. 
원핫으로 바꾸면 차원이 늘어나서 안좋게 나올 수도 있고, 좋게 나올수도 있다.

자연어처리에서 word2vec에 따라 성능이 달라진다.

케라스 쓸 때는 loss function을 원핫 인코딩에 따라 기법이 다르다. (sparse 있고 없고)
{% endcapture %}

<div class="notice">{{ notice-2 | markdownify }}</div>

- one-hot encoding (sklearn)

```python
from sklearn.preprocessing import OneHotEncoder
import sklearn
sklearn.__version__ # 20 버전 이상에서만 쓸 수 있다. 
# '0.21.3'

ohe = OneHotEncoder()
# reshape 해줘야한다. 
train_labels.reshape(-1,1)
'''
array([[9],
       [0],
       [0],
       ...,
       [3],
       [0],
       [5]], dtype=uint8)
'''
ohe.fit_transform(train_labels.reshape(-1,1)).toarray()
'''
array([[0., 0., 0., ..., 0., 0., 1.],
       [1., 0., 0., ..., 0., 0., 0.],
       [1., 0., 0., ..., 0., 0., 0.],
       ...,
       [0., 0., 0., ..., 0., 0., 0.],
       [1., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.]])
'''
```
- one-hot encoding (tensorflow)

```python
from tensorflow.keras.utils import to_categorical
test_labels = to_categorical(test_labels)
train_labels = to_categorical(train_labels)
```

- 모델링

```python
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28,28)))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])
# onehot encoding 할 때는 categorical_crossentropy 를 써야한다. 

model.fit(train_im,train_labels, epochs=5)
'''
Train on 60000 samples
Epoch 1/5
60000/60000 [==============================] - 8s 140us/sample - loss: 0.3463 - accuracy: 0.8741
Epoch 2/5
60000/60000 [==============================] - 9s 147us/sample - loss: 0.3199 - accuracy: 0.8826
Epoch 3/5
60000/60000 [==============================] - 9s 154us/sample - loss: 0.2993 - accuracy: 0.8895
Epoch 4/5
60000/60000 [==============================] - 9s 143us/sample - loss: 0.2859 - accuracy: 0.8923
Epoch 5/5
60000/60000 [==============================] - 8s 136us/sample - loss: 0.2718 - accuracy: 0.8983
'''
```

### input_shape

Flatten하면 시간이 많이 걸리므로 
아예 처음부터 데이터를 전처리(flatten)하고 쓰는게 시간상 더 빠르게 쓸 수 있습니다.

정확도는 flatten을 모델에서 하는 모델과 똑같이 나옵니다. 다만 나중에 학습 속도를 줄이기 위해
전처리에서 모양에 맞게 flatten해주는 것이 효율적입니다. 


```python
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
## 미리 전처리(flatten)를 해준다. 
train_im = train_im.reshape(-1,28*28)

train_im.shape
# (60000, 784)

model = keras.Sequential()
# sequential 쓸 때는 무조건 input_shape이 들어가야한다. 
# shape은 튜플: 1개짜리 튜플은 ,(콤마)를 붙여야한다. 
model.add(keras.layers.Dense(128, activation='relu', input_shape=(28*28,)))
model.add(keras.layers.Dense(10, activation='softmax'))

model.summary()
'''
Model: "sequential_20"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_38 (Dense)             (None, 128)               100480    
_________________________________________________________________
dense_39 (Dense)             (None, 10)                1290      
=================================================================
Total params: 101,770
Trainable params: 101,770
Non-trainable params: 0
_________________________________________________________________
'''
```



{% include gallery id="gallery" caption="dummydata" %}

{% capture notice-2 %}

{% endcapture %}

<div class="notice">{{ notice-2 | markdownify }}</div>




---
**무단 배포 금지** 
{: .notice--danger}
