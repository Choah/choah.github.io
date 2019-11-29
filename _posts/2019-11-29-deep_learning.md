---
excerpt: "(1) tensorflow"
header:
  overlay_image: /assets/images/2computers.jpg
  overlay_filter: 0.5 # same as adding an opacity of 0.5 to a black background
  caption: "Photo credit: [**Unsplash**]"
  actions:
    - label: "Reference"
      url: "https://docs.python.org/3/"
title: "[Tensorflow] Deep learning"
date: 2019-11-29 00:00:00 -0400
categories: deep_learning
tags: deep_learning 
gallery1:
  - url: /assets/images/dummydata.JPG
    image_path: assets/images/dummydata.JPG
    alt: "placeholder image"
gallery2:
  - url: /assets/images/train_test.JPG
    image_path: assets/images/train_test.JPG
    alt: "placeholder image"
gallery3:
  - url: /assets/images/roc_1.JPG
    image_path: assets/images/roc_1.JPG
    alt: "placeholder image"    
---

---
**본 과정은 제가 파이썬 수업을 들으면서 배운 내용을 복습하는 과정에서 적어본 것입니다.<br> 틀린 부분이 있다면 댓글에 남겨주시면 고치도록 하겠습니다.<br> 확실하지 않은 내용은 '(?)'을 함께 적었으니 그 내용을 아신다면 댓글에 남겨주시면 감사하겠습니다.** 
{: .notice--warning}
--- 

# Deep learning  

- unit 개수/ layer가 많을수록 성능은 좋다. 

## Tensorflow 

텐서플로우는 딥러닝을 위해 구글에서 제공하는 프레임워크입니다. 다시 말해, 어느 누구나 사용할수 있는 머신러닝 오픈소스 라이브러리입니다.
텐서 기반으로 artificial 뉴럴 네트워크 계열 프로그래밍 하는 것에 최적화되어 있는 라이브러리입니다.

Tensor = Multidimensional Arrays = Data (다차원 배열)

텐서는 가속기 메모리(GPU, TPU와 같은)에서 사용할 수 있습니다.
텐서는 불변성(immutable)을 가집니다.


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


```python
import tensorflow as tf
tf.__version__
# '2.0.0'
# 쿠다 지원 그래픽 카드 
# 쿠다 버전/ 설치/ mac에서 사용 불가 
```

### mnist 데이터 로드 

```python
mnist = tf.keras.datasets.mnist

# Normalization
x_train, x_test = x_train/255.0, x_test / 255.0

# 0과 1사이에서 찾는게 빠르기 때문에 이걸 하면 더 빠르게 값을 찾을 수 있다. 
```

### Tensorflow로 모델 만드는 법 2가지

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


### Compile 

- loss

    출력변수가 categorical 형태일 경우 변수를 
    Onehot encoding을 하면 categorical_crossentropy를 쓰고, oneHot encoding을 하지 않으면
    sparse_categorical_crossentropy를 쓴다.

- optimizer
- metrics 

 

```python
model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
```

### fit

```python
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

model.evaluate(x_test,y_test, verbose=2)
10000/1 - 1s - loss: 0.0407 - accuracy: 0.9762
[0.07903090946055018, 0.9762]
```

{% include gallery id="gallery" caption="dummydata" %}

{% capture notice-2 %}

{% endcapture %}

<div class="notice">{{ notice-2 | markdownify }}</div>




---
**무단 배포 금지** 
{: .notice--danger}
