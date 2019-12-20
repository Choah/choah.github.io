---
excerpt: "(1) autoencoder (2) keras deep autoencoder (3) keras convolutional autoencoder"
header:
  overlay_image: /assets/images/2computers.jpg
  overlay_filter: 0.5 # same as adding an opacity of 0.5 to a black background
  caption: "Photo credit: [**Unsplash**]"
  actions:
    - label: "Reference"
      url: "https://blog.keras.io/building-autoencoders-in-keras.html"
title: "[Tensorflow] Autoencoder"
date: 2019-12-19 00:00:00 -0400
categories: tensorflow
tags: Autoencoder
gallery1:
  - url: /assets/images/mse.JPG
    image_path: assets/images/mse.JPG
    alt: "placeholder image"
gallery2:
  - url: /assets/images/binary_crossentropy.JPG
    image_path: assets/images/binary_crossentropy.JPG
    alt: "placeholder image"
gallery3:
  - url: /assets/images/encoder.JPG
    image_path: assets/images/encoder.JPG
    alt: "placeholder image"   
gallery4:
  - url: /assets/images/encoder+decoder.JPG
    image_path: assets/images/encoder+decoder.JPG
    alt: "placeholder image"    
gallery5:
  - url: /assets/images/flower_for.JPG
    image_path: assets/images/flower_for.JPG
    alt: "placeholder image"    
gallery6:
  - url: /assets/images/dog_cat.JPG
    image_path: assets/images/dog_cat.JPG
    alt: "placeholder image"  
---



# Autoencoder 


## mse autoencoder 

```python
# 데이터 불러오기
from tensorflow.keras.datasets import mnist
# unsupervised learning 
(X_train,_), (X_test,_) = mnist.load_data()

X_train = X_train/255.
X_test = X_test/255.

# 데이터 모델 만들기
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D

model = Sequential()

model.add(Dense(64, activation='relu', input_shape=(28,28,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(28))

model.summary()
'''
Model: "sequential_25"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_105 (Dense)            (None, 28, 64)            1856      
_________________________________________________________________
dense_106 (Dense)            (None, 28, 32)            2080      
_________________________________________________________________
dense_107 (Dense)            (None, 28, 8)             264       
_________________________________________________________________
dense_108 (Dense)            (None, 28, 32)            288       
_________________________________________________________________
dense_109 (Dense)            (None, 28, 64)            2112      
_________________________________________________________________
dense_110 (Dense)            (None, 28, 28)            1820      
=================================================================
Total params: 8,420
Trainable params: 8,420
Non-trainable params: 0
_________________________________________________________________
'''

model.compile(loss='mse', metrics = ['mse'], optimizer= 'adam')

model.fit(X_train,X_train,epochs=3)
'''
Train on 60000 samples
Epoch 1/3
60000/60000 [==============================] - 9s 157us/sample - loss: 0.0077 - mse: 0.0077
Epoch 2/3
60000/60000 [==============================] - 9s 151us/sample - loss: 0.0027 - mse: 0.0027
Epoch 3/3
60000/60000 [==============================] - 10s 174us/sample - loss: 0.0023 - mse: 0.0023
'''
```

- autoencoder한 값 비교해보기 

```python
a = model.predict(X_train)[0]

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1,2)
axes[0].imshow(a)
axes[1].imshow(X_train[0])
```
{% include gallery id="gallery1" caption="mse" %}



## binary_crossentropy로 autoencoder

```python
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(28*28,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(28*28, activation='sigmoid'))

model.summary()
'''
Model: "sequential_7"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_42 (Dense)             (None, 64)                50240     
_________________________________________________________________
dense_43 (Dense)             (None, 32)                2080      
_________________________________________________________________
dense_44 (Dense)             (None, 8)                 264       
_________________________________________________________________
dense_45 (Dense)             (None, 32)                288       
_________________________________________________________________
dense_46 (Dense)             (None, 64)                2112      
_________________________________________________________________
dense_47 (Dense)             (None, 784)               50960     
=================================================================
Total params: 105,944
Trainable params: 105,944
Non-trainable params: 0
_________________________________________________________________
'''

model.compile(loss='binary_crossentropy', optimizer= 'adam')

model.fit(X_test.reshape(-1,28*28), X_test.reshape(-1,28*28), epochs=10)
'''
Train on 10000 samples
Epoch 1/10
10000/10000 [==============================] - 2s 184us/sample - loss: 0.2877
Epoch 2/10
10000/10000 [==============================] - 1s 114us/sample - loss: 0.2203
Epoch 3/10
10000/10000 [==============================] - 1s 138us/sample - loss: 0.1823
Epoch 4/10
10000/10000 [==============================] - 1s 120us/sample - loss: 0.1705
Epoch 5/10
10000/10000 [==============================] - 1s 114us/sample - loss: 0.1638
Epoch 6/10
10000/10000 [==============================] - 1s 131us/sample - loss: 0.1595
Epoch 7/10
10000/10000 [==============================] - 1s 130us/sample - loss: 0.1568
Epoch 8/10
10000/10000 [==============================] - 1s 144us/sample - loss: 0.1547
Epoch 9/10
10000/10000 [==============================] - 2s 156us/sample - loss: 0.1530
Epoch 10/10
10000/10000 [==============================] - 1s 122us/sample - loss: 0.1515
<tensorflow.python.keras.callbacks.History at 0x1d092704c18>
'''
```

- 데이터 확인

```python
a = model.predict(X_train.reshape(-1,28*28))[0]

fig, axes = plt.subplots(1,2)
axes[0].imshow(a.reshape(28,28))
axes[1].imshow(X_train[0])
```

{% include gallery id="gallery2" caption="binary_crossentropy" %}



## Encoder / decoder 

```python
layer1 = Dense(64, activation = 'relu', input_shape = (28*28,))
layer2 = Dense(32, activation = 'relu')
layer3 = Dense(8, activation = 'relu')
layer4 = Dense(28*28)
encoder = Sequential([layer1,layer2,layer3, layer4])

layer5 = Dense(32, activation='relu', input_shape=(8,))
layer6 = Dense(64, activation='relu')
layer7 = Dense(28*28, activation = 'sigmoid')
decoder = Sequential([layer5, layer6, layer7])

encoder.compile(loss= 'mse', optimizer= 'adam')
encoder.output_shape
# (None, 784)

encoder.fit(X_train.reshape(-1,28*28),X_train.reshape(60000,28*28),epochs=2)
'''
Train on 60000 samples
Epoch 1/2
60000/60000 [==============================] - 5s 84us/sample - loss: 0.0493
Epoch 2/2
60000/60000 [==============================] - 5s 75us/sample - loss: 0.0452
<tensorflow.python.keras.callbacks.History at 0x1d09aac4208>
'''
```

- encoder 데이터 확인 

```python
a = ende.predict(X_train.reshape(-1,28*28))[0]
fig, axes = plt.subplots(1,2)
axes[0].imshow(a.reshape(28,28))
axes[1].imshow(X_train[0])
```

{% include gallery id="gallery3" caption="encoder" %}


- encoder + decoder 

```python
ende = Sequential([encoder,decoder])
ende.summary()
'''
Model: "sequential_22"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
sequential_20 (Sequential)   (None, 8)                 52584     
_________________________________________________________________
sequential_21 (Sequential)   (None, 784)               53360     
=================================================================
Total params: 105,944
Trainable params: 105,944
Non-trainable params: 0
_________________________________________________________________
'''

ende.compile(loss='binary_crossentropy', optimizer= 'adam')
ende.fit(X_train.reshape(-1,28*28),X_train.reshape(-1,28*28),epochs=2)
'''
Train on 60000 samples
Epoch 1/2
60000/60000 [==============================] - 7s 121us/sample - loss: 0.1896
Epoch 2/2
60000/60000 [==============================] - 6s 107us/sample - loss: 0.1549
<tensorflow.python.keras.callbacks.History at 0x1d09a26c550>
'''
```

- encoder + decoder 데이터 확인

```python
a = ende.predict(X_train.reshape(-1,28*28))[0]
fig, axes = plt.subplots(1,2)
axes[0].imshow(a.reshape(28,28))
axes[1].imshow(X_train[0])
```

{% include gallery id="gallery4" caption="encoder+decoder" %}



## Building Autoencoders in keras

### Deep autoencoder

- activity_regularizer=regularizers.l1(10e-5) (no free lunch) 

```python
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model 
from tensorflow.keras import regularizers

input_img = Input(shape=(784,))
encoded = Dense(32, activation='relu',
                activity_regularizer=regularizers.l1(10e-5))(input_img)
# kernel_regularizer, bias_regularizer 은 자기 자신값을 변형시켜버리기 때문에 쓰지 않는다.

# mse는 값이 너무 커진다. mse로 풀어도 상관없다.
decoded = Dense(784, activation='sigmoid')(encoded)
autoencoder = Model(input_img, decoded)


encoder = Model(input_img, encoded)

encoded_output = Input(shape=(32,))
decoder_layer = autoencoder.layers[-1]

decoder = Model(encoded_output, decoder_layer(encoded_output))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train/255.
x_test = x_test/255.

x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))
'''
Train on 60000 samples, validate on 10000 samples
Epoch 1/10
60000/60000 [==============================] - 8s 140us/sample - loss: 0.6967 - val_loss: 0.6967
Epoch 2/10
60000/60000 [==============================] - 5s 83us/sample - loss: 0.6965 - val_loss: 0.6965
Epoch 3/10
60000/60000 [==============================] - 5s 80us/sample - loss: 0.6963 - val_loss: 0.6962
Epoch 4/10
60000/60000 [==============================] - 5s 78us/sample - loss: 0.6960 - val_loss: 0.6960
Epoch 5/10
60000/60000 [==============================] - 5s 84us/sample - loss: 0.6958 - val_loss: 0.6958
Epoch 6/10
60000/60000 [==============================] - 5s 86us/sample - loss: 0.6956 - val_loss: 0.6956
Epoch 7/10
60000/60000 [==============================] - 5s 88us/sample - loss: 0.6954 - val_loss: 0.6954
Epoch 8/10
60000/60000 [==============================] - 5s 91us/sample - loss: 0.6952 - val_loss: 0.6952
Epoch 9/10
60000/60000 [==============================] - 5s 78us/sample - loss: 0.6950 - val_loss: 0.6950
Epoch 10/10
60000/60000 [==============================] - 5s 82us/sample - loss: 0.6948 - val_loss: 0.6948
<tensorflow.python.keras.callbacks.History at 0x1d0d206ae80>
'''
```

- Deep Autoencoder 

sparse 방식

```python
input_img = Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)


autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train,
                epochs=5,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))
'''
Train on 60000 samples, validate on 10000 samples
Epoch 1/5
60000/60000 [==============================] - 6s 92us/sample - loss: 0.6942 - val_loss: 0.6941
Epoch 2/5
60000/60000 [==============================] - 4s 61us/sample - loss: 0.6941 - val_loss: 0.6941
Epoch 3/5
60000/60000 [==============================] - 4s 63us/sample - loss: 0.6940 - val_loss: 0.6940
Epoch 4/5
60000/60000 [==============================] - 4s 61us/sample - loss: 0.6939 - val_loss: 0.6939
Epoch 5/5
60000/60000 [==============================] - 4s 59us/sample - loss: 0.6938 - val_loss: 0.6938
<tensorflow.python.keras.callbacks.History at 0x1d0ce1e0b70>
'''
```

### Convolutional autoencoder 

- fully connected 보다 convolution한 autoencoder가 성능이 더 좋다. 

stack 방식

```python
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np


(x_train,_), (x_test,_) = mnist.load_data()
x_train = x_train/255.
x_test = x_test/255.
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)


input_img = Input(shape=(28,28,1))
x = Conv2D(16,(3,3), activation='relu', padding ='same')(input_img)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2), padding='same')(x)

x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2,2), padding='same')(x)

x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
# padding: same - > valid
x = Conv2D(16, (3,3), activation = 'relu', padding = 'same')(x)
x = UpSampling2D((2,2))(x)
# 마지막에 Conv2D 1로 해줘야한다.
decoded = Conv2D(1, (3,3), activation='sigmoid', padding = 'same')(x)


autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
```




{% include gallery id="gallery" caption="" %}




{% capture notice-2 %}

{% endcapture %}

<div class="notice">{{ notice-2 | markdownify }}</div>



