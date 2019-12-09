---
excerpt: "(1) Tensorboard"
header:
  overlay_image: /assets/images/2computers.jpg
  overlay_filter: 0.5 # same as adding an opacity of 0.5 to a black background
  caption: "Photo credit: [**Unsplash**]"
  actions:
    - label: "Reference"
      url: "https://www.tensorflow.org/tensorboard"
title: "[Python] Tensorboard"
date: 2019-12-09 00:00:00 -0400
categories: tensorflow
tags: tensorboard 
gallery1:
  - url: /assets/images/number.JPG
    image_path: assets/images/number.JPG
    alt: "placeholder image"
gallery2:
  - url: /assets/images/train_test.JPG
    image_path: assets/images/train_test.JPG
    alt: "placeholder image" 
---

---
**본 과정은 제가 파이썬 수업을 들으면서 배운 내용을 복습하는 과정에서 적어본 것입니다.<br> 틀린 부분이 있다면 댓글에 남겨주시면 고치도록 하겠습니다.<br> 확실하지 않은 내용은 '(?)'을 함께 적었으니 그 내용을 아신다면 댓글에 남겨주시면 감사하겠습니다.** 
{: .notice--warning}
--- 

# Tensorboard 

TensorFlow에서 제공한 툴로 데이터의 시각화를 도와줍니다. Tensor의 흐름을 쉽게 이해하고, 각
변수들의 변화를 한 눈에 알 수 있도록 다양한 시각 기능을 제공합니다. 

## 설치 

Tensorboard를 사용하기 위해서는 Tensorflow의 버전이 2.0.0 이상이어야 합니다. 
또한 Tensorflow의 버전과 Tensorboard의 버전이 일치해야합니다. 

```python
!pip uninstall tensorboard

# 버전 확인해보기
%load_ext watermark
%watermark -acho -d -ptensorflow,tensorboard
'''
cho 2019-12-09 

tensorflow 2.0.0
tensorboard 2.0.0
'''
```

```ptyhon
# Clear any logs from previous runs
# 누적이 되므로 지워준다. 
!rm -rf ./logs/ 

import tensorflow as tf 
```

## 모델링

```python
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def create_model():
  return tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
  ])
  
model = create_model()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten (Flatten)            (None, 784)               0         
_________________________________________________________________
dense (Dense)                (None, 512)               401920    
_________________________________________________________________
dropout (Dropout)            (None, 512)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 10)                5130      
=================================================================
Total params: 407,050
Trainable params: 407,050
Non-trainable params: 0
_________________________________________________________________
'''
```


## 저장 경로에 fit 모델 저장하기 

```python
tensorboard_callback = tf.keras.callbacks.TensorBoard('model')

model.fit(x=x_train, 
          y=y_train, 
          epochs=3, 
          validation_data=(x_test, y_test), 
          callbacks=[tensorboard_callback])
'''
Train on 60000 samples, validate on 10000 samples
Epoch 1/3
60000/60000 [==============================] - 21s 350us/sample - loss: 0.2209 - accuracy: 0.9350 - val_loss: 0.1008 - val_accuracy: 0.9694
Epoch 2/3
60000/60000 [==============================] - 21s 347us/sample - loss: 0.0969 - accuracy: 0.9707 - val_loss: 0.0801 - val_accuracy: 0.9762
Epoch 3/3
60000/60000 [==============================] - 20s 329us/sample - loss: 0.0698 - accuracy: 0.9779 - val_loss: 0.0693 - val_accuracy: 0.9778
'''
```

## tensorboard

```python
%load_ext tensorboard

# 보안이슈때문에 안나오는 경우가 있다. 
%tensorboard --logdir model
# !tensorboard --logdir model 를 실행시키면 새로운 창에 tensorboard가 켜진다. 
```

{% include gallery id="gallery" caption="tensorboard" %}


# Tensorboard: image 보기 


## 라이브러리 불러들이기 

```python
%load_ext tensorboard
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
```

## 데이터 로드

```python
# Download the data. The data is already divided into train and test.
# The labels are integers representing classes.
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = \
    fashion_mnist.load_data()

# Names of the integer classes, i.e., 0 -> T-short/top, 1 -> Trouser, etc.
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("Shape: ", train_images[0].shape)
print("Label: ", train_labels[0], "->", class_names[train_labels[0]])
'''
Shape:  (28, 28)
Label:  9 -> Ankle boot
'''
```

## CNN 

CNN은 데이터를 3차원으로 만들어야 합니다.

```python
# Reshape the image for the Summary API.
# CNN은 3차원 
img = np.reshape(train_images[0], (-1, 28, 28, 1))
```

## Tensorboard

```python
# Clear out any prior log data.
!rm -rf logs

# create_file_write: 어디에 저장할건지 지정해준다. 
file_writer = tf.summary.create_file_writer('model2')

# Using the file writer, log the reshaped image.
# 이건 그냥 암기해야한다. 
with file_writer.as_default():
  tf.summary.image("Training data", img, step=0)

%tensorboard --logdir model2
```


## 경로 지정시 

```python
from datetime import datetime
# Sets up a timestamped log directory.
# 날짜시간이 너무 길어서 이름 치는게 너무 귀찮을 수 있음...
logdir = "logs/train_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# Creates a file writer for the log directory.
# 파일이 비어있는 directory이면 에러가 나지 않는다. 
# tf.summary는 데이터 확인하는 용도로 다양한기능을 제공하고 있다.
file_writer = tf.summary.create_file_writer(logdir)

# Using the file writer, log the reshaped image.
# 최근파일에 내용이 as_default()를 이용하여 덮어진다. 
with file_writer.as_default():
  tf.summary.image("Training data", img, step=0, max_outputs=4)
# step: (int64를 monotonic level 으로 바꿔준다.): 대조레벨 (흑백이라서 잘 안보임)
# max_outputs 를 4로하면 한꺼번에 사진을 4개 볼 수 있도록 도와준다. 

%tensorboard --logdir logs/train_data
```


{% include gallery id="gallery" caption="validation_curve" %}


{% capture notice-2 %}

{% endcapture %}

<div class="notice">{{ notice-2 | markdownify }}</div>






{% capture notice-2 %}

{% endcapture %}

<div class="notice">{{ notice-2 | markdownify }}</div>




---
**무단 배포 금지** 
{: .notice--danger}
