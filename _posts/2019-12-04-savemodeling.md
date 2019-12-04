---
excerpt: "(1) callbacks (2) HDF5 (3) save_weights"
header:
  overlay_image: /assets/images/2computers.jpg
  overlay_filter: 0.5 # same as adding an opacity of 0.5 to a black background
  caption: "Photo credit: [**Unsplash**]"
  actions:
    - label: "Reference"
      url: "https://www.tensorflow.org/tutorials/load_data/images"
title: "[Tensorflow] 학습한 모델 저장하기"
date: 2019-12-04 00:00:00 -0400
categories: tensorflow
tags: callbacks  
gallery1:
  - url: /assets/images/sunflower.JPG
    image_path: assets/images/sunflower.JPG
    alt: "placeholder image"
gallery2:
  - url: /assets/images/flowercls.JPG
    image_path: assets/images/flowercls.JPG
    alt: "placeholder image"
gallery3:
  - url: /assets/images/flowerscls2.JPG
    image_path: assets/images/flowerscls2.JPG
    alt: "placeholder image"   
---



# 1. Callback

Tensorflow에는 Callback이라는 클래스가 있습니다. 
Callbacks를 통해서 가중치를 저장하고 불러오는 다양한 기능들을 추가할 수 있습니다. 
Tensorflow 에 있는 callback을 쓰면 tenorflow에서만 학습된 모델을 불러올 수 있습니다.
keras


- 데이터 로드 

Callback 예시를 사용할 예시 데이터 mnist를 불러오도록 하겠습니다.

```python
import os

import tensorflow as tf
from tensorflow import keras

# 데이터 다운받는다.
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels
test_labels = test_labels

# 일반적인 기계학습 (2차원 - 행, 열 )
train_images = train_images.reshape(-1, 28 * 28) / 255.0
test_images = test_images.reshape(-1, 28 * 28) / 255.0
```

- 모델 정의

```python
# 간단한 Sequential 모델을 반환합니다
def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
  ])

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                # 원핫 인코딩 안해서 sparse_
                metrics=['accuracy'])

  return model


# 모델 객체를 만듭니다
model = create_model()
model.summary()
'''
Model: "sequential_5"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_13 (Dense)             (None, 512)               401920    
_________________________________________________________________
dropout_8 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_14 (Dense)             (None, 512)               262656    
_________________________________________________________________
dropout_9 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_15 (Dense)             (None, 10)                5130      
=================================================================
Total params: 669,706
Trainable params: 669,706
Non-trainable params: 0
_________________________________________________________________
'''
```

## checkpoint

### tf.keras.callbacks.ModelCheckpoint(최종 모델을 저장하기)

체크포인트 콜백을 만들면 학습시킨 모델을 저장하고 불러올 수 있습니다.
ModelCheckpoint는 시행착오용으로 씁니다.

save_best_only: True (가장 좋은 하이퍼파라미터일때 불러들일 수 있습니다.)


```python
# 체크포인트 콜백 만들기 - 잘못되면 되돌아 갈 수 있는 것 
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model = create_model()
model.fit(train_images, train_labels,  epochs = 2,
          validation_data = (test_images,test_labels),
           # 훈련 단계에 콜백을 전달합니다
          callbacks = [cp_callback]) 

# 옵티마이저의 상태를 저장하는 것과 관련되a어 경고가 발생할 수 있습니다.
# 이 경고는 (그리고 이 노트북의 다른 비슷한 경고는) 이전 사용 방식을 권장하지 않기 위함이며 무시해도 좋습니다.
'''
Train on 60000 samples, validate on 10000 samples
Epoch 1/10
59904/60000 [============================>.] - ETA: 0s - loss: 0.2143 - accuracy: 0.9344 ETA: 0s - loss: 0.2152 - accuracy: 0.93 - ETA: 0s - loss: 0.2149 - accura
Epoch 00001: saving model to training_1/cp.ckpt
60000/60000 [==============================] - 38s 638us/sample - loss: 0.2143 - accuracy: 0.9344 - val_loss: 0.1012 - val_accuracy: 0.9658
Epoch 2/10
59904/60000 [============================>.] - ETA: 0s - loss: 0.1053 - accuracy: 0.9675 ETA: 0s - loss: 0.1053 - accuracy: 
Epoch 00002: saving model to training_1/cp.ckpt
60000/60000 [==============================] - 40s 662us/sample - loss: 0.1053 - accuracy: 0.9675 - val_loss: 0.0853 - val_accuracy: 0.9732
Epoch 3/10
59968/60000 [============================>.] - ETA: 0s - loss: 0.0795 - accuracy: 0.9751
Epoch 00003: saving model to training_1/cp.ckpt
60000/60000 [==============================] - 39s 653us/sample - loss: 0.0796 - accuracy: 0.9751 - val_loss: 0.0751 - val_accuracy: 0.9777
Epoch 4/10
59936/60000 [============================>.] - ETA: 0s - loss: 0.0629 - accuracy: 0.9796
Epoch 00004: saving model to training_1/cp.ckpt
60000/60000 [==============================] - 38s 629us/sample - loss: 0.0629 - accuracy: 0.9796 - val_loss: 0.0725 - val_accuracy: 0.9776
Epoch 5/10
59904/60000 [============================>.] - ETA: 0s - loss: 0.0573 - accuracy: 0.9822
Epoch 00005: saving model to training_1/cp.ckpt
60000/60000 [==============================] - 38s 641us/sample - loss: 0.0574 - accuracy: 0.9822 - val_loss: 0.0917 - val_accuracy: 0.9765
Epoch 6/10
59936/60000 [============================>.] - ETA: 0s - loss: 0.0498 - accuracy: 0.9842
Epoch 00006: saving model to training_1/cp.ckpt
60000/60000 [==============================] - 41s 677us/sample - loss: 0.0498 - accuracy: 0.9841 - val_loss: 0.0866 - val_accuracy: 0.9760
Epoch 7/10
59936/60000 [============================>.] - ETA: 0s - loss: 0.0475 - accuracy: 0.9852
Epoch 00007: saving model to training_1/cp.ckpt
60000/60000 [==============================] - 46s 764us/sample - loss: 0.0476 - accuracy: 0.9852 - val_loss: 0.0802 - val_accuracy: 0.9795
Epoch 8/10
59936/60000 [============================>.] - ETA: 0s - loss: 0.0412 - accuracy: 0.9875
Epoch 00008: saving model to training_1/cp.ckpt
60000/60000 [==============================] - 41s 679us/sample - loss: 0.0412 - accuracy: 0.9875 - val_loss: 0.0746 - val_accuracy: 0.9822
Epoch 9/10
59936/60000 [============================>.] - ETA: 0s - loss: 0.0417 - accuracy: 0.9870
Epoch 00009: saving model to training_1/cp.ckpt
60000/60000 [==============================] - 42s 694us/sample - loss: 0.0417 - accuracy: 0.9870 - val_loss: 0.0853 - val_accuracy: 0.9772
Epoch 10/10
59936/60000 [============================>.] - ETA: 0s - loss: 0.0356 - accuracy: 0.9883
Epoch 00010: saving model to training_1/cp.ckpt
60000/60000 [==============================] - 40s 663us/sample - loss: 0.0357 - accuracy: 0.9883 - val_loss: 0.0751 - val_accuracy: 0.9823
<tensorflow.python.keras.callbacks.History at 0x20355940978>
'''
```

- 학습시킨 모델 불러오기 

load_weights를 써서 학습시킨 모델을 불러올 수 있습니다. 

```python
model.load_weights(checkpoint_path)
loss,acc = model.evaluate(test_images,  test_labels, verbose=2)
print("복원된 모델의 정확도: {:5.2f}%".format(100*acc))

'''
10000/1 - 2s - loss: 0.0376 - accuracy: 0.9823
복원된 모델의 정확도: 98.23%
'''


model = create_model()

loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
print("훈련되지 않은 모델의 정확도: {:5.2f}%".format(100*acc))
'''
10000/1 - 2s - loss: 2.3628 - accuracy: 0.0975
훈련되지 않은 모델의 정확도:  9.75%
'''
```



### tf.keras.callbacks.ModelCheckpoint(몇 번째 epoch 마다 가중치 저장하기)

```python
# 파일 이름에 에포크 번호를 포함시킵니다(`str.format` 포맷)
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # period: 한 번째 에포크마다 가중치를 저장합니다
    period=1)
# period를 굳이 쓸 필요는 없다. (test용으로 쓰기 때문에 필요없다.)
# save_best_model를 쓰는게 훨씬 낫다.

model = create_model()
model.save_weights(checkpoint_path.format(epoch=0))
model.fit(train_images, train_labels,
          epochs = 2, callbacks = [cp_callback],
          validation_data = (test_images,test_labels),
          verbose=0)
```

- 저장한 경로 찾기

```python
latest = tf.train.latest_checkpoint(checkpoint_dir)
latest
# 'training_2\\cp-0002.ckpt'
```

- 저장한 경로에서 학습된 모델 불러오기 

```python
# keras 버전에서만 불러들일 수 있다. 
# keras checkpoint는 테스트용이다. 
model = create_model()
model.load_weights(latest)
loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
print("복원된 모델의 정확도: {:5.2f}%".format(100*acc))
'''
10000/1 - 2s - loss: 0.0538 - accuracy: 0.9711
복원된 모델의 정확도: 97.11%
'''
```


# 2. HDF5 

모델 전체를 저장하는 방법으로 HDF5 파일로 저장하는 방법이 있습니다. 
HDF5는 tensorflow, keras 뿐만 아니라 다양한 라이브러리에서 모델를 불러올 수 있어서 callback 보다 
많이 쓰지만 최종 모델(마지막 모델)만 저장할 수 있습니다. 

```python
!pip install -q h5py pyyaml
# 데이터를 관리하기 위해서 많이 쓴다.

model = create_model()

model.fit(train_images, train_labels, epochs=5)

# 전체 모델을 HDF5 파일로 저장합니다
model.save('my_model.h5')
# model checkpoint 
# 이건 최종(마지막) 모델만 저장할 수 있다. 
'''
Train on 60000 samples
Epoch 1/5

60000/60000 [==============================] - 34s 568us/sample - loss: 0.2153 - accuracy: 0.9335
Epoch 2/5
60000/60000 [==============================] - 35s 591us/sample - loss: 0.1066 - accuracy: 0.9672
Epoch 3/5
60000/60000 [==============================] - 38s 638us/sample - loss: 0.0801 - accuracy: 0.9746
Epoch 4/5
60000/60000 [==============================] - 36s 600us/sample - loss: 0.0686 - accuracy: 0.9790- loss: 0.0683 - accuracy:  - ETA: 0s - loss: 0.0684 - 
Epoch 5/5
60000/60000 [==============================] - 38s 638us/sample - loss: 0.0589 - accuracy: 0.9811
'''
```


# 3. save_weights 

수동으로 가중치를 저장할 수 있습니다. 

```python
# 가중치를 저장합니다
model.save_weights('./checkpoints/my_checkpoint')

# 가중치를 복원합니다
model = create_model()
model.load_weights('./checkpoints/my_checkpoint')

loss,acc = model.evaluate(test_images,  test_labels, verbose=2)
print("복원된 모델의 정확도: {:5.2f}%".format(100*acc))
```

- save_format 형태를 변경시킬 수 있습니다. 

```python
model.save_weights('./checkpoints/my_checkpoint', save_format='tf')  
```


{% include gallery id="gallery" caption="flowers" %}



{% capture notice-2 %}

{% endcapture %}

<div class="notice">{{ notice-2 | markdownify }}</div>



