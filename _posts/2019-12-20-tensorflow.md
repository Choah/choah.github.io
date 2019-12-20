---
excerpt: "(1) 연산"
header:
  overlay_image: /assets/images/2computers.jpg
  overlay_filter: 0.5 # same as adding an opacity of 0.5 to a black background
  caption: "Photo credit: [**Unsplash**]"
  actions:
    - label: "Reference"
      url: "https://blog.keras.io/building-autoencoders-in-keras.html"
title: "[Tensorflow] model 연산"
date: 2019-12-20 00:00:00 -0400
categories: tensorflow
tags: model
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



# Tensorflow 모델 연산 

Model은 multi inputs, multi outputs이 가능합니다.

{% capture notice-2 %}
- Model은 다방향으로 가능하다. -> sequence가 안된다. 
- sequential은 한방향만 가능하다. (sequence를 하기 위해서는 합병(merge)하는 방법이 있다.)
{% endcapture %}

<div class="notice">{{ notice-2 | markdownify }}</div>



- add, Average, Concatenate, Dot, Lambda

```python
from tensorflow.keras.layers import Lambda, Dense, Add, Dot, Average, Concatenate
from tensorflow.keras.models import Sequential, Model
input1 = tf.keras.layers.Input(shape=(2,2))


layer1 = Dense(2, kernel_initializer='ones')(input1)
layer2 = Dense(2,  kernel_initializer='ones')(input1)


x = np.array([[1,2],[3,4]])


mo_1 = Model(input1,layer1)
mo_1.summary()
'''
Model: "model_7"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_7 (InputLayer)         [(None, 2, 2)]            0         
_________________________________________________________________
dense_14 (Dense)             (None, 2, 2)              6         
=================================================================
Total params: 6
Trainable params: 6
Non-trainable params: 0
_________________________________________________________________
'''
mo_1.predict(x[np.newaxis])
'''
array([[[3., 3.],
        [7., 7.]]], dtype=float32)
'''


# add
layer3 = Add()([layer1,layer2])
mo = Model(input1, layer3)
mo.summary()
'''
Model: "model_5"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_7 (InputLayer)            [(None, 2, 2)]       0                                            
__________________________________________________________________________________________________
dense_14 (Dense)                (None, 2, 2)         6           input_7[0][0]                    
__________________________________________________________________________________________________
dense_15 (Dense)                (None, 2, 2)         6           input_7[0][0]                    
__________________________________________________________________________________________________
add_6 (Add)                     (None, 2, 2)         0           dense_14[0][0]                   
                                                                 dense_15[0][0]                   
==================================================================================================
Total params: 12
Trainable params: 12
Non-trainable params: 0
__________________________________________________________________________________________________
'''
mo.predict(x[np.newaxis])
'''
array([[[ 6.,  6.],
        [14., 14.]]], dtype=float32)
'''


# average
layer4 = Average()([layer1,layer2])
mo_4 = Model(input1, layer4)
mo_4.summary()
'''
Model: "model_8"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_7 (InputLayer)            [(None, 2, 2)]       0                                            
__________________________________________________________________________________________________
dense_14 (Dense)                (None, 2, 2)         6           input_7[0][0]                    
__________________________________________________________________________________________________
dense_15 (Dense)                (None, 2, 2)         6           input_7[0][0]                    
__________________________________________________________________________________________________
average (Average)               (None, 2, 2)         0           dense_14[0][0]                   
                                                                 dense_15[0][0]                   
==================================================================================================
Total params: 12
Trainable params: 12
Non-trainable params: 0
__________________________________________________________________________________________________
'''
mo_4.predict(x[np.newaxis])
'''
array([[[3., 3.],
        [7., 7.]]], dtype=float32)
'''


# concatenate
# 자동적으로 맞춰줄때 열로 붙여진다. (계산의 유연성을 위해서)
# (a*n)(n*b) -> n은 같아야한다. 따라서 열로 맞춘다. 
layer5 = Concatenate()([layer1,layer2])
mo_5 = Model(input1, layer5)
mo_5.summary()
'''
Model: "model_9"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_7 (InputLayer)            [(None, 2, 2)]       0                                            
__________________________________________________________________________________________________
dense_14 (Dense)                (None, 2, 2)         6           input_7[0][0]                    
__________________________________________________________________________________________________
dense_15 (Dense)                (None, 2, 2)         6           input_7[0][0]                    
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 2, 4)         0           dense_14[0][0]                   
                                                                 dense_15[0][0]                   
==================================================================================================
Total params: 12
Trainable params: 12
Non-trainable params: 0
__________________________________________________________________________________________________
'''
mo_5.predict(x[np.newaxis])
'''
array([[[3., 3., 3., 3.],
        [7., 7., 7., 7.]]], dtype=float32)
'''


# dot
layer6 = Dot(1)([layer1,layer2])
mo_6 = Model(input1, layer6)
mo_6.summary()
'''
Model: "model_10"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_7 (InputLayer)            [(None, 2, 2)]       0                                            
__________________________________________________________________________________________________
dense_14 (Dense)                (None, 2, 2)         6           input_7[0][0]                    
__________________________________________________________________________________________________
dense_15 (Dense)                (None, 2, 2)         6           input_7[0][0]                    
__________________________________________________________________________________________________
dot (Dot)                       (None, 2, 2)         0           dense_14[0][0]                   
                                                                 dense_15[0][0]                   
==================================================================================================
Total params: 12
Trainable params: 12
Non-trainable params: 0
__________________________________________________________________________________________________
'''
mo_6.predict(x[np.newaxis])
'''
mo_6.predict(x[np.newaxis])
mo_6.predict(x[np.newaxis])
array([[[58., 58.],
        [58., 58.]]], dtype=float32)
'''


# lambda
# lambda는 함수형 패러다임에 첫번째 인자로 들어갈 때 많이 쓴다.
layer7 = Lambda(lambda x: x**2)(layer1,layer2)
mo_7 = Model(input1, layer7)
mo_7.summary()
# multi inputs => Connected to 
'''
Model: "model_14"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_7 (InputLayer)            [(None, 2, 2)]       0                                            
__________________________________________________________________________________________________
dense_14 (Dense)                (None, 2, 2)         6           input_7[0][0]                    
__________________________________________________________________________________________________
dense_15 (Dense)                (None, 2, 2)         6           input_7[0][0]                    
__________________________________________________________________________________________________
lambda_8 (Lambda)               (None, 2, 2)         0           dense_14[0][0]                   
==================================================================================================
Total params: 12
Trainable params: 12
Non-trainable params: 0
__________________________________________________________________________________________________
'''
mo_7.predict(x[np.newaxis])
'''
mo_7.predict(x[np.newaxis])
mo_7.predict(x[np.newaxis])
array([[[ 9.,  9.],
        [49., 49.]]], dtype=float32)
'''
```



{% include gallery id="gallery" caption="" %}




{% capture notice-2 %}

{% endcapture %}

<div class="notice">{{ notice-2 | markdownify }}</div>



