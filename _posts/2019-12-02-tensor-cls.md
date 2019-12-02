---
excerpt: "(1) 넘파이 (2) PIL (3) tf.data"
header:
  overlay_image: /assets/images/2computers.jpg
  overlay_filter: 0.5 # same as adding an opacity of 0.5 to a black background
  caption: "Photo credit: [**Unsplash**]"
  actions:
    - label: "Reference"
      url: "https://docs.python.org/3/"
title: "[Tensorflow] Image classification"
date: 2019-11-29 00:00:00 -0400
categories: image classification
tags: image calssifcation  
---



# 넘파이

넘파이로 데이터를 열어서 이미지 분류하기 

```python
import numpy as np
import tensorflow as tf 

# 데이터 다운받는다.
DATA_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
path = tf.keras.utils.get_file('mnist.npz', DATA_URL)

with np.load(path) as data:
    train_examples = data['x_train']
    train_labels = data['y_train']
    test_examples = data['x_test']
    test_labels = data['y_test']

# open -> __exit__
# load -> source 확인하면 어떻게 닫히는지 알 수 있다. 
```







