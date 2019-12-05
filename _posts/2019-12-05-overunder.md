---
excerpt: "(1) over/underfitting"
header:
  overlay_image: /assets/images/2computers.jpg
  overlay_filter: 0.5 # same as adding an opacity of 0.5 to a black background
  caption: "Photo credit: [**Unsplash**]"
  actions:
    - label: "Reference"
      url: "https://www.tensorflow.org/tutorials/keras/overfit_and_underfit"
title: "[deep learning] Over/Underfitting"
date: 2019-12-05 00:00:00 -0400
categories: deep_learning
tags: overfitting underfitting
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



# Overfitting / Underfitting 

overfitting 과 underfitting을 보려면 learning curve 그래프를 확인해야합니다. 

[Learning curve] 
- x: epoch or data(keras 자체로는 그리기 힘들다, 따라서 sklearn이용해서 그린다.)
- y: loss(작을수록 좋다.) or 정확도(클수록 좋다.) 

성능이 비슷하면 **'오컴의 면도날'**로 간단한 모델을 선택한다.

- underfitting: 모델을 다시 만드는게 좋다. 
- overfitting: validation이 학습될수록 올라간다. 
    - 데이터를 늘린다. 
    - 모델을 간단히 한다. 
    - feature selection (차원 축소) 
    - deeplearning: layer, node 줄인다. (dropout)



{% capture notice-2 %}
### No Free Lunch 
모델링은 다양한 시도를 통해 최적의 모델링을 찾아야한다. 

딥로닝은 성능이 좋지만, 과적합되기가 쉽다. 
- layer
- node
- epoch

**[overfitting 막는 법]**

**layers 에서 패널티를 줘서 과적합을 막을 수 있다.**
- penalty, regularizer
- l1
- l2 
- l1 & l2 두개를 쓸 수도 있다.

kernel_regularizer (l1 (|x|+|y|), l2 (x**2+y**2))

**cross_validation** <br>
- 확인하는 용도로 쓴다. 
- 데이터가 작을 때-차원의 저주로 오버피팅이 더 잘 난다. 
- 하지만 시간이 오래 걸린다는 단점이 있다. 

**dropout**  <br>
layer 뒤에 추가 가능.
activation 뒤에 추가 가능.
0.2 (20%)를 랜덤으로 빼버린다는 것.
학습속도가 느리다. 

matplotlib에서 (state machine) - 앞에 있는 가장 가까운거 붙어서 하는 것 -> tf.keras.layers.Dropout(0.2) (앞에 Dense 붙어서 실행)

**Early stopping** <br>

**Ensemble** <br>
- bagging <br>
random boostrap 방법으로 샘플을 여러 번 뽑아 각 모델을 학습시켜 결과를 집계하는 방식이다.
- boosting <br>
성능 안좋은 것에 가중치줘서 학습시키는 것 
- stacking <br>
A 알고리즘, B 알고리즘, C 알고리즘을 또 학습시켜서 나온결과로 또 학습시키는 것이다. 

{% endcapture %}

<div class="notice">{{ notice-2 | markdownify }}</div>




{% include gallery id="gallery" caption="" %}



{% capture notice-2 %}

{% endcapture %}

<div class="notice">{{ notice-2 | markdownify }}</div>



