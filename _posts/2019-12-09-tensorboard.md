---
excerpt: "(1) Tensorboard"
header:
  overlay_image: /assets/images/2computers.jpg
  overlay_filter: 0.5 # same as adding an opacity of 0.5 to a black background
  caption: "Photo credit: [**Unsplash**]"
  actions:
    - label: "Reference"
      url: "https://docs.python.org/3/"
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



{% include gallery id="gallery4" caption="validation_curve" %}





{% capture notice-2 %}

{% endcapture %}

<div class="notice">{{ notice-2 | markdownify }}</div>






{% capture notice-2 %}

{% endcapture %}

<div class="notice">{{ notice-2 | markdownify }}</div>




---
**무단 배포 금지** 
{: .notice--danger}
