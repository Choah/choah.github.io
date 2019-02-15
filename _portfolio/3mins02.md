---
title: "[3분 딥러닝]Ch.2 텐서플로 설치와 주피터 노트북"
excerpt: "파이선 및 필수 라이브러리 설치하기"
header:
  image: /assets/images/2computers.jpg
  teaser: /assets/images/2computers.jpg
sidebar:
  - title: "3분 딥러닝"
    image: /assets/images/3min.jpg
    image_alt: "logo"
    text: "텐서플로맛"
  - title: "지은이"
    text: "골빈해커(김진중)"
gallery:
  - url: /assets/images/3min.jpg
    image_path: assets/images/3min.jpg
    alt: "placeholder image 1"
  - url: /assets/images/3min.jpg
    image_path: assets/images/3min.jpg
    alt: "placeholder image 2"
  - url: /assets/images/3min.jpg
    image_path: assets/images/3min.jpg
    alt: "placeholder image 3"
---

## 2.1 파이썬 및 필수 라이브러리 설치하기

  - [파이썬 3.6 버전](http://www.python.org/downloads/) (윈도우에서는 반드시 파이썬 3.5 이상, 64비트용을 사용해야 합니다.) 
  - 텐서플로 1.2 버전 

  파이썬을 잘 설치했다면 텐서플로 설치는 매우 쉽습니다. 리눅스의 터미널 또는 윈도우의 명령 프롬프트에서 pip3 명령어를 사용하면 됩니다. 
  ```python
  C:\> pip3 install --upgrade tensorflow
  ``` 
  만약 엔비아 GPU를 사용하고 있다면, 엔비디아 사이트에서 CUDA 툴킷을 설치한 뒤 다음의 명렁어로 쉽게 GPU 가속을 지원하는 텐서플로를 설치할 수 있습니다. ([CUDA 툴킷 문서] (http://docs.nvidia.com/cuda)참조).   
  ```python
  C:\> pip3 install --upgrade tensorflow-gpu
  ```
  그리고 이 책에서 사용하는 라이브러리들을 설치합니다. 
  ```python
  C:\> pip3 install numpy matplotlib pillow
  ```
  * numpy-수치 계산 라이브러리
  * matplotlib-그래프 출력 라이브러리
  * pillow-이미지 처리 라이브러리 
  
**Note:** 저는 개인적으로 아나콘다를 이용해서 설치합니다. 아니면 주피터노트에서 !pip3 를 이용하여서 설치할 수 있습니다.   
{: .notice--danger}

## 2.2 텐서플로 예제 내려받고 실행해보기 
 
 [깃허브 저장소](https://github.com/golbin/TensorFlow-Tutorials)에서 모든 예제를 다운받을 수 있습니다. 
 ```python
 C:\> git clone https://github.com/golbin/TensorFlow-Tutorials.git
 ```
 
## 2.3 주피터 노트북 

  주피터 노트북은 웹브라우저상에서 파이썬 코드를 단계적으로 쉽게 실행하고, 시각적으로 빠르게 확인해볼 수 있도록 도와주는 프로그램입니다. 
  설치는 일반적인 파이썬 패키지와 같이 pip3를 이용하면 됩니다. 
  ```python
  C:\> pip3 install jupyter
  ```
  그런 다음 프로젝트를 진행할 폴더의 터미널 또는 명령 프롬프트에서 다음 명령을 실행합니다. (폴더에서 `shift` + 오른쪽 클릭 --> 'PowerSheel 창 열기')
  ```python
  C:\> pip3 jupyter notebook
  ```
  그러면 웹브라우저가 열리면서 주피터 노트북이 실행될 것입니다. 
 

{% include gallery caption="This is a summary of '3minutes Deep learning'." %}
