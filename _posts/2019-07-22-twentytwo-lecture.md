---
excerpt: "(1) NumPy (2) Array (3) 차원 (4) 사칙연산 (5) 행렬 (6) 인덱싱 (7) Broadcasting
(8) Universal function (9) Numpy Quickstart Tutorial"
header:
  overlay_image: /assets/images/2computers.jpg
  overlay_filter: 0.5 # same as adding an opacity of 0.5 to a black background
  caption: "Photo credit: [**Unsplash**]"
  actions:
    - label: "Reference"
      url: "https://docs.python.org/ko/3/index.html"
title: "[Python] NumPy"
date: 2019-07-22 00:00:00 -0400
categories: python jupyter
tags: Numpy
---


---
**본 과정은 제가 파이썬 수업을 들으면서 배운 내용을 복습하는 과정에서 적어본 것입니다.<br> 틀린 부분이 있다면 댓글에 남겨주시면 고치도록 하겠습니다.<br> 확실하지 않은 내용은 '(?)'을 함께 적었으니 그 내용을 아신다면 댓글에 남겨주시면 감사하겠습니다.**
{: .notice--warning}

## 1) NumPy
### 넘파이 장점

- 속도가 빠르다
    - C언어로 만들어졌다.
    - 벡터화된 array기법
    - 팩토리 메소드: 클래스를 통해 인스턴스화 하지 않고 다른 함수의 힘을 빌려서 인스턴스
- 편하고 쉽다.
- GPU 지원 안한다. -> tensorflow는 GPU 지원한다.
- NumPy -> tensorflow, pandas, pytorch 넘파이 기반으로 만들어짐 
- statsmodel
- scikit-image
- scikit-learn
- pandas
- matplotlib

```python
import numpy as np
```

{% capture notice-2 %}
**복습/참조**
* for문을 쓰지 않도록 도와주는 것
1. 재귀함수
2. iterator, generator
3. comprehension
4. map, reduce
5. high other function 
{% endcapture %}

<div class="notice">
  {{ notice-2 | markdownify }}
</div>

## 2) Array 

```python
import array
```

{% capture notice-2 %}
**Array**
- 파이썬 공식 라이브러리
- homogeneous
- mutable
{% endcapture %}

<div class="notice">
  {{ notice-2 | markdownify }}
</div>

```python
# 1 차원
np.array([1,2],[3,4])
# 투플로 적어도 리스트 형식으로 바뀜 

# 2 차원
a = np.array([[1,2],[3,4]])
a

# 3 차원
a = np.array([[[1,2],[3,4]],[[1,2],[3,4]]])
a

type(a)

a = np.array(['1',2,3]) # 타입을 다 str로 바꿔버림 
---
array([1, 3])

array([[1, 2],
       [3, 4]])
       
array([[[1, 2],
        [3, 4]],

       [[1, 2],
        [3, 4]]])

numpy.ndarray
```

{% capture notice-2 %}
**인스턴스화 3가지**
1) 클래스 인스턴스화 방식
2) iteral 방식
3) factory 메소드 방식
{% endcapture %}

<div class="notice">
  {{ notice-2 | markdownify }}
</div>

## 3) 차원 

```python
a.ndim # 차원알려주는 것 
len(a) # ndim 이랑 비슷함
a.shape # 행렬 연산할 때 모양 맙춰야한다. 
a.flags
---
3
3
(3, 3, 3)
  C_CONTIGUOUS : True
  F_CONTIGUOUS : False
  OWNDATA : False
  WRITEABLE : True
  ALIGNED : True
  WRITEBACKIFCOPY : False
  UPDATEIFCOPY : False
```
{% capture notice-2 %}
**( 1,2,3,4 )**
- C: 1,2,3,4 - 메모리 상 (12,4)
1,2
3,4

- Fortran 방식 (4,12)
1,3,2,4
{% endcapture %}

<div class="notice">
  {{ notice-2 | markdownify }}
</div>

```python
a.itemsize # 몇 비트인지 알려주는 것
a.dtype
---
x = [1,2,3,4]
y = [2,3,4,5]
```

## 4) 사칙연산 

```python
x = [1,2,3,4]
y = [2,3,4,5]
x+y
# zip 
[a+b for a,b, in zip(x,y)]
---
[1, 2, 3, 4, 2, 3, 4, 5]
[3, 5, 7, 9]
```
```python
a = np.array([[1,2],[3,4]])
b = np.array([[1,2],[3,4]])
# element wise 방식 
a+b
a*b
a.T
---
array([[2, 4],
       [6, 8]])]
array([[ 1,  4],
       [ 9, 16]])
array([[1, 3],
       [2, 4]])
```

## 5) 행렬 

```python
a = np.array([[1,2],[3,4]])
a.strides # (2행 2열)
# 실제는 한개지만 스트라이드를 이용해서 bytes로 알려줌. 
# 1 memory block -> 4 bytes 
---
(8, 4)
```
* axis: ndarray(N차원 배열 객체) 

```python
# range 비슷한 것 
a = np.arange(10)

a = a.reshape(2,5) # 2행 5열 
# reshape 은 copy 개념이라서 재할당 해야한다.
a

a[0]+a[1]
a.sum(axis=0)
a.sum(axis=1)
---
array([[0, 1, 2, 3, 4],
       [5, 6, 7, 8, 9]])
array([ 5,  7,  9, 11, 13])
array([ 5,  7,  9, 11, 13])
array([10, 35])
```

{% capture notice-2 %}
**복습/참조**
mutable
1) 자기자신 바뀌지만 리턴값이 nan
2) 리턴값이 있지만 안 바뀌는것
3) 리턴값도 있고 바뀌는 것 
{% endcapture %}

<div class="notice">
  {{ notice-2 | markdownify }}
</div>

```
sum(x for x in range(10000000))

np.sum(np.arange(10000000))
넘파이기반 아래가 훨씬 빠르다.
```

```python
a= np.zeros((3,4)) # 0으로 만들수있다.
a
c= np.ones((3,4))
c
b = np.ones_like(a)
b
a= np.full((3,4), 6)
a
b = np.eye(3) # 단위 행렬은 정방행렬만 단위행렬이다.
b
c= np.identity(3) # 단위행렬
c
a= np.tri(3) # 상삼각행렬
a
a=np.tril((3,3)) # 상삼각행렬
a
b=np.triu((3,3)) # 하삼각행렬
b
np.transpose(b)
---
array([[0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.]])
array([[1., 1., 1., 1.],
       [1., 1., 1., 1.],
       [1., 1., 1., 1.]])
array([[1., 1., 1., 1.],
       [1., 1., 1., 1.],
       [1., 1., 1., 1.]])       
array([[6, 6, 6, 6],
       [6, 6, 6, 6],
       [6, 6, 6, 6]])
array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])       
array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])
array([[1., 0., 0.],
       [1., 1., 0.],
       [1., 1., 1.]])
array([[3, 0],
       [3, 3]])
array([[3, 3],
       [0, 3]])
array([[3, 0],
       [3, 3]])
```


## 6) 인덱싱 

뽑아내는 것 
1) 인덱싱, 슬라이싱 -> sequence 타입 (numpy도 시퀀스타입) 
2) KEY 

```python
a = np.arange(10).reshape(2,5)
a

a=np.arange(27).reshape(3,3,3)
a

a[1,:]
a[1,:][1,:]
a[:,1]
a[:,1][:,1]

a[1,1,]
---
array([[0, 1, 2, 3, 4],
       [5, 6, 7, 8, 9]])
array([[[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8]],

       [[ 9, 10, 11],
        [12, 13, 14],
        [15, 16, 17]],

       [[18, 19, 20],
        [21, 22, 23],
        [24, 25, 26]]])
array([[ 9, 10, 11],
       [12, 13, 14],
       [15, 16, 17]])
array([12, 13, 14])
array([[ 3,  4,  5],
       [12, 13, 14],
       [21, 22, 23]])
array([ 4, 13, 22])

array([12, 13, 14])
```

```python
a= np.arange(10)
a[(a>3) & (a<8)] # | or, & and
---
array([4, 5, 6, 7])
```


## 7) Broadcasting
```python

```


## 8) Universal function 

```python

```

## 9) Numpy Quickstart Tutorial

```python

```

 
---
**무단 배포 금지** 
{: .notice--danger}
