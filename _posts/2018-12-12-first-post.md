---
excerpt: "This post should [...]"
header:
  overlay_image: /assets/images/2computers.jpg
  overlay_filter: 0.5 # same as adding an opacity of 0.5 to a black background
  caption: "Photo credit: [**Unsplash**]"
  actions:
    - label: "Original"
      url: "https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html "
title: "초보자를 위한 판다스 (10 Minutes to Pandas!)"
date: 2019-02-07 10:28:28 -0400
categories: python jupyter
tags: pandas
---

## 10 Minutes to pandas
### 초보자를 위한 Pandas 소개
* 자세한 내용은'[cookbook](https://pandas.pydata.org/pandas-docs/stable/user_guide/cookbook.html#cookbook)'참조 
* pandas 0.24.1 documentation


#### 1. 라이브러리

```python
import numpy as np
import pandas as pd
```

* 최종 결과뿐만 아니라 모든 Output 출력을 표시하기

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

## 2. 객체 생성
값 리스트를 전달하여 pandas가 기본 정수 인덱스를 생성하도록 Series 만들기
```python
s = pd.Series([1,3,5, np.nan, 6,8])

s
Out[4]:
0    1.0
1    3.0
2    5.0
3    NaN
4    6.0
5    8.0
dtype: float64
```

datetime 인덱스와 레이블된 열이 있는 NumPy배열을 전달하여 DataFrame 만들기
```python
dates=pd.date_range('20130101',periods=6)
dates
df=pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
df

DatetimeIndex(['2013-01-01', '2013-01-02', '2013-01-03', '2013-01-04',
               '2013-01-05', '2013-01-06'],
              dtype='datetime64[ns]', freq='D')
Out[5]:
A	B	C	D
2013-01-01	-1.313690	0.374085	0.143973	-0.624099
2013-01-02	-0.659125	-0.595977	0.403210	0.814707
2013-01-03	0.105771	0.338891	-0.404629	0.032208
2013-01-04	0.376774	-0.035958	-0.659694	0.261746
2013-01-05	0.317915	1.052522	-0.794585	1.480261
2013-01-06	0.579203	1.709605	-0.546063	0.336379
```

직렬로 변환 할 수 있는 개체의 지시를 전달하여 DataFrame 만들기
```python
df2=pd.DataFrame({'A':1.,
                 'B':pd.Timestamp('20130102'),
                 'C':pd.Series(1, index=list(range(4)),dtype='float32'),
                 'D':np.array([3]*4,dtype='int32'),
                 'E':pd.Categorical(["test","train","test","train"]),
                 'F':'foo'})

df2
Out[6]:
A	B	C	D	E	F
0	1.0	2013-01-02	1.0	3	test	foo
1	1.0	2013-01-02	1.0	3	train	foo
2	1.0	2013-01-02	1.0	3	test	foo
3	1.0	2013-01-02	1.0	3	train	foo
```
