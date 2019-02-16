---
excerpt: "초보자를 위한 판다스!"
header:
  overlay_image: /assets/images/2computers.jpg
  overlay_filter: 0.5 # same as adding an opacity of 0.5 to a black background
  caption: "Photo credit: [**Unsplash**]"
  actions:
    - label: "Reference"
      url: "https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html "
title: "[Part.2] 초보자를 위한 판다스 (10 Minutes to Pandas!)"
date: 2019-02-13 10:28:28 -0400
categories: python jupyter
tags: pandas
---

## 10 Minutes to pandas
### 초보자를 위한 Pandas 소개
* 자세한 내용은'[cookbook](https://pandas.pydata.org/pandas-docs/stable/user_guide/cookbook.html#cookbook)'참조 
* pandas 0.24.1 documentation

## 7. 병합
### Concat
pandas는 Series, DataFrame 및 Panel 객체를 조인/병합 유형 작업의 경우 인덱스 및 관계 대수 기능에 대한 다양한 유형의 논리로 손쉽게 결합할 수 있는 다양한 기능을 제공합니다.
```python
df=pd.DataFrame(np.random.randn(10,4))
df

#break it into pieces
pieces=[df[:3],df[3:7],df[7:]]
pd.concat(pieces)

Out[65]:
0	1	2	3
0	-0.936497	0.035785	-1.948344	-0.872485
1	-0.901573	-0.406977	0.282420	1.995679
2	0.350583	0.023508	-0.231986	-0.637906
3	0.614168	-0.807614	1.062147	0.174521
4	2.325358	0.375285	-0.004919	-0.772673
5	0.753493	0.153424	-1.026853	0.456579
6	-2.457705	-1.039590	-0.252836	-1.177802
7	-0.152389	3.716780	-0.278010	-1.362557
8	-0.073911	-1.453387	-2.222094	-1.114078
9	-0.786243	-0.570721	-0.835087	-0.736105
Out[65]:
0	1	2	3
0	-0.936497	0.035785	-1.948344	-0.872485
1	-0.901573	-0.406977	0.282420	1.995679
2	0.350583	0.023508	-0.231986	-0.637906
3	0.614168	-0.807614	1.062147	0.174521
4	2.325358	0.375285	-0.004919	-0.772673
5	0.753493	0.153424	-1.026853	0.456579
6	-2.457705	-1.039590	-0.252836	-1.177802
7	-0.152389	3.716780	-0.278010	-1.362557
8	-0.073911	-1.453387	-2.222094	-1.114078
9	-0.786243	-0.570721	-0.835087	-0.736105
```

### 결합

SQL 스타일 병합
```python
left=pd.DataFrame({'key':['foo','foo'],'lval':[1,2]})
right=pd.DataFrame({'key':['foo','foo'],'rval':[4,5]})

left
right

pd.merge(left,right,on='key')

Out[66]:
key	lval
0	foo	1
1	foo	2
Out[66]:
key	rval
0	foo	4
1	foo	5
Out[66]:
key	lval	rval
0	foo	1	4
1	foo	1	5
2	foo	2	4
3	foo	2	5
```

다른 예시
```python
left=pd.DataFrame({'key':['foo','bar'],'lval':[1,2]})
right=pd.DataFrame({'key':['foo','bar'],'rval':[4,5]})

left
right

pd.merge(left,right, on='key')

Out[67]:
key	lval
0	foo	1
1	bar	2
Out[67]:
key	rval
0	foo	4
1	bar	5
Out[67]:
key	lval	rval
0	foo	1	4
1	bar	2	5
```

### 추가

데이터프레임에 행 추가
```python
df=pd.DataFrame(np.random.randn(8,4),columns=['A','B','C','D'])
df

s=df.iloc[3] #index '3'
df.append(s,ignore_index=True)

Out[73]:
A	B	C	D
0	2.835013	1.367990	1.000113	0.033557
1	1.642647	0.735804	-1.374949	-0.451156
2	0.316412	-0.493159	0.956598	-0.695106
3	0.226823	-1.536158	0.818281	-0.871151
4	-2.216179	0.853335	-1.642484	-0.732483
5	-0.721202	0.678233	-0.973721	0.837790
6	-0.698786	0.762075	2.243878	0.473854
7	-0.365788	1.036647	-0.291522	-1.256139
Out[73]:
A	B	C	D
0	2.835013	1.367990	1.000113	0.033557
1	1.642647	0.735804	-1.374949	-0.451156
2	0.316412	-0.493159	0.956598	-0.695106
3	0.226823	-1.536158	0.818281	-0.871151
4	-2.216179	0.853335	-1.642484	-0.732483
5	-0.721202	0.678233	-0.973721	0.837790
6	-0.698786	0.762075	2.243878	0.473854
7	-0.365788	1.036647	-0.291522	-1.256139
8	0.226823	-1.536158	0.818281	-0.871151
```

## 8. 그룹화

### "그룹화"는 다음 단계 중 하나 이상의 프로세스를 포함하는 것을 말합니다.
* 몇 가지 기준에 따라 그룹으로 데이터 '분할'
* 독립적으로 각 그룹에 기능 '적용'
* 결과를 데이터 구조로 '결합'

```python
df=pd.DataFrame({'A':['foo','bar','foo','bar','foo','bar','foo','foo'],
                'B':['one','one','two','three','two','two','one','three'],
                'C':np.random.randn(8),
                'D':np.random.randn(8)})
df

Out[76]:
A	B	C	D
0	foo	one	2.241761	1.644150
1	bar	one	1.481125	-0.138093
2	foo	two	2.315399	-1.089430
3	bar	three	0.494300	-0.910367
4	foo	two	-1.566179	0.645582
5	bar	two	0.345112	-0.177249
6	foo	one	0.427744	-1.611746
7	foo	three	-1.834687	1.673733
```

그룹화 한 다음 결과 그룹에 sum() 함수를 적용합니다.
```python
df.groupby('A').sum()

Out[77]:
C	D
A		
bar	2.320536	-1.225709
foo	1.584038	1.262290
```

여러 열로 그룹화하면 계층적 인덱스가 형성되고 다시 sum함수를 적용할 수 있습니다.
```python
df.groupby(['A','B']).sum()

Out[78]:
C	D
A	B		
bar	one	1.481125	-0.138093
three	0.494300	-0.910367
two	0.345112	-0.177249
foo	one	2.669505	0.032404
three	-1.834687	1.673733
two	0.749220	-0.443848
```

## 9. Reshaping
### stack
```python
tuples=list(zip(*[['bar','bar','baz','baz',
                  'foo','foo','qux','qux'],
                 ['one','two','one','two',
                 'one','two','one','two']]))
index=pd.MultiIndex.from_tuples(tuples, names=['first','second'])
df=pd.DataFrame(np.random.randn(8,2),index=index, columns=['A','B'])
df2=df[:4]
df2

Out[81]:
A	B
first	second		
bar	one	0.640996	-2.115345
two	-0.562975	-1.098557
baz	one	-0.799104	0.858243
two	1.120342	0.239564
```

stack()은 DataFrame의 열에 있는 레벨을 "압축"합니다.
```python
stacked=df2.stack()
stacked

Out[82]:
first  second   
bar    one     A    0.640996
               B   -2.115345
       two     A   -0.562975
               B   -1.098557
baz    one     A   -0.799104
               B    0.858243
       two     A    1.120342
               B    0.239564
dtype: float64
```

"stacked" DataFrame이나 Series(MultiIndex를 가지고있는인덱스)인 경우, stack()의 반대 연산은 unstack()이며, 디폴트 값으로 마지막 레벨을 unstack 합니다.
```python
stacked.unstack()
stacked.unstack(1)
stacked.unstack(0)

Out[83]:
A	B
first	second		
bar	one	0.640996	-2.115345
two	-0.562975	-1.098557
baz	one	-0.799104	0.858243
two	1.120342	0.239564
Out[83]:
second	one	two
first			
bar	A	0.640996	-0.562975
B	-2.115345	-1.098557
baz	A	-0.799104	1.120342
B	0.858243	0.239564
Out[83]:
first	bar	baz
second			
one	A	0.640996	-0.799104
B	-2.115345	0.858243
two	A	-0.562975	1.120342
B	-1.098557	0.239564
```

### Pivot Tables
```python
df=pd.DataFrame({'A':['one','one','two','three']*3,
                'B':['A','B','C']*4,
                'C':['foo','foo','foo','bar','bar','bar']*2,
                'D':np.random.randn(12),
                'E':np.random.randn(12)})
df

Out[85]:
A	B	C	D	E
0	one	A	foo	-0.545239	0.601491
1	one	B	foo	1.635836	-0.058278
2	two	C	foo	0.266583	-0.664336
3	three	A	bar	0.533973	0.193423
4	one	B	bar	0.502112	0.070346
5	one	C	bar	0.948488	-1.854831
6	two	A	foo	-2.469761	1.168192
7	three	B	foo	-2.207726	1.249103
8	one	C	foo	-0.498839	-1.149512
9	one	A	bar	1.058183	0.676254
10	two	B	bar	-0.165524	0.295981
11	three	C	bar	1.268656	0.816636
```
```python
pd.pivot_table(df,values=['D','E'],index=['A','B'],columns=['C'])

Out[90]:
D	E
C	bar	foo	bar	foo
A	B				
one	A	1.058183	-0.545239	0.676254	0.601491
B	0.502112	1.635836	0.070346	-0.058278
C	0.948488	-0.498839	-1.854831	-1.149512
three	A	0.533973	NaN	0.193423	NaN
B	NaN	-2.207726	NaN	1.249103
C	1.268656	NaN	0.816636	NaN
two	A	NaN	-2.469761	NaN	1.168192
B	-0.165524	NaN	0.295981	NaN
C	NaN	0.266583	NaN	-0.664336
```

## 10. 시계열
pandas는 빈도 전환(예시: 초단위 데이터에서 5분 데이터로 바꾸는 경우) 중에 리샘플링 동작을 수행하기 위한 간단하고도 강력한, 효율적인 기능을 갖는다. 이는 금융에서 많이 볼 수 있다. 하지만 그 외에도 여러 곳에서 시계열이 쓰인다.
```python
rng=pd.date_range('1/1/2012',periods=100,freq='S') #freq=빈도, 's'=seconds
ts=pd.Series(np.random.randint(0,500,len(rng)),index=rng)
ts.resample('5Min').sum()

Out[99]:
2012-01-01    26154
Freq: 5T, dtype: int32
```
