---
excerpt: "(1) 파이썬 소개 (2) 파이썬 장단점 (3) 파이썬 Data type (4) 기본 코딩"
header:
  overlay_image: /assets/images/2computers.jpg
  overlay_filter: 0.5 # same as adding an opacity of 0.5 to a black background
  caption: "Photo credit: [**Unsplash**]"
  actions:
    - label: "Reference"
      url: "https://docs.python.org/ko/3/index.html "
title: "[Python] 기초1"
date: 2019-07-01 00:00:00 -0400
categories: python jupyter
tags: lecture CS
---

---
**본 과정은 제가 파이썬 수업을 들으면서 배운 내용을 복습하는 과정에서 적어본 것입니다.<br> 틀린 부분이 있다면 댓글에 남겨주시면 고치도록 하겠습니다.<br> 확실하지 않은 내용은 '(?)'을 함께 적었으니 그 내용을 아신다면 댓글에 남겨주시면 감사하겠습니다.**
{: .notice--warning}

## 1) Python 
### python의 철학
```python
import this
import antigravity
```
처음 python을 배운다면 먼저 python의 철학을 알아야하지 않을까? 
* The Zen of Python, by Tim Peters <br>
  Beautiful is better than ugly.<br>
  Explicit is better than implicit.<br>
  Simple is better than complex.<br>
  Complex is better than complicated.<br>
  Flat is better than nested.<br>
  Sparse is better than dense.<br>
  Readability counts.<br>
  Special cases aren't special enough to break the rules.<br>
  Although practicality beats purity.<br>
  Errors should never pass silently.<br>
  Unless explicitly silenced.<br>
  In the face of ambiguity, refuse the temptation to guess.<br>
  There should be one-- and preferably only one --obvious way to do it.<br>
  Although that way may not be obvious at first unless you're Dutch.<br>
  Now is better than never.<br>
  Although never is often better than *right* now.<br>
  If the implementation is hard to explain, it's a bad idea.<br>
  If the implementation is easy to explain, it may be a good idea.<br>
  Namespaces are one honking great idea -- let's do more of those!<br>

## 2) python 장단점
### 장점 
- 멀티형 패러다임: 하나 이상의 프로그래밍 패러다임을 지원하는 프로그래밍 언어 (절차지향, 객체지향, 함수형 패러다임) 
- 객체지향 패러다임: 프로그래머들이 프로그램을 상호작용하는 객체들의 집합으로 볼 수 있게 함 (대표적인 프로그래밍 언어: Java) 객체지향 프로그래밍에 overloading(중복 정의)이 없다고 하는데 지원함 <- 구조적 언어에서 발전된 것 
- 함수형 패러다임: 상태값을 지니지 않는 함수값들의 연속으로 생각할 수 있도록 해줌 
- 절차 지향 패러다임: 기능(함수) 중심 (대표적인 프로그래밍 언어: C)

- 다이나믹 타입(동적 타이핑): 실행 시간에 자료형을 검사함. 대부분의 언어(C,java)는 자료 타입을 입력해야하지만 파이썬은 자료 타입을 안써도 자료 할당하는데 가능 

#### 할당문 특징
- 할당 (assignments) 
    * 종류: 할당문 ((식별자, 이름, 변수[반만 맞다]) a=1 (표현식: 하나의 결과값으로 축약할 수 있는 것)) 
    * 식별자 이름 규칙 PEP 8 coding style guide <br>
                                               * 카멜방식(두단어 이상 연결할 때 두번째 문자를 대문자) 클래스이름 만들 때  ex) moonBeauty <br>
                                               * 파스칼방식 ex) MoonBeauty <br>
                                               * snake방식-함수이름 ex) moon_beauty  <br> 
                                               * 갈래상 상수: 다 대문자 (?) <br>
                                               * 헝가리안방식: 접두어에 자료형을 붙이는 것 ex) strName, bBusy, szName (별로 사용 안함) <br>
                                               * 이름에 _(*)가 붙어 있으면 import 할 때 실행이 인된다. 이러한 규칙들은 [언어레퍼런스](https://docs.python.org/ko/3/reference/index.html)에서 찾아볼 수 있다. <br>
                                               * 표현식: ex) 3 if a>0 else 5 <br>
                                               * 표현식의 종류: 람다- 함수식, etc <br>
    * a=1 가 만들어지면
      1) 메모리의 공간 할당  
      2) 메모리 이름 생성 - binding 되는 것 
#### 파이썬 특징 
- complier 안 한 언어 -> interpreted language (JIT 없는 dynamic 인터프리터 언어) 
- python 3 부터 유니코드 
- python은 C로 만들어진 언어 

### 단점 
- 느리다: numpy - 내부구조 c와 연동, 하드웨어 지원 필요
- 모바일 지원이 약하다. 
- py game이 생겨서 게임을 만들 수는 있음 

#### 파이썬에서 지원하는 프로그램
- 코틀린 자바 안드로이드 앱
- 웹기반 파이썬: skulpt, brython, PyPy(파이썬으로 만든 파이썬)
- tenserflow: 속도가 느려서 실제 회사에서는 잘 안씀

## 3) Data Type 13가지 
- 수치형 (4가지): 정수, 부동소수점, 복합, 불린
integers(python은 integer 1가지), floating, complex numbers, boolean[1,0] <- integers이랑 똑같다.  
- python integers은 overflow가 있다.-> 다른방식으로: 정수의 범위는 있는데, 정수의 범위를 넘어서면 자동적으로 메모리를 확대-> (import sys)에서 확인해 볼 수 있다.
- 정수는 4가지 표현방식 - 2, 8, 10, 16진수
* a= 0b (2진수) 따라서 a=0b23 안된다
* a=0o (8진수) 
* a=0x (16진수)

- 문자형 (3가지)
bytes & bytearray(binary), char, str

- 정적 타이핑: 변수를 선언하고 초기화할 때 타입을 명시(직접 작성)해주어야 한다는 의미
- 동적 타이핑: 데이터 타입을 명시(직접 작성)하지 않아도 컴퓨터가 초기화 값을 이용해 자동으로 데이터 타입을 할당해준다는 의미 [[참고](https://christoper31.postype.com/post/1891094)]

- 가변형(mutable): 변경이 가능한 데이터의 성질 - ex) list, dict, set, bytearray (데이터 추가, 삭제 수정이 가능한 메서드를 가짐)
- 불변형(immutable): 변경이 불가능한 데이터 성질 - ex) str, tuple, byte

- Literal: 몇몇 내장형들의 상숫값을 위한 표기법- 숫자, 문자, 문자열 등의 상수값에 대한 데이터 표기 
    - 정수는 리터럴이 없다 
    - float은 리터럴 존재
* 컨테이너 타입의 리터럴: bytes, list, tuple, set, dictionary

- Sequence: 순서가 있는 복수의 데이터 모음 (list, tuple, range) 

```ruby 
가상환경 만드는 3가지 방법 (?)
1) 도코로 만드는 사람
```

## (4) 위 내용 관련 코딩 
### python keyword
```python
import keyword
dir(keyword)

['__all__',
 '__builtins__',
 '__cached__',
 '__doc__',
 '__file__',
 '__loader__',
 '__name__',
 '__package__',
 '__spec__',
 'iskeyword',
 'kwlist',
 'main']
 ```
* kw (keyword)
* __: magic, special function
- keyword 가 이정도로 작다는 것은 쉽게 배울 수 있다는 것 

### 메모리 생성
```python
a=1
b=1000
id(a) # 140721169535808
id(b) # 1643931089072
hex(id(a)) # '0x7ffc33529340'
hex(id(b)) # '0x17ec1edb8b0'
```
* -5, 256 사이는 메모리 값이 똑같다 : 재활용하게 만들어 놓은 것 

```python
from sys import intern

a=1000
b=1000

a==b # True
a is b # False
```
* memory가 같은지 안 같은지까지 보는게 a is b: interning이라는 기법

### 유리수 
#### 정수
```python
import sys # 정수 가장 큰 수
sys.maxsize # 9223372036854775807
9223372036854775807+1 # 내부적으로 처리하는 수는 급격히 떨어진다.

a=100_000_000 
```
* 숫자가 너무 크면 구분자로 언더바를 쓴다.
* 하나의 값으로 축약할 수 있기 때문에 이것도 할당문의 표현식으로 쓴다.

#### 절대값 함수

```python
a=-1
a.__abs__() # 1
-1 .__abs__() # -1: -1.abs() 이면 float으로 처리되서 한칸 뛰어쓰면 에러가 안난다.
(-1).__abs__() # 1
```

### 부동소수점 (float)
```python
sys.float_info
```
- 부동소수점에 대한 정보를 알 수 있다. 
- 무한대는 부동소수점으로 표현한다.
- 부동소수점을 정확하게 표현 못한다. 

```python
a=float('nan')
a #nan
0.1+0.1+0.1 # 0.30000000000000004
```
- 숫자가 아닌 값도 소수점으로 들어간다.
- decimal, fractions 은 float보다 더 정확하다. ; 컴퓨터는 부동소수점을 읽기가 어렵다.
- numpy 는 속도도 빠르지만 부동소수점의 오류를 좀 잡아준다.


### 무리수 (float으로 표현)
```python
a=float('infinity')
a==a+1 # True

1.7976931348623157e+309 # inf
a=1.7976931348623157e+309
a==a+1 # True
```

### 복소수
```python
a= 3+2j
type(a) # complex
```

### List
```python
a=[2]
id(a) # 1643931491272
a.append(3)
a # [2, 3]
id(a) # 1643931491272 - 주소 값의 첫번째 값 메모리를 보여주기 때문에 같은 값
```



### 문자형 
#### 1번째 이슈
```python
a= u'경아'
type(a) # str

# python2 는 byte type
# python3 는 unicode (?)

a='경아'
type(a) # str
a=b'경아' # error: bytes can only contain ASCII literal characters.
# b는 byte의 약자로 쓰임

a=b'abde'
# python 은 byte를 문자로도 쓰지만 숫자로도 쓸 수 있다.
```

#### 2번째 이슈 
* 문자형은 메모리가 어떻게 될까?

```python
a='경아'
b='경아'
a==b # True
a is b # False

a='moon'
b='moon'
a is b # True - 영어는 메모리가 20자 이내 같음

# 문자형 인터닝: 공백 있으면 안된다, 20자 이내는 메모리가 같다.

a='moon i'
b='moon i'
a is b # False - 공백
```
- ==: equality 
- is: identification 

#### 3번째 이슈
- 문자열은 한꺼번에 데이터를 담는가 안담는가? - Yes
- container - 나누는 방법: - 같은 데이터 형이냐? 
                          - 순서가 중요한가? 


- sequence type: 문자열은 순서가 중요한 것
- indexing 과 slicing을 지원 - sequence type (순서가 중요한 애들) 
* 인덱싱(Indexing)이란 무엇인가를 "가리킨다"는 의미
* 슬라이싱(Slicing)은 무엇인가를 "잘라낸다"는 의미
- indexing, slicing 은 sequence형 따라서 mutable type  (?) 
* 'append', 'clear', 'copy', 'count', 'extend', 'index', 'insert', 'pop', 'remove', 'reverse' -> 모양 더하고 빼고하는 게 있으면 mutable type


- homogeneous (데이터형이 같다): str, byte & bytearray, range (3가지)
- heterogeneous (데이터형이 다르다): list 형 ex) a=[1,'a'] - mutable

- tuple: 레코드를 튜플이라고 함 (DB에서 사용하는 것) -> list와 비슷하지만 immutable 

- 순서가 없는 heterogeneous : set 
* 집합은 중복이 없고 순서가 없다. 

```python
a='abc'
len(a) # 3
a[0] # 'a' indexing 
a[4] # indexError

a[1:3] # 'bc' slice [범위를 잡는 것]
a[1:5] # 'bc' slice는 범위에 벗어나도 에러가 안난다.

a='어쩌다 오늘'
a[::-1] # '늘오 다쩌어' -> ::-마다 ::-1 역수,  
a[::2] # '어다오'
a[slice(1,3)] # '쩌다'

range(100)[50] # 50-> range는 homo and sequence 
```

### 집합(set)

```python
a={3,1,2,1,1,3} # 중복이 없다 , 순서가 없기 때문에 index, slice 불가능
a # {1, 2, 3}

b=frozenset({1,2,3}) # mutable set 이지만 동결집합(frozenset)을 이용하여 불변형으로도 사용할 수 있다.
b # frozenset({1, 2, 3})
```

```
**mutable, immutable 짝이 있을까?** 
- mutable은 바꾸는데 실수할 수 있으니까 
- python은 상수가 없다. -> 상수: 재할당을 할 수 있냐 없냐의 차이 
- python은 모든 것을 재할당 할 수 있다. 
- 따라서 immutable 중요한 역할이 되는 것 
```
 
---
**무단 배포 금지** 
{: .notice--danger}
