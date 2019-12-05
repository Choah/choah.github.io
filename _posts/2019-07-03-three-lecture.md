---
excerpt: "(1) 함수 (2) iterator (3) generator (4) comprehension (5) 성능비교 (6) 재귀함수(recursive call) (7) map (8) filter "
header:
  overlay_image: /assets/images/2computers.jpg
  overlay_filter: 0.5 # same as adding an opacity of 0.5 to a black background
  caption: "Photo credit: [**Unsplash**]"
  actions:
    - label: "Reference"
      url: "https://docs.python.org/ko/3/index.html "
title: "[Python] 기초3: 함수형 패러다임 고급 기법"
date: 2019-07-03 00:00:00 -0400
categories: python jupyter
tags: lecture CS functional iterator generator comprehension timeit recursive map filter
---


**본 과정은 제가 파이썬 수업을 들으면서 배운 내용을 복습하는 과정에서 적어본 것입니다.<br> 틀린 부분이 있다면 댓글에 남겨주시면 고치도록 하겠습니다.<br> 확실하지 않은 내용은 '(?)'을 함께 적었으니 그 내용을 아신다면 댓글에 남겨주시면 감사하겠습니다.**
{: .notice--warning}

* 참고 - 파이썬 HOWTO [https://docs.python.org/ko/3/howto/index.html]
  - 로깅 HOWTO
  - Regular Expression HOWTO
  - 함수형 프로그래밍 HOWTO: 모든 기반을 함수로 하는 것 -> bigdata 처리할 때 효율적 - java, pandas 

## 1) 함수 
- g(f(x))
- input과 output이 1:1 로 매칭해야한다. 
- 정의역을 한거번에 처리하는데 효율적 
- LISP (?)
- 하스켈에서 배껴왔다. 
- 함수형 프로그래밍은 함수들의 세트로 문제를 분해. 이상적으로 말하면, 함수들은 입력을 받아서 출력을 만들어내기만 하며, 주어진 입력에 대해 생성된 출력에 영향을 끼칠만한 어떠한 내부적인 상태도 가지지 않는다. 잘 알려진 함수형 언어로는 ML 계열(Standard ML, OCaml 및 다른 변형)과 하스켈이 있다.
- 함수형 패러다임은 mutable이 없다. immutable 형만 있다.

- 객체지향: class를 선언하고 인스턴스화 

- side effect 가 없어야 한다. 
- 파이썬은 기본 객체지향 패러다임을 지원해준다. 따라서 '순수 함수 패러다임' 언어는 아니다. 
- 형식적 증명 가능성: 이론적인 장점은 함수형 프로그램이 정확하다는 수학적 증명을 만드는 것이 더 쉽다는 것
- 결합성: 함수형 방식의 프로그램을 만들 때, 다양한 입력과 출력으로 여러 가지 함수를 작성하게 된다. 이러한 함수 중 일부는 불가피하게 특정 응용 프로그램에 특화될 수 있지만, 대체로 다양한 프로그램에서 유용하게 사용할 수 있다.

- iterator: 이터레이터는 데이터 스트림을 나타내는 객체

```python
a = int(input ()) # input이 또다른 함수값에 들어감
a= int(a)

# 값에 따라 중간값이 바뀌면 안된다. 
import time
def x(a=time.time):
    return a

time.time() # 현재시간을 보여주는 것 
# 함수형 패러다임에서는 쓰면 안되는 애이다. 
```

## 2) iterator
* iterable: iterator가 될 수 있는 순회가능한 것 
* 이터레이터 함수는 숨어있다. (하나씩 순회해서 나오는 것)

```python
for i in [1,2,3]: # in 다음에 이터레이터가 올 수 있다. -> 이터레이터 함수는 숨어있다. (하나씩 순회해서 나오는 것) -> i 가 이터레이터로 바뀐다.
    print (i)
    
out:
1
2
3

--------------------------

a=[1,2,3]
b= iter(a)
type(b) # list_iterator -> NEXT라는 애를 사용할 수 있다.

next(b)
list(b)
# next(b) # StopIteration: 4번째 때 에러가 난다.

out:
list_iterator
1
[2, 3]

--------------------------------

a=[1,2,3]
b= iter(a)
type(b) # list_iterator -> NEXT라는 애를 사용할 수 있다.

next(b)
next(b)
list(b)

out:
list_iterator
1
2
[3]

-------------------------------------

b # <list_iterator at 0x1a6d3c96e80>: <> 꺽쇠는 확인을 못한다.

list(b) # [] : next 는 앞에서 부터 뽑아낸다. pop은 뒤에서부터 뽑아낸다. 
# 팝은 최적화가 안되어있지만 넥스트는 최적화가 되어있어서 빠르다.
```
* Lazy 기법: 실행될 때 메모리상에 다시 뽑아서 올린다. -> 빅데이터 상에서 많이 쓴다. 
- next 할 때 메모리가 올라가기 때문에 메모리상 효율적이다. 하지만 속도의 문제가 있다. (속도가 좀 떨어짐)
- 파이썬은 이를 감안해서 속도를 올릴 수 있도록 만듦. 

```python
import dis # dis: 기계어로 분해 

def iterator_exam():
    for i in [1,2,3]:
        print(i)
        
dis.dis(iterator_exam) # iterator_exam의 흐름을 보여줌        
```

* set, str, tuple -> iterator로 만들 수 있다.

```python
a={1,2,3}
b= iter(a)
type(b) # set_iterator: set 도 이터레이터로 만들 수 있다. 

a=(1,2,3)
b= iter(a)
type(b) # tuple_iterator: tuple 도 이터레이터 가능 

a= '경아'
b= iter(a)
type(b) # str_iterator: 문자열도 이터레이터 가능 

# type을 바꾸면 역할과 기능이 다르기 때문에 타입 바꾸면 그 역할을 잃어버림 

```
* iter은 function 

```python
int('3') # iterator 와 비슷해보이지만 완전히 다른 기법 -> int 는 class
iter # iter 은 function 이다. 
```

* [참조] 객체 값을 만드는 방법 2가지
  1) 클래스 -> 인스턴스화
  2) 리터럴
      - 리터럴(literal)은 몇몇 내장형들의 상숫값을 위한 표기법


```python
# 객체 값을 만드는 방법은 2가지 
# 1) 클래스 -> 인스턴스화
a= int(1) # 기호가 하나씩 붙어있다. 기호에 따라서 내부적으로 체크된다. 이 기호를 literal이라고 한다. 

# 2) 리터럴
# 리터럴(literal)은 몇몇 내장형들의 상숫값을 위한 표기법
# a=0b2  #(2진수)
a=2e1 # 2 * 10^1
a # 20.0 # 인스턴화하지 않아도 자동으로 float으로 나타나짐 
a=2e-1 
a # 0.2

##### 
a= u'ab' # 유니코드 
a= r'ab' # raw 형태의 유니코드: 그대로 나오는 것  
a= 'ab\n' # ab: \n-> 개행문자 
a= 'ab\nc' # ab <br> c 
print(a)
a= f'ab\n' 
type(a) # str: 타입 붙여주는 것을 리터럴 
a=1.2

# 내장 객체는 앞에 소문자 카멜 방식으로 쓴다. (실행시킬 때: class 만들 때 첫글자 소문자 쓰지 말라고 한다. ) 
# 객체지향은 class 를 선언하고 인스턴스화 & 리터럴 방식도 지원 
# type 쳤을 때 나오는 애가 int 

a = int() # ?설명: / 포지셔널 온리, *args 가변 포지션, **kwargs 가변 키워드 
a # 0 -> False 
# a=int(1) -> 객체:a (인스턴스) 

b= float() # ?설명: / 포지셔널 온리 
b # 0.0

b= list()
b # []-> 아무것도 없기 때문에 False (존재론적 무)

b= set()
b # set이 원래는 없었다. 기호자체가 이미 할당 되어 있었기 때문에 이렇게 나온다. 
```

```python
iter # <function iter>: 객체가 아니고 function이다. 
# type을 바꿔주는 function은 거의 없다. 
# iter 예외적인 function ** 중요 
# type을 바꿔주려면 인스터스화해서 새로 만들어야한다. 

```

## 3) generator
- generator 만드는 2가지 방법
  1) comprehension -> tuple 
  2) yield 넣은다. 

```python
# iter와 비슷한 generator -> next 밖에 쓸 수 없다. 만드는 방식이 다르다. 
# generator 은 함수를 만들어서 만든다. 

def generator_exam():
    yield 1
    yield 2
    yield 3
    
x = generator_exam() # iterator 과 똑같다. 따라서 next 쓸 수 있다. 

next(x) # StopIteration: 4번 실행시키면 에러가 나온다. 

def generator_exam():
    yield from [1,2,3,4] # from 도 쓸 수 있다. 
     
x = generator_exam()   

next(x) # StopIteration: 5번 실행시키면 에러가 나온다. 
```
- iterator: iterable 인 애를 (객체를) 바꿀 수 있는 것 - 속도 최적화 했다. 
- generator 은 내 마음대로 next를 쓸 수 있는 애 
    - 장점: 
        - 메모리 효율성 좋다. 
        - 속도 최적화 했다. (python) 
    - 함수로 만드는 방식이 있고 ...
    - 식으로 만들 수 있다. 
    - 제너레이터는 이터레이터를 작성하는 작업을 단순화하는 특별한 클래스의 함수입니다.
    
    
## 4) 반복식 (comprehension)
- comprehension -> 하스켈에서 배낀 것 
- comprehension은 식이다.

```python
[x for x in range(10)] # for문 구조랑 같다. 

out:
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

* timeit: 여려번 실행시켜서 평균값 보여줌  

```python
%timeit [x for x in range(10)] 
# 밑에거와 비교했을 때 comprehension 방식이 훨씬 빠르다. 

%%timeit
temp=[]
for i in range(10):
    temp.append(i)
    
out: 
893 ns ± 48.8 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)

1.28 µs ± 38.5 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
```

* 반복식 for 앞에 여러가지가 올 수 있다.
* if 붙이는 방식 2가지

```python
[x+1 for x in range(10)]
[str(x) for x in range(10)] # str을 집어넣어서 어떤 값도 다 나타나게 할 수 있다. 

# if 붙이는 방식 2가지 
# 1) 뒤에 붙이는 것 
[x for x in range(10) if x%2==0] # % 나누는 것 

# python은 왼쪽에서 오른쪽으로 계산 
1+2+3+4

# 2) 앞에 붙이는 것
[x if x%2==0 else 3 for x in range(10)] # 조건식: else  (if 참 or else)

out:
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
[0, 2, 4, 6, 8]
10
[0, 3, 2, 3, 4, 3, 6, 3, 8, 3]
```

```python
# comprehension은 식이다.
[(x,y) for x in range(1,11)for y in range(10)] # list -> 중첩시킬 때 나타낼 수 있다.

out:
[(1, 0),
 (1, 1),
 (1, 2),
 (1, 3),
 (1, 4),
 (1, 5),
 (1, 6),
 (1, 7),
 (1, 8),
 (1, 9),
 (2, 0),
 (2, 1),
 (2, 2),
 (2, 3),
...
```

```python


```
*comprehension 나타낼 수 있는 3가지 
   - list, set, dictionary, but tuple (immutable) 은 안된다.
   - comprehension을 tuple로 만들면 generator

* 함수형 패러다임은 mutable이 없다. immputable 형만 있다.  
* 형식적 증명 가능성을 두기 때문에 mutable처럼 바뀌면 안된다. 

```python
# comprehension은 값을 초기화 할 때 주로 쓴다.

((x,y) for x in range(1,11)for y in range(10)) # <generator object <genexpr> at 0x000001A6D4157390>
# generator 가 나온다. 
# tuple 은 안된다. 

{x for x in range (10)} # {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}: set 

# comprehension 나타낼 수 있는 3가지 
# list, set, dictionary, but tuple (immutable) 은 안된다. -> generator나옴. (next 쓸 수 있다.)

# generator 
a= (x for x in range(10))

next(a) # 0
# 최적화시킬 때 함수형 패러다임에서 가져오는 경우가 매우 많음. 
```

```python

* for 를 쓰지 않은 기법
1) iterator & generator <- 하스켈 언어에서 가져왔다. 
    - 실제 for는 있지만 for문을 쓰지 않고 동시에 여러개를 처리하는 기법
 
* 참고: __next__ : 파이썬 객체 내 메서드 

```
## 5) 성능비교 

```python
%%timeit
temp=[]
for i in range(10):
    temp.append(i)

%%timeit # iter 쓰면 최적화되지만 파이썬은 코딩을 짧게 하는 것을 지향 
temp=[]
for i in iter(range(10)):
    temp.append(i)

out:
1.63 µs ± 246 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
1.54 µs ± 45.5 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
```
* 컴프리핸선: 데이터 만들 때 쓴다.

```python
from collections.abc import Iterable
from collections.abc import Iterator

set(dir(Iterator)) - set(dir(Iterable)) #  {'__next__'}: iterable 써야하는데 iterator 써도 된다. 

set(dir(Iterable)) - set(dir(Iterator)) # set(): 공집합 
```

```python
sum # <function sum(iterable, start=0, /)>
%timeit sum((1,2,3,4,5,6,7,8,9,10))

%timeit sum([x for x in range(1,11)]) # 밑에는 만들어서 계산했기 때문에 좀 더 오래 걸린다.

%timeit sum(x for x in range(1,11)) # generator 할 때는 예외적으로 괄호를 하나 없애도 작동한다. 

%timeit sum((x for x in range(1,11))) 
# 제너레이터 표현식은 항상 괄호 안에 작성해야 하지만 함수 호출을 알리는 괄호도 포함된다. 

out:
312 ns ± 26.3 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
1.29 µs ± 25.2 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
1.91 µs ± 441 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
1.55 µs ± 17.8 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)

```
* 중첩 

```python
a=[1,2]
b=[3,4]
[(x, y) for x in a for y in b] 
# [(1, 3), (1, 4), (2, 3), (2, 4)] 
# 식은 식끼리 중첩시킬 수 있다. 
# 식은 함수 argument로도 들어갈 수 있다.
```

## 6) 재귀함수
* 함수형 패러다임은 재귀를 좋아한다. (많이 쓴다)
    - 피보나치 구할 때 재귀 함수로 구한다. 
    - for문 을 쓰지하고 쓰는 함수가 재귀 함수 
    
* 재귀함수란 어떤 함수에서 자신을 다시 호출하여 작업을 수행하는 방식의 함수를 의미
- 다른 말로는 재귀호출, 되부름이라고 부르기도 함
- 반복문을 사용하는 코드는 항상 재귀함수를 통해 구현하는 것이 가능하며 그 반대도 가능


```python
def fibo(n):
    if n<3:
        return 1
    return fibo(n-1) + fibo(n-2)

fibo(3) # 2
```
* 깊이가 길어지면 stack이 계속 쌓이면 overflow가 될 수 있다. 
* 꼬리 재귀: 계산을 여러번하고 계속 쌓이니까 꼬리 루트를 일렬로 만드는 일을 한다. 
* 재귀를 쓰면 꼬리 재귀 기법을 만들 수 있다. 
* 파이썬은 꼬리재귀를 지원하지 않기 때문에 잘 안쓴다. 


## 7) map

* iterator, generator - 사실상 많이 응용되진 않는다.
* map 많이 응용된다.


```python
map # map(func, *iterables -> 가변 포지셔널) 

def add_one(x):
    return x+1  
    
%timeit map(add_one, [1,2,3,4,5])
# 353 ns ± 10.1 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
```

```python
list(map(add_one, [1,2,3,4,5])) # [2, 3, 4, 5, 6]

lambda :1 # 함수식이기 때문에 인자로 집어 넣을 수 있다. 
# <function __main__.<lambda>()>

%timeit list(map(lambda x : x+1, [1,2,3,4,5]))
# 1.33 µs ± 38.2 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
```
```python
# for문
%timeit
temp = []
for i in [1,2,3,4,5]:
    temp.append(i)  # 값이 안나온다. (?)

# comprehension 
%timeit [x+1 for x in [1,2,3,4,5]]
# 529 ns ± 6.97 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
```

## 8) filter
- 필터 안에 함수 넣어야한다. 
- if 보다 빠르다 

```python
filter # function, iterable
```
* predicate : True, False 를 반환하는 함수
- 필터는 true만 해준다. 

```python
tuple(filter(lambda x: x>3, [1,2,3,4,5,6])) # (4, 5, 6):  filter은 if 보다 속도도 빠르다. 
```
* reduce 

```python
from functools import reduce # reduce도 function, sequence[,] -> 순서가 있어야한다. 

reduce(lambda x, y :x+y, [1,2,3,4,5])  # 15

# 활용: 벡터 내적에 쓸 수 있다. 

```
- [복습] for문을 사용하지 않고도 짤 수 있는 것들 
1) iterator, generator
2) comprehension
3) 재귀
4) map, filter, reduce

* Enumerate 

```python
a=[1,2,3,4,5]
for i in a:
    print(i)
    
out:
1
2
3
4
5

----------------------------

a= '어쩌다 오늘'
for i in a:
    print(i)

out:
어
쩌
다
 
오
늘

-----------------------------
a= '어쩌다 오늘'
for i in enumerate(a):
    print(i) 

out:
(0, '어')
(1, '쩌')
(2, '다')
(3, ' ')
(4, '오')
(5, '늘')

----------------------------------

a= '어쩌다 오늘'
for i,j in enumerate(a):
    print(i,j) # unpacking 할 수 있다. 

out:
0 어
1 쩌
2 다
3  
4 오
5 늘

--------------------------

enumerate # iterate 인자로 받는다. 
for i,j in enumerate(a,1):
    print(i,j) 

out:
1 어
2 쩌
3 다
4  
5 오
6 늘
```

- 함수형 패러다임은 for, while을 바꿀 수 있다. 

---
**무단 배포 금지** 
{: .notice--danger}
