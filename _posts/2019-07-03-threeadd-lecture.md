---
excerpt: "(1) 함수형 프로그래밍 특징 (2) closure (3) decorator (4) pbd.set_trace() (5) encapsulation "
header:
  overlay_image: /assets/images/2computers.jpg
  overlay_filter: 0.5 # same as adding an opacity of 0.5 to a black background
  caption: "Photo credit: [**Unsplash**]"
  actions:
    - label: "Reference"
      url: "https://docs.python.org/ko/3/index.html "
title: "[Python] Python 고급 강의3: 함수형 패러다임 고급 기법 (add)"
date: 2019-07-03 00:00:00 -0400
categories: python jupyter
tags: lecture CS encapsulation comprehension recursion eliminting_loops callables lambda closure decorator
---


**본 과정은 제가 파이썬 수업을 들으면서 배운 내용을 복습하는 과정에서 적어본 것입니다.<br> 틀린 부분이 있다면 댓글에 남겨주시면 고치도록 하겠습니다.<br> 확실하지 않은 내용은 '(?)'을 함께 적었으니 그 내용을 아신다면 댓글에 남겨주시면 감사하겠습니다.**
{: .notice--warning}

## 1) 함수형 프로그래밍 특징 
- 값을 파이썬에서는 객체 
- 값이면 할당할 때 오른쪽 식에 쓸 수 있다.

• Functions are first class (objects). -> 값처럼 쓸 수 있다. 
• Recursion is used as a primary control structure. In some languages, no other “loop” construct exists.
 - python은 recursion(재귀) 쓸 수는 있는데 속도가 느려서 안쓴다. 
• There is a focus on list processing (for example, it is the source of the name Lisp). Lists are often used with recursion on sublists as a substitute for loops.

* mutable 쓰면 쉽다. 
* 자연어처리에도 많이쓰고 빅데이터에 많이 쓰인다. 
* for문을 줄일려고 한다. (순회가 많으므로 읽기가 어렵다.) 
* 오컴의 면도날 : 단순한게 좋다. 

```python
a= print # 얇은 초록색: 함수, class / 두꺼운 초록색: keyword 
a('경아') # 경아
# 함수를 값으로 (class)로 만들었다. 

a.__name__ # 'print' -> 실제 이름은 print이지만 a로 이름만 바뀜 


sum = 0 # sum이 사라진다. 
del sum # 원래 함수로 돌아간다. 

sum (range(1,101)) # 5050

sum = 0 # 어리석은 방법
for i in range(1,101):
    sum+=i # sum 함수를 덮어버리게 되면 기본 sum 기능을 못쓴다. <- 안좋은 것 
sum # 5050

del sum    


a = [str, int, float]

a[0](2) # '2' 함수니까 값처럼 만들 수는 있지만 이렇게 사용하진 않음


def a(x):
    print(x())

a(print) # 'None': 이것도 된다. 하지만 이렇게 하면 안된다. <- anti pattern 

```

## 2) closure
- closure: 함수 안에 함수를 집어넣어서 하는 것 : 중첩된 함수 (Nested function)
- 자습서에도 있다. https://docs.python.org/ko/3/tutorial/index.html
 
+ decorator: 함수를 중첩시키는 것 

```python
x=1 
def y():
    print(x)

y() # 1


x=1 
def y():
    x=2
    print(x)

y() # 2 -> 찾는 순서 : L(logcal) E(Enclosed) G (global -> 함수 밖 영역) B(built-in -> 있는지 없는지 찾고 )


x=1 
def y():
    x=x+1
    print(x)

y()  # unboundlocalerror : 안에 있는 애 접근은 할 수 있지만 변경은 못시킴 
```

```python
x=1 
x=1 
def y():
    global x # 글로벌 붙이면 싱크가된다. (?) 안에 있는 로컬과 밖에 있는 글로벌이 서로 싱크가된다. 
    x=x+1
    print(x)

y() # control+shift+'-' 하면 나눠진다. 

y() # Shift + m  

x # 안좋다. -> x=3으로 바뀌기 때문에

out:
2
3
3
```

```python
# closure
def y(x):
    def z(n):
        return x+n # parameter 인자 2개 
    return z

two_add=y(2)
two_add(3)

out:
5

---------------------

def y(n):     
    return lambda n: n+x

a=y(3)
a(2)

out:
3
```


## 3) decorator 
- 함수 or 클래스를 인자로 받아서 기능을 바꿔주는 애
- ex) telegram 
- @ 써서 다른 기능을 추가 가능

* 함수형 패러다임은 for문 쓰는거 안좋아한다.

+ descripter: private 화 


## 4) pbd.set_trace() 
- 파이썬 디버깅 패키지 


```python
def y():
    import pdb; pdb.set_trace() # h : help, c: countinue, p x : 현재 p의 기법이 뭔가 알려주는 것 , l: 어디가 문제인지 알려주는 것 
    x=x+1
    print(x)

y() # x+1이 안된다.

----------------------

x=1
def y():
    breakpoint() # 잠시 멈춰라라는 명령어 -> p: print , p x: print x 값 , l : 현재 소스 코드, q: quit
    x= x+1
    print(x)
    
y() # x+1이 안된다.

------------------------

x=1
def y():
    global x # 싱크 해주는 것 하지만 수정, 변경은 못한다. -> 단 글로벌을 붙이면 바꿀수 있다. 
    x= x+1
    print(x)
    
y() # 2

----------------------------

x=1
def y(): 
    x= 3 # 함수 안에서 쓰는 임시 변수 # 안에서 밖 접근할 수 있다. 하지만 변경 못함, but 글로벌 붙이면 변경가능 
    print(x)
    
y() # 3 
x # 1

----------------------------

x=1
def y(): 
    x=x+3 # 오른쪽 식에 있으면 함수 밖에 있는 값 -> 밖에서는 안에 접근할 수 없다. : encapsulation 
    print(x)
    
y() # UnboundLocalError 에러가 뜬다.
x 
```

```python
def y():
    z=1
    print(x)
    
y.z = 3
    
y() # 1
y.z  # 3
```
## 5) encapsulation
- 밖에서는 안에 접근할 수 없다. : encapsulation 

```python
class A: 
    x=1 # function은 바꿀 수 없다. 근데 2가 나온다. -> class는 밖에서 안에서 바꿀 수 있다. 
A.x=2
a=A()
a.x # 2
```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```
```python

```
```python

```
```python

```

```python

```

```python

```


---
**무단 배포 금지** 
{: .notice--danger}

