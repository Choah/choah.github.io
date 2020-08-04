---
excerpt: "Level1 Python3 Algorithm"
header:
  overlay_image: /assets/images/2computers.jpg
  overlay_filter: 0.5 # same as adding an opacity of 0.5 to a black background
  caption: "Photo credit: [**Unsplash**]"
  actions:
    - label: "Reference"
      url: "https://programmers.co.kr/learn/challenges"
title: "[프로그래머스] Python3 알고리즘"
date: 2020-08-04 00:00:00 -0400
categories: Python
tags: Algorithm
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



# 인형뽑기

https://programmers.co.kr/learn/courses/30/lessons/64061

```python
import numpy as np

def solution(board, moves):
    answer = 0
    a = np.array(board).T.tolist()
    
    # 뽑은 값을 저장해 놓을 리스트
    lst = list()
    
    for i in moves:
        n=0
        try:
            while n ==0:
                n = a[i-1].pop(0)
            lst.append(n)
            if (len(lst)>1) & (n == lst[-2]):
                answer += 1
                del lst[-1]
                del lst[-1]
        except:
            pass
        
    # 사라진 인형들 개수
    return answer*2
```




# 모의고사 

https://programmers.co.kr/learn/courses/30/lessons/42840

```python
def solution(answers):
    p = [[1, 2, 3, 4, 5],
         [2, 1, 2, 3, 2, 4, 2, 5],
         [3, 3, 1, 1, 2, 2, 4, 4, 5, 5]]
    s = [0] * len(p)

    for q, a in enumerate(answers):
        for i, v in enumerate(p):
            if a == v[q % len(v)]:
                s[i] += 1
    return [i + 1 for i, v in enumerate(s) if v == max(s)] 
```


# 두 정수 사이의 합 

https://programmers.co.kr/learn/courses/30/lessons/12912

```python
def solution(a, b):
    return sum(range(min(a,b), max(a,b) + 1))
'''



# 2016년 

https://programmers.co.kr/learn/courses/30/lessons/12901

```python
from datetime import datetime

def solution(a, b):
    week = ['MON','TUE','WED','THU','FRI','SAT','SUN']
    day = datetime(year = 2016, month = a, day =b)
    return week[day.weekday()]
'''


# 같은 숫자는 싫어 

https://programmers.co.kr/learn/courses/30/lessons/12906

```python
def solution(arr):
    answer = [arr[0]]
    
    for i in range(1,len(arr)):
        if arr[i] != arr[i-1]:
            answer.append(arr[i])
    
    return answer
'''

```python
def no_continuous(s):
    a = []
    for i in s:
        if a[-1:] == [i]: continue
        a.append(i)
    return a
```

# 문자열 다루기 기본 

https://programmers.co.kr/learn/courses/30/lessons/12918

```python
def solution(s):
    if len(s) not in (4,6):
        return False
    
    for ch in s:
        if ord(ch) not in range(48, 58):
            return False

    return True
```

# 문자열 내 맘대로 정렬하기

https://programmers.co.kr/learn/courses/30/lessons/12915

```python
def solution(strings, n):
    return sorted(strings, key = lambda x : (x[n], x))
```


# 나누어 떨어지는 숫자 배열

https://programmers.co.kr/learn/courses/30/lessons/12910

```python
def solution(arr, divisor):
    return sorted([i for i in arr if i % divisor == 0]) or [-1]
```



# 문자열 내 p와 y의 개수

https://programmers.co.kr/learn/courses/30/lessons/12916

```python
def solution(s):
    if s.lower().count('p') == s.lower().count('y'):
        return True
    else:
        return False
```


# 문자열 내림차순으로 배치하기

https://programmers.co.kr/learn/courses/30/lessons/12917

```python
def solution(s):
    return ''.join(sorted(s, reverse=True))
```



# 문자열 다루기 기본

https://programmers.co.kr/learn/courses/30/lessons/12918

{% capture notice-2 %}
#### [정규식]
- \d: 숫자
- \D: 숫자가 아닌 것
- \s: whitespace (\t,\n,\r,\f,\v)
- \S: whitespace가 아닌 것
- \w: 문자 + 숫자와 매치
- \W: 문자 + 숫자가 아닌 것 
{% endcapture %}

```python
def solution(s):
    if len(s) in (4, 6):
        try: 
            int(s)
            return True
        except:
            return False
    else:
        return False
```


# 서울에서 김서방 찾기

https://programmers.co.kr/learn/courses/30/lessons/12919

```python
def solution(seoul):
    for i,j in enumerate(seoul):
        if j == 'Kim':
            return "김서방은 {}에 있다".format(i)
```

```python
def findKim(seoul):
    return "김서방은 {}에 있다".format(seoul.index('Kim'))
```


# 소수찾기

https://programmers.co.kr/learn/courses/30/lessons/12921

- 에라토스테네스의 체

```python
def solution(n):
    a = [False, False] + [True]*(n-1)
    answer = []
    for i in range(2,n+1):
        if a[i]:
            answer.append(i)
            for j in range(2*i, n+1, i):
                a[j] = False
    return len(answer)
```

```python
def solution(n):
    num=set(range(2,n+1))

    for i in range(2,n+1):
        if i in num:
            num-=set(range(2*i,n+1,i))
    return len(num)
```


# 수박수박수

https://programmers.co.kr/learn/courses/30/lessons/12922

```python
def solution(n):
    b = '수박'*n
    answer = ''
    for i in range(n):
        answer = answer + b[i]
    return answer
```

```python
def water_melon(n):
    s = "수박" * n
    return s[:n]
```


# 시저암호

https://programmers.co.kr/learn/courses/30/lessons/12926

```python
def solution(s,n):
    abc = 'abcdefghijklmnopqrstuvwxyz'
    ABC = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    answer = ''
    for i in s:
        if i in abc:
            index = abc.find(i)
            index += n
            answer = answer + abc[index%26]
        elif i in ABC:
            index = ABC.find(i)
            index += n
            answer = answer + ABC[index%26]
        else:
            answer = answer + ' '
    return answer
```

```python
def caesar(s, n):
    s = list(s)
    for i in range(len(s)):
        if s[i].isupper():
            s[i]=chr((ord(s[i])-ord('A')+ n)%26+ord('A'))
        elif s[i].islower():
            s[i]=chr((ord(s[i])-ord('a')+ n)%26+ord('a'))

    return "".join(s)
```


# 약수의 합

https://programmers.co.kr/learn/courses/30/lessons/12928

```python
def solution(n):
    answer = 0
    for i in range(1,n+1):
        if n % i == 0:
            answer += i
    return answer
```

```python
def sumDivisor(num):
    # num / 2 의 수들만 검사하면 성능 약 2배 향상된다.
    return num + sum([i for i in range(1, (num // 2) + 1) if num % i == 0])
```


{% include gallery id="gallery" caption="flowers" %}



{% capture notice-2 %}

{% endcapture %}

<div class="notice">{{ notice-2 | markdownify }}</div>



