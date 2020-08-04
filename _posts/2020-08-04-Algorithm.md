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
'''

# 문자열 내 맘대로 정렬하기

https://programmers.co.kr/learn/courses/30/lessons/12915

```python
def solution(strings, n):
    return sorted(strings, key = lambda x : (x[n], x))
'''


# 나누어 떨어지는 숫자 배열

https://programmers.co.kr/learn/courses/30/lessons/12910

```python
def solution(arr, divisor):
    return sorted([i for i in arr if i % divisor == 0]) or [-1]
'''



# 문자열 내 맘대로 정렬하기

https://programmers.co.kr/learn/courses/30/lessons/12915

```python
def solution(strings, n):
    return sorted(strings, key = lambda x : (x[n], x))
'''


{% include gallery id="gallery" caption="flowers" %}



{% capture notice-2 %}

{% endcapture %}

<div class="notice">{{ notice-2 | markdownify }}</div>



