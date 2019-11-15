---
excerpt: "PIL로 달력 만들어 보기"
header:
  overlay_image: /assets/images/2computers.jpg
  overlay_filter: 0.5 # same as adding an opacity of 0.5 to a black background
  caption: "Photo credit: [**Unsplash**]"
title: "[Python] 달력만들기"
date: 2019-11-15 00:00:00 -0400
categories: python Image
tags: lecture python Image
gallery1:
  - url: /assets/images/calendar.JPG
    image_path: assets/images/calendar.JPG
    alt: "placeholder image"   
---


## 파이썬으로 달력만들기

### PIL로 달력을 만들어 봅시다


```python
from PIL import ImageDraw, ImageFont
calendar = Image.new('RGB',(300,500), color = 'brown') # concpet-modes 

a = Image.open('bono.jpg')
a = a.crop((2,42,160,150))
a = a.resize((280,200))

# 달력 바탕이미지에 원하는 이미지를 원하는 위치에 붙인다.
calendar.paste(a,(10,10)) 

# ImageDraw는 클래스이기 때문에 인스턴스화해서 써야한다. 
a = ImageDraw.ImageDraw(calendar)

# 달력 내모 칸을 만든다.
a.rectangle(((10,220),(290,490)))
for i in range(1,8):
    a.line(((10+290/7*i,220),(10+290/7*i,490)), width =1)
    a.line(((10,220+270/6*i),(290,220+270/6*i)), width =1)

# Font를 지정해서 한국어를 쓸 수 있도록 도와준다. 
f = ImageFont.truetype('malgun.ttf', size=14, encoding='utf-8')

# 달력에 요일 이름 넣기
week = (_ for _ in ['일','월','화','수','목','금','토'])
a.text((20,20),'11월', font=f)
for i in range(0,7):
    a.text(((20+290/7*i,225)),next(week), font=f)

# 달력에 11월달 숫자 넣기
day = (_ for _ in range(3,31))
for i in range(2,6):
    for j in range(7):
        xy = ((15+j*(290/7),222+i*(270/6)))
        try:
            a.text((xy),str(next(day)))    
        except:
            pass
        else:
            a.text((15+5*(290/7),222+1*(270/6)),'1')
            a.text((15+6*(290/7),222+1*(270/6)),'2')

# Calendar 보기
calendar
```
{% include gallery id="gallery1" caption="calendar" %}



---
**무단 배포 금지** 
{: .notice--danger}
