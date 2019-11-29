---
excerpt: "위도 경도 찾기"
header:
  overlay_image: /assets/images/2computers.jpg
  overlay_filter: 0.5 # same as adding an opacity of 0.5 to a black background
  caption: "Photo credit: [**Unsplash**]"
  actions:
    - label: "Reference"
      url: "https://docs.python.org/3/"
title: "[Python] 카카오api 기반 주소로 위도 경도 찾기"
date: 2019-11-28 00:00:00 -0400
categories: kakao_api
tags: kakao_api
gallery:
  - url: /assets/images/dummydata.JPG
    image_path: assets/images/dummydata.JPG
    alt: "placeholder image"   
---

# 카카오 API 위도, 경도 뽑기

카카오 API를 이용하여 주소를 input으로 위도 경도를 뽑아봅시다.  


- 예제 데이터 로드 

대한민국 도시 가운데 공시지가가 가장 높은 대지를 뽑아 데이터셋으로 만들었습니다. 
다음과 같은 데이터를 가지고 위도 경도를 뽑아보도록 하겠습니다. address는 'x'변수 
이름을 가지고 있는 DataFrame입니다.


```python
address
'''
x
0	서울특별시 명동2가 33-2대지
1	부산광역시 온천동 142-19광천지
2	대구광역시 동성로2가 162대지
3	인천광역시 부평동 212-69대지
4	광주광역시 충장로2가 15-1대지
5	대전광역시 봉명동 549-24광천지
6	울산광역시 성남동 249-5대지
7	경기도 안흥동 317-5잡종지
8	강원도 노학동 972-72광천지
9	충청북도 내수읍 초정리 63-2광천지
10	충청남도 온천동 221-19광천지
11	전라북도 고사동 72-6대지
12	전라남도 교동 275대지
13	경상북도 온정면 온정리 968-10광천지
14	경상남도 부곡면 거문리 213-25광천지
15	제주도 일도일동 1145-17대지
'''
```

- 카카오 API 연결하기 

[카카오 API](https://developers.kakao.com/) 이 사이트에 접속하여 '앱개발 시작하기'를 눌러 
APIkey를 받습니다. 카카오 API를 이용하여 위도, 경도를 찾아보도록 하겠습니다. 

```python
import requests
url = "https://dapi.kakao.com/v2/local/search/address.json?"
apikey = "자신의 APIkey를 넣으세요."
```

```python
x = [] # 경도를 저장합니다
y = [] # 위도를 저장합니다
c = [] # 위도 경도를 불러들인 주소를 저장합니다
for i in address['x']:
    query = i
    r = requests.get( url, params = {'query':query}, headers={'Authorization' : 'KakaoAK ' + apikey } )
    try: 
        xx=r.json()["documents"][0]['address']['x']
        x.append(xx)
    except:
        print('못불러들인 주소: '+ i) # 못불러들인 주소를 나타냅니다. 
    try:
        yy=r.json()["documents"][0]['address']['y']
        y.append(yy)
        cc=i
        c.append(cc)
    except:
        pass
'''
못불러들인 주소: 대전광역시 봉명동 549-24광천지
못불러들인 주소: 충청북도 내수읍 초정리 63-2광천지
못불러들인 주소: 충청남도 온천동 221-19광천지
못불러들인 주소: 경상북도 온정면 온정리 968-10광천지
못불러들인 주소: 경상남도 부곡면 거문리 213-25광천지
'''
```

```python
Address={'Longitude':x,'Latitude':y,'Address':c}
Address=pd.DataFrame(Address)
Address
'''
Longitude	Latitude	Address
0	126.98472897755484	37.56364462871111	서울특별시 명동2가 33-2대지
1	129.0813712491951	35.221331015489596	부산광역시 온천동 142-19광천지
2	128.59538022479325	35.86898584107002	대구광역시 동성로2가 162대지
3	126.72314687578324	37.49413510188036	인천광역시 부평동 212-69대지
4	126.91734757706894	35.14776730821751	광주광역시 충장로2가 15-1대지
5	129.32162000634725	35.554043433950476	울산광역시 성남동 249-5대지
6	127.45166238889941	37.27667464356648	경기도 안흥동 317-5잡종지
7	128.5741722395675	38.19448647687141	강원도 노학동 972-72광천지
8	127.14638214310412	35.81931305313351	전라북도 고사동 72-6대지
9	127.73341151866971	34.74139610567234	전라남도 교동 275대지
10	126.52736200678335	33.51291020592632	제주도 일도일동 1145-17대지
'''
```

이로서 주소를 기반으로 위도, 경도를 뽑아냈습니다. 

{% include gallery id="gallery1" caption="" %}





{% capture notice-2 %}

{% endcapture %}

<div class="notice">{{ notice-2 | markdownify }}</div>




---
**무단 배포 금지** 
{: .notice--danger}
