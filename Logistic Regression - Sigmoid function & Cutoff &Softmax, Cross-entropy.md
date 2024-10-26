# 🐌 Logstic Regression

- 이진 분류(binary classification) 문제를 해결하기 위한 모델
- 변형된 모델로 다항 로지스틱 회귀(k-class)와 서수 로지스틱 회귀(k-class & ordinal)도 존재

ex) 스팸 메일 분류, 질병 양성/음성 분류 …

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4e47aff9-5aaf-4160-abc5-4928e78ba81b/Untitled.png)

- Sigmoid function 을 이용하여 기본적으로 특정 Input data가 양성 class에 속할 확률을 계산
- Sigmoid function 의 정확한 y 값을 받아 양성일 확률을 예측하거나, 특정 **cutoff**를 넘는 경우 양성, 그 외에는 음성으로 예측할 수 있음

## 🐌 Sigmoid function & Cross-entropy function

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/01656d71-9dfe-4dbb-9514-cddd50b97a8a/Untitled.png)

: 성능 지표로 MSE가 아니라 분류를 위한 Cost function인 **Cross entropy**를 활용

예측값의 분포와 실제값의 분포를 비교하여 그 차이를 Cost로!

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/98d04960-499f-4457-9a5e-6abe4905ab7e/Untitled.png)

**❗Logistic Regression에서 MSE를 안 쓰는 이유**

Linear regression에서는 MSE를 적용하였을 때 Gradient descent로 값을 찾는데 용이한 이차 함수 그래프가 그려지지만, Logisitc regression에서는 MSE를 쓰면 위의 그림과 같이 함정들이 많이 생겨 경사 하강법으로는 값을 찾기 어려움

## **🐌** Logistic regression 과정

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/68b9b586-c0d7-40e6-83e1-0c4848329a2a/Untitled.png)

1. 입력층에 넣을 행을 1열 n행으로 회전시켜 신경망 입력층에 넣는다.
2. **softmax 함수**를 이용하여 각 행의 확률을 계산한다.
3. class(분류)를 **one-hot encoding**으로 **one-hot label**을 만들어준다.
4. **softmax함수**로 구해논 각 행의 **확률 벡터**와, 만들어논 **one-hot label**을 **Cross-entropy** 계산을 적용하여 **결과 값이 낮은 모델**을 찾는다.

## 🐌 Softmax 함수

: 다중 클래스 분류(multi-class/ Multinomial classification) 문제를 위한 함수

ex) 강아지 품종 분류, 필기체 숫자 인식

- Model의 output에 해당하는 logit(score)을 각 클래스에 소속될 확률에 해당하는 값들의 벡터로 변환해준다.
- Logistic regression를 변형/ 발전시킨 방법으로, binary class 문제를 벗어나 multiple class 문제로 일반화시켜 적용할 수 있다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3b67976f-ab2f-4c7b-87bf-c3a3a64fad29/Untitled.png)
