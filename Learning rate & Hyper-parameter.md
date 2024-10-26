# 🐌 Gradient Descent Algorithm (경사 하강법)

- Gradient: 모든 변수의 편미분을 벡터로 정리한 것(= 함수의 기울기, 경사)
- 편미분: 변수가 2개 이상인 함수를 미분할 때 미분 대상 변수 외에 나머지 변수를 상수처럼 고정시켜 미분하는 것
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ed8376dd-9fa7-4c49-bc9d-0a18298ab5cf/Untitled.png)
    

1) 변수(θ)의 초기값을 설정

2) 현재 변수 값에 대응되는 Cost function의 경사도 계산(미분)

3) 변수를 경사 방향(기울기의 음의 방향 = Gradient 의 음의 방향)으로 움직여 다음 변수 값으로 설정

4) 1~3 을 반복하며 Cost function이 최소가 되;도록 하는 변수 값으로 근접해 나간다.

(=전체 Cost 값이 변하지 않거나 매우 느리게 변할 때 까지 접근)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/348df093-b351-44c8-bc58-52d5e13b0e5d/Untitled.png)

**학습률 테스트**

      ⇒ https://developers.google.com/machine-learning/crash-course/fitter/graph?hl=ko

**Hyper-parameter** : 초매개변수 ( 사람이 설정하는 모든 것)

**AutoML**

Auto F.E

Auto M.S

Auto HPO (hyper-parameter tuning / model tuning)
