## Cost Function & MSE

**Cost Function (비용함수)**

: 예측 값과 실제 값의 차이를 기반으로 모델의 성능(정확도)을 판단하기 위한 함수

Linear regression의 경우 MEan squared error function(평균 제곱 오차 함수)을 활용

→ MSE(cost)가 최소가 되도록 하는 θ(parameter,a&b)를 찾아야 한다. (y=ax+b)

### MSE

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5835817a-f6c5-404c-9724-61af1b7218c3/Untitled.png)

*그외

- MAE
- MAPE
- RMSE

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7be1b827-6609-46ab-857e-b1cdfba72db3/Untitled.png)

## Gradient Descent Algorithm(경사하강법)

- Cost Function의 값을 최소로 만드는 θ를 찾아나가는 방법
- Cost Function의 Gradient에 상수를 곱한 값을 빼서 θ를 조정

→ Cost Function 에 경사가 아래로 되어있는 방향으로 내려가서 Cost가 최소가 되는 지점을 찾는다. 어느 방향으로 θ를 움직이면 Cost값이 작아지는지 **현재 θ위치에서 Cost함수를 미분하여 판단**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/33730f92-5f3b-4d5e-9ca6-2ae11156596d/Untitled.png)

MSE를 미분하여 얻은 기울기의 반대(음수면 양수로)로 θ값을 이동시킨다.

→ 이 과정을 반복하여 접선의 기울기가 0이 되는 지점의 **θ값**을 찾는다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/07b0f66d-d8ba-47bb-afaa-bf0924396500/Untitled.png)

Jθ : Cost Function
