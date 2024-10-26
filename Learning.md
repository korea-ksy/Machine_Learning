# 🐌 학습(learning)이란?

=실제 정답과 예측 결과 사이의 오차(Loss,Cost,Error)를 줄여나가는 최적화 과정

# 🐌 Overfitting & Generalization(일반화)

Capacity의 극대화 -> Overfitting 발생 -> Generalization Error 증가 -> 새로운 데이터에 잘 대응 X

---

# 🐌 새로운 데이터들에 대해서도 좋은 결과를 내게 하려면?

## Cross validation(교차검증)

### 데이터를 3개의 그룹으로 나눈다.

1) 60%의 Training Data로 모델을 학습 시킨다.

2) 20%의 Validation data로 모델을 최적화/ 선택(Tune)한다.

3) 20%의 Test Data로 모델을 평가 한다.(Test only)

무조건 6:2:2 or 7:2??

100~n*10000 data -< 6:2:2 or 7:3

or 98:1:1 (test data가 많아서 1%도 test용으로 충분히 양이 많을 경우)

---

# 🐌 (startified) K-fold cross validation(후보 모델 간 비교 및 선택을 위한 알고리즘)

k: 연구자가 주는 값(보통 5나 10)

## 사용 이유

- 총 데이터 갯수가 적은 데이터 셋에 대하여 정확도를 향상시킬수 있음
- 이는 기존에 Training / Validation / Test 세 개의 집단으로 분류하는 것보다, Training과 Test로만 분류할 때 학습 데이터 셋이 더 많기 때문
- 데이터 수가 적은데 검증과 테스트에 데이터를 더 뺐기면 underfitting 등 성능이 미달되는 모델이 학습됨

## 과정

- 기존 과정과 같이 Training Set과 Test Set을 나눈다
- Training을 K개의 fold로 나눈다
- 한 개의 Fold에 있는 데이터를 다시 K개로 쪼갠다음, K-1개는 Training Data, 마지막 한개는 Validation Data set으로 지정한다
- 모델을 생성하고 예측을 진행하여, 이에 대한 에러값을 추출한다
- 다음 Fold에서는 Validation셋을 바꿔서 지정하고, 이전 Fold에서 Validatoin 역할을 했던 Set은 다시 Training set으로 활용한다
- 이를 K번 반복한다

## 과정(이어서)

- 각각의 Fold의 시도에서 기록된 Error를 바탕(에러들의 평균)으로 최적의 모델(조건)을 찾는다
- 해당 모델(조건)을 바탕으로 전체 Training set의 학습을 진행한다
- 해당 모델을 처음에 분할하였던 Test set을 활용하여 평가한다

## 단점

- 그냥 Training set / Test set 을 통해 진행하는 일반적인 학습법에 비해 시간 소요가 크다

---

[딥러닝에서 클래스 불균형을 다루는 방법](https://3months.tistory.com/414?category=756964)

## 🐌 왜 클래스 균형이 필요한가?

왜 데이터가 클래스 균형을 이루어야할까? 그리고 언제 클래스 균형이 필요할까? 핵심은 다음과 같다. **클래스 균형 클래스 균형은 소수의 클래스에 특별히 더 큰 관심이 있는 경우에 필요하다.**

### (1) Weight balancing

 **Weight balancing** 은 **training set** 의 각 데이터에서 **loss** 를 계산할 때 **특정 클래스의 데이터에 더 큰 loss 값을 갖도록 하는 방법**이다. 

 이를 구현하는 한 가지 간단한 방법은 원하는 클래스의 데이터의 loss 에는 특정 값을 곱하고, 이를 통해 딥러닝 모델을 트레이닝하는 것이다.

 예를 들어, "집을 사라" 클래스에는 75 %의 가중치를 두고, "집을 사지마라" 클래스에는 25 %의 가중치를 둘 수 있다. 이를 **python keras** 를 통해 구현하면 아래와 같다. class_weight 라는 dictionary 를 만들고, keas model 의 class_weight parameter 로 넣어주면 된다.

```python
import keras

class_weight = {"buy": 0.75,
                "don't buy": 0.25}

model.fit(X_train, Y_train, epochs=10, batch_size=32, class_weight=class_weight)
```

 물론 이 값을 예를 든 값이며, 분야와 최종 성능을 고려해 **가중치 비율**의 최적 세팅을 찾으면 된다. 

 다른 한 가지 방법은 **클래스의 비율에 따라 가중치를 두는 방법**인데, 
 예를 들어, 클래스의 비율이 1:9 라면 가중치를 9:1로 줌으로써 적은 샘플 수를 가진 클래스를 전체 loss 에 동일하게 기여하도록 할 수 있다.

 Weight balancing 에 사용할 수 있는 다른 방법은 **[Focal loss](https://arxiv.org/pdf/1708.02002.pdf)** 를 사용하는 것이다. 
 **Focal loss** 의 메인 아이디어는 다음과 같다. 
→ 다중 클래스 분류 문제에서, A, B, C 3개의 클래스가 존재한다고 하자.
  A 클래스는 상대적으로 분류하기 쉽고, B, C 클래스는 쉽다고 하자. 
 총 100번의 epoch 에서 단지 10번의 epoch 만에 validation set 에 대해 99 % 의 정확도를 얻었다. 
 그럼에도 불구하고 나머지 90 epoch 에 대해 A 클래스는 계속 loss 의 계산에 기여한다. 
 만약 상대적으로 분류하기 쉬운 A 클래스의 데이터 대신, B, C 클래스의 데이터에 더욱 집중을 해서 loss 를 계산을 하면 전체적인 정확도를 더 높일 수 있지 않을까? 
 예를 들어 batch size 가 64 라고 하면, 64 개의 sample 을 본 후, loss 를 계산해서 backpropagation 을 통해 weight 를 업데이트 하게 되는데 이 때, 이 loss 의 계산에 현재까지의 클래스 별 정확도를 고려한 weight 를 줌으로서 전반적인 모델의 정확도를 높이고자 하는 것이다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8945b734-be10-4194-b4ff-00185dff7fd1/Untitled.png)

**Focal loss** 는 어떤 batch 의 트레이닝 데이터에 같은 weight 를 주지 않고, **분류 성능이 높은 클래스에 대해서는 down-weighting** 을 한다. 이 때, **gamma** (위 그림) 를 주어, 이 down-weighting 의 정도를 결정한다. 
 이 방법은 **분류가 힘든 데이터에 대한 트레닝을 강조하는 효과**가 있다. 
 Focal loss 는 Keras 에서 아래와 같은 **custom loss function** 을 정의하고 loss parameter 에 넣어줌으로써 구현할 수 있다.

```python
import keras
from kerasimport backendas K
import tensorflowas tf

# Define our custom loss function
def focal_loss(y_true, y_pred):
    gamma = 2.0, alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

# Compile our model
adam = Adam(lr=0.0001)
model.compile(loss=[focal_loss], metrics=["accuracy"], optimizer=adam)
```

## (2) Over and under sampling

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/afe346cf-735e-400e-98d7-c4a515ecf722/Untitled.png)

 예를 들어, 위 그림에서 파란색 데이터가 주황색 데이터에비해 양이 현저히 적다. 
이 경우 두 가지 방법 - **Undersampling, Oversampling** 으로 샘플링을 할 수 있다.

- **Undersampling:** Majority class (파란색 데이터) 의 일부만을 선택하고, 
Minority class (주황색 데이터) 는 최대한 많은 데이터를 사용하는 방법이다. 
이 때 Undersampling 된 파란색 데이터가 원본 데이터와 비교해 대표성이 있어야 한다.
- **Oversampling: Minority class** 의 복사본을 만들어, Majority class 의 수만큼 데이터를 만들어주는 것이다. 똑같은 데이터를 그대로 복사하는 것이기 때문에 새로운 데이터는 기존 데이터와 같은 성질을 갖게된다.
