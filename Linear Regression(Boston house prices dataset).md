# 🐌 Linear Regression 
-Scikit-learn & One-hot encoding

## scikit-learn

:python으로 Traditional Machine Learning 알고리즘들을 구현한 오픈 소스 라이브러리

## scikit-learn의 장점

- 파이썬의 다른 라이브러리들과의 호환성이 좋음(Numpy, pandas, Matplotlib 등)
- 전체에 걸쳐 통일된 인터페이스를 가지고 있기 때문에 매우 간단하게 여러 알고리즘들을 적용할 수 있음

### 1. 데이터셋 불러오기

2. Train/ Test set으로 데이터 나누기

3. 모델 객체(Model Instance)생성하기

4. 모델 학습 시키기(Model fitting)

5. 모델로 새로운 데이터 예측하기 (predict on test data)

### 1-1. (미국 보스턴의 주택 가격) 데이터 읽어들이기

### 1) Features

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

### df_data (Data, x)

- 0 : **범죄율**
- 1 : **25,000 평방피트를 초과하는 거주지역 비율**
- 2 : **비소매상업지역 면적 비율**
- 3 : **찰스강의 경계에 위치한 경우는 1, 아니면 0**
- 4 : **일산화질소 농도**
- 5 : **주택당 방 수 (거실 외 subroom)**
- 6 : **1940년 이전에 건축된 주택의 비율**
- 7 : **직업센터의 거리**
- 8 : **방사형 고속도로까지의 거리**
- 9 : **재산세율**
- 10 : **학생/교사 비율**
- 11 : **인구 중 흑인 비율**
- 12 : **인구 중 하위 계층 비율**

```python
f_data = pd.read_excel('/content/drive/MyDrive/카카오 sw ai 중급 2(part5)/실습 파일/2. Scikit-learn (Answer)/boston_house_data.xlsx',index_col=0) # 엑셀 파일 읽기
df_data.head() # 윗부분만 보려면?

#Column == Attribute == Dimension == Feature(only X_data)
```

|  | **0** | **1** | **2** | **3** | **4** | **5** | **6** | **7** | **8** | **9** | **10** | **11** | **12** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **0** | 0.00632 | 18.0 | 2.31 | 0 | 0.538 | 6.575 | 65.2 | 4.0900 | 1 | 296 | 15.3 | 396.90 | 4.98 |
| **1** | 0.02731 | 0.0 | 7.07 | 0 | 0.469 | 6.421 | 78.9 | 4.9671 | 2 | 242 | 17.8 | 396.90 | 9.14 |
| **2** | 0.02729 | 0.0 | 7.07 | 0 | 0.469 | 7.185 | 61.1 | 4.9671 | 2 | 242 | 17.8 | 392.83 | 4.03 |
| **3** | 0.03237 | 0.0 | 2.18 | 0 | 0.458 | 6.998 | 45.8 | 6.0622 | 3 | 222 | 18.7 | 394.63 | 2.94 |
| **4** | 0.06905 | 0.0 | 2.18 | 0 | 0.458 | 7.147 | 54.2 | 6.0622 | 3 | 222 | 18.7 | 396.90 | 5.33 |

# 🐌Feature Normalization (Scailing)

### Numercial Column(Variable)

- > Min-max algorithm or Standardization

### Categorical Column(Variable)

- > One-hot encoding

```python
df_data[8].value_counts(sort=0).index # .Keys()
# from collections import Counter
# Counter(df_data[3])
```

### **2) Target**

```python
df_target = pd.read_excel('/content/drive/MyDrive/카카오 sw ai 중급 2(part5)/실습 파일/2. Scikit-learn (Answer)/boston_house_target.xlsx', index_col=0)
df_target.head()
```

|  | **0** |
| --- | --- |
| **0** | 24.0 |
| **1** | 21.6 |
| **2** | 34.7 |
| **3** | 33.4 |
| **4** | 36.2 |

### df_target (Target, y)

- Town 내 주택 가격의 중앙값 (단위 : $1,000)

### **3) Features & Target 합쳐서 살펴보기**

```python
#A.join(B)
#pd.merge(A,B,left_on="A기준여르 right_on='B기준열", how='inner')
#pd.concat([A,B],axis=1)

df_main = pd.concat([df_data, df_target], axis=1) # concatenate
df_main.head()
```

|  | **0** | **1** | **2** | **3** | **4** | **5** | **6** | **7** | **8** | **9** | **10** | **11** | **12** | **0** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **0** | 0.00632 | 18.0 | 2.31 | 0 | 0.538 | 6.575 | 65.2 | 4.0900 | 1 | 296 | 15.3 | 396.90 | 4.98 | 24.0 |
| **1** | 0.02731 | 0.0 | 7.07 | 0 | 0.469 | 6.421 | 78.9 | 4.9671 | 2 | 242 | 17.8 | 396.90 | 9.14 | 21.6 |
| **2** | 0.02729 | 0.0 | 7.07 | 0 | 0.469 | 7.185 | 61.1 | 4.9671 | 2 | 242 | 17.8 | 392.83 | 4.03 | 34.7 |
| **3** | 0.03237 | 0.0 | 2.18 | 0 | 0.458 | 6.998 | 45.8 | 6.0622 | 3 | 222 | 18.7 | 394.63 | 2.94 | 33.4 |
| **4** | 0.06905 | 0.0 | 2.18 | 0 | 0.458 | 7.147 | 54.2 | 6.0622 | 3 | 222 | 18.7 | 396.90 | 5.33 | 36.2 |

```python
#열 이름 통째로 바꾸기
df_main.columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV'] 
df_main.head()
```

|  | **CRIM** | **ZN** | **INDUS** | **CHAS** | **NOX** | **RM** | **AGE** | **DIS** | **RAD** | **TAX** | **PTRATIO** | **B** | **LSTAT** | **MEDV** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **0** | 0.00632 | 18.0 | 2.31 | 0 | 0.538 | 6.575 | 65.2 | 4.0900 | 1 | 296 | 15.3 | 396.90 | 4.98 | 24.0 |
| **1** | 0.02731 | 0.0 | 7.07 | 0 | 0.469 | 6.421 | 78.9 | 4.9671 | 2 | 242 | 17.8 | 396.90 | 9.14 | 21.6 |
| **2** | 0.02729 | 0.0 | 7.07 | 0 | 0.469 | 7.185 | 61.1 | 4.9671 | 2 | 242 | 17.8 | 392.83 | 4.03 | 34.7 |
| **3** | 0.03237 | 0.0 | 2.18 | 0 | 0.458 | 6.998 | 45.8 | 6.0622 | 3 | 222 | 18.7 | 394.63 | 2.94 | 33.4 |
| **4** | 0.06905 | 0.0 | 2.18 | 0 | 0.458 | 7.147 | 54.2 | 6.0622 | 3 | 222 | 18.7 | 396.90 | 5.33 | 36.2 |

```python
df_main.describe() # description
```

|  | **CRIM** | **ZN** | **INDUS** | **CHAS** | **NOX** | **RM** | **AGE** | **DIS** | **RAD** | **TAX** | **PTRATIO** | **B** | **LSTAT** | **MEDV** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **count** | 506.000000 | 506.000000 | 506.000000 | 506.000000 | 506.000000 | 506.000000 | 506.000000 | 506.000000 | 506.000000 | 506.000000 | 506.000000 | 506.000000 | 506.000000 | 506.000000 |
| **mean** | 3.593761 | 11.363636 | 11.136779 | 0.069170 | 0.554695 | 6.284634 | 68.574901 | 3.795043 | 9.549407 | 408.237154 | 18.455534 | 356.674032 | 12.653063 | 22.532806 |
| **std** | 8.596783 | 23.322453 | 6.860353 | 0.253994 | 0.115878 | 0.702617 | 28.148861 | 2.105710 | 8.707259 | 168.537116 | 2.164946 | 91.294864 | 7.141062 | 9.197104 |
| **min** | 0.006320 | 0.000000 | 0.460000 | 0.000000 | 0.385000 | 3.561000 | 2.900000 | 1.129600 | 1.000000 | 187.000000 | 12.600000 | 0.320000 | 1.730000 | 5.000000 |
| **25%** | 0.082045 | 0.000000 | 5.190000 | 0.000000 | 0.449000 | 5.885500 | 45.025000 | 2.100175 | 4.000000 | 279.000000 | 17.400000 | 375.377500 | 6.950000 | 17.025000 |
| **50%** | 0.256510 | 0.000000 | 9.690000 | 0.000000 | 0.538000 | 6.208500 | 77.500000 | 3.207450 | 5.000000 | 330.000000 | 19.050000 | 391.440000 | 11.360000 | 21.200000 |
| **75%** | 3.647423 | 12.500000 | 18.100000 | 0.000000 | 0.624000 | 6.623500 | 94.075000 | 5.188425 | 24.000000 | 666.000000 | 20.200000 | 396.225000 | 16.955000 | 25.000000 |
| **max** | 88.976200 | 100.000000 | 27.740000 | 1.000000 | 0.871000 | 8.780000 | 100.000000 | 12.126500 | 24.000000 | 711.000000 | 22.000000 | 396.900000 | 37.970000 | 50.000000 |

### **1-2. Dataframe 을 Numpy array (배열, 행렬)로 바꿔주기**

```python
boston_data = np.array(df_data)
boston_target = np.array(df_target)

pd.DataFrame(boston_data)
```

| **0** | **1** | **2** | **3** | **4** | **5** | **6** | **7** | **8** | **9** | **10** | **11** | **12** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **0** | 0.00632 | 18.0 | 2.31 | 0.0 | 0.538 | 6.575 | 65.2 | 4.0900 | 1.0 | 296.0 | 15.3 | 396.90 |
| **1** | 0.02731 | 0.0 | 7.07 | 0.0 | 0.469 | 6.421 | 78.9 | 4.9671 | 2.0 | 242.0 | 17.8 | 396.90 |
| **2** | 0.02729 | 0.0 | 7.07 | 0.0 | 0.469 | 7.185 | 61.1 | 4.9671 | 2.0 | 242.0 | 17.8 | 392.83 |
| **3** | 0.03237 | 0.0 | 2.18 | 0.0 | 0.458 | 6.998 | 45.8 | 6.0622 | 3.0 | 222.0 | 18.7 | 394.63 |
| **4** | 0.06905 | 0.0 | 2.18 | 0.0 | 0.458 | 7.147 | 54.2 | 6.0622 | 3.0 | 222.0 | 18.7 | 396.90 |
| **...** | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| **501** | 0.06263 | 0.0 | 11.93 | 0.0 | 0.573 | 6.593 | 69.1 | 2.4786 | 1.0 | 273.0 | 21.0 | 391.99 |
| **502** | 0.04527 | 0.0 | 11.93 | 0.0 | 0.573 | 6.120 | 76.7 | 2.2875 | 1.0 | 273.0 | 21.0 | 396.90 |
| **503** | 0.06076 | 0.0 | 11.93 | 0.0 | 0.573 | 6.976 | 91.0 | 2.1675 | 1.0 | 273.0 | 21.0 | 396.90 |
| **504** | 0.10959 | 0.0 | 11.93 | 0.0 | 0.573 | 6.794 | 89.3 | 2.3889 | 1.0 | 273.0 | 21.0 | 393.45 |
| **505** | 0.04741 | 0.0 | 11.93 | 0.0 | 0.573 | 6.030 | 80.8 | 2.5050 | 1.0 | 273.0 | 21.0 | 396.90 |

506 rows × 13 columns

```python
type(boston_data) #타입
boston_data.shape #차원 수 확인

>>> (506, 13)
```

### **2. Feature 선택하기**

```python
# Use only one feature 

# 항상 행렬 형태로 뽑아서 모델에게 던져줘야 합니다
boston_X = boston_data[:, 12:13] # 인구 중 하위 계층 비율 
boston_X
# boston_X = boston_data[:, 12]
# boston_X.reshape(-1,1) <- 1열로 맞춰주고 행의 수는 알아서

boston_Y = boston_target
boston_Y
```

### **3. Training & Test set 으로 나눠주기**

```python
from sklearn import model_selection

x_train, x_test, y_train, y_test = model_selection.train_test_split(boston_X, boston_Y, test_size=0.3, random_state=0)
# random_state (random_seed or seed) : make the result reproducible
# random_state 값에 따라 랜덤한 값이 달라짐(같은 random_state값끼리는 값이 같음)
```

```python
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

>>>(354, 1)
	 (152, 1)
	 (354, 1)
	 (152, 1)
```

### **4. 비어있는 모델 객체 만들기**

```python
from sklearn import linear_model

model = linear_model.LinearRegression() # 선형회귀
```

### **5. 모델 객체 학습시키기 (on training data)**

```python
# Train the model using the training sets

model.fit(x_train, y_train) # 모델에 데이터를 '맞춰줍니다'
```

```python
print('Coefficients: ', model.coef_) # coefficient : 계수 -> a == -0.968

>>> Coefficients:  [[-0.96814078]]
```

```python
print('Intetcepts: ', model.intercept_) # intetcept: 교차점(y절편) -> b == 34.789

>>> Intetcepts:  [34.78978059]

# y = -0.968 *x + 34.789
```

### **6. 학습이 끝난 모델 테스트하기 (on test data)**

```python
model.predict(x_train) # '예측하다'
```

```python
# 354개 Train 데이터에 대한 Model 의 Mean squared error 
print('MSE(Training data) : ', np.mean((model.predict(x_train) - y_train) ** 2))
```

```
# Use this!
from sklearn.metrics import mean_squared_error

print('MSE(Training data) : ', mean_squared_error(model.predict(x_train), y_train))
```

```python
#152개 Test 데이터에 대한 Model 의 Mean squared error 
print('MSE(Test data) : ', mean_squared_error(model.predict(x_test), y_test))

>>> MSE(Test data) :  39.81715050474416
```

```python
#RMSE (Root mean squared error)

# Square root(제곱근) of error
np.sqrt(mean_squared_error(model.predict(x_test), y_test))

>>> 6.310083240714354
```

### **7. 모델 시각화**

```python
plt.figure(figsize=(10, 10))

plt.scatter(x_test, y_test, color="black") # Test data
plt.scatter(x_train, y_train, color="red", s=1) # Train data

plt.plot(x_test, model.predict(x_test), color="blue", linewidth=3) # Fitted line

plt.show()
```

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/68bb1ca2-fb4b-440f-8efe-40f62b774bc9/Untitled.png)
