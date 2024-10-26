```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets, model_selection, linear_model
from sklearn.metrics import mean_squared_error
```

## 1. (당뇨병 진행도) 데이터 읽어들이기

```python
diabetes = datasets.load_diabetes()
```

### **그 외 sklearn 의 datasets**

```python
diabetes['data'].shape
print(diabetes.DESCR)
```

.. _diabetes_dataset:

Diabetes dataset
----------------

Ten baseline variables, age, sex, body mass index, average blood
pressure, and six blood serum measurements were obtained for each of n =
442 diabetes patients, as well as the response of interest, a
quantitative measure of disease progression one year after baseline.

**Data Set Characteristics:**

  :Number of Instances: 442

  :Number of Attributes: First 10 columns are numeric predictive values

  :Target: Column 11 is a quantitative measure of disease progression one year after baseline

  :Attribute Information:
      - age     age in years
      - sex
      - bmi     body mass index
      - bp      average blood pressure
      - s1      tc, total serum cholesterol
      - s2      ldl, low-density lipoproteins
      - s3      hdl, high-density lipoproteins
      - s4      tch, total cholesterol / HDL
      - s5      ltg, possibly log of serum triglycerides level
      - s6      glu, blood sugar level

Note: Each of these 10 feature variables have been mean centered and scaled by the standard deviation times the square root of `n_samples` (i.e. the sum of squares of each column totals 1).

Source URL:

https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html

For more information see:
Bradley Efron, Trevor Hastie, Iain Johnstone and Robert Tibshirani (2004) "Least Angle Regression," Annals of Statistics (with discussion), 407-499.
(

https://web.stanford.edu/~hastie/Papers/LARS/LeastAngle_2002.pdf

)

---

```python
df = pd.DataFrame(diabetes.data) # Array to dataframe
df.head()
```

- 나이, 성별, BMI(body mass index) 지수, 혈압 및 6개의 혈청 검사 수치
- > 1년 후의 당뇨병 진행도

### 기본적으로 array 형태로 저장되어 있으므로 바로 활용 가능

```python
print(diabetes.data.shape) # '모양'
print(diabetes.target.shape)

>>> (442, 10)
		(442,)
```

```python
diabetes['data'][0, :]

>>> array([ 0.03807591,  0.05068012,  0.06169621,  0.02187239, -0.0442235 ,
		       -0.03482076, -0.04340085, -0.00259226,  0.01990749, -0.01764613])
```

## **2. Feature 선택하기**

```python
diabetes_X = diabetes.data[:, 2:3] # Body Mass Index, Mean centered and scaled by the standard deviation time n_samples (Sum of squares of each column totals 1)
diabetes_X

>>> array([[ 0.06169621],
		       [-0.05147406],
		       [ 0.04445121],
		       [-0.01159501],
		       [-0.03638469],
		       [-0.04069594],
								...
					 [ 0.03906215],
		       [-0.0730303 ]])
```

```python
diabetes_Y = diabetes.target
diabetes_Y

array([151.,  75., 141., 206., 135.,  97., 138.,  63., 110., 310., 101.,
        69., 179., 185., 118., 171., 166., 144.,  97., 168.,  68.,  49.,
        68., 245., 184., 202., 137.,  85., 131., 283., 129.,  59., 341.,
        87.,  65., 102., 265., 276., 252.,  90., 100.,  55.,  61.,  92.,
       259.,  53., 190., 142.,  75., 142., 155., 225.,  59., 104., 182.,
       128.,  52.,  37., 170., 170.,  61., 144.,  52., 128.,  71., 163.,
       150.,  97., 160., 178.,  48., 270., 202., 111.,  85.,  42., 170.,
       200., 252., 113., 143.,  51.,  52., 210.,  65., 141.,  55., 134.,
        42., 111.,  98., 164.,  48.,  96.,  90., 162., 150., 279.,  92.,
        83., 128., 102., 302., 198.,  95.,  53., 134., 144., 232.,  81.,
       104.,  59., 246., 297., 258., 229., 275., 281., 179., 200., 200.,
       173., 180.,  84., 121., 161.,  99., 109., 115., 268., 274., 158.,
       107.,  83., 103., 272.,  85., 280., 336., 281., 118., 317., 235.,
        60., 174., 259., 178., 128.,  96., 126., 288.,  88., 292.,  71.,
       197., 186.,  25.,  84.,  96., 195.,  53., 217., 172., 131., 214.,
        59.,  70., 220., 268., 152.,  47.,  74., 295., 101., 151., 127.,
       237., 225.,  81., 151., 107.,  64., 138., 185., 265., 101., 137.,
       143., 141.,  79., 292., 178.,  91., 116.,  86., 122.,  72., 129.,
       142.,  90., 158.,  39., 196., 222., 277.,  99., 196., 202., 155.,
        77., 191.,  70.,  73.,  49.,  65., 263., 248., 296., 214., 185.,
        78.,  93., 252., 150.,  77., 208.,  77., 108., 160.,  53., 220.,
       154., 259.,  90., 246., 124.,  67.,  72., 257., 262., 275., 177.,
        71.,  47., 187., 125.,  78.,  51., 258., 215., 303., 243.,  91.,
       150., 310., 153., 346.,  63.,  89.,  50.,  39., 103., 308., 116.,
       145.,  74.,  45., 115., 264.,  87., 202., 127., 182., 241.,  66.,
        94., 283.,  64., 102., 200., 265.,  94., 230., 181., 156., 233.,
        60., 219.,  80.,  68., 332., 248.,  84., 200.,  55.,  85.,  89.,
        31., 129.,  83., 275.,  65., 198., 236., 253., 124.,  44., 172.,
       114., 142., 109., 180., 144., 163., 147.,  97., 220., 190., 109.,
       191., 122., 230., 242., 248., 249., 192., 131., 237.,  78., 135.,
       244., 199., 270., 164.,  72.,  96., 306.,  91., 214.,  95., 216.,
       263., 178., 113., 200., 139., 139.,  88., 148.,  88., 243.,  71.,
        77., 109., 272.,  60.,  54., 221.,  90., 311., 281., 182., 321.,
        58., 262., 206., 233., 242., 123., 167.,  63., 197.,  71., 168.,
       140., 217., 121., 235., 245.,  40.,  52., 104., 132.,  88.,  69.,
       219.,  72., 201., 110.,  51., 277.,  63., 118.,  69., 273., 258.,
        43., 198., 242., 232., 175.,  93., 168., 275., 293., 281.,  72.,
       140., 189., 181., 209., 136., 261., 113., 131., 174., 257.,  55.,
        84.,  42., 146., 212., 233.,  91., 111., 152., 120.,  67., 310.,
        94., 183.,  66., 173.,  72.,  49.,  64.,  48., 178., 104., 132.,
       220.,  57.])
```

## **3. Training & Test set 으로 나눠주기**

```python
from sklearn import model_selection

x_train, x_test, y_train, y_test = model_selection.train_test_split(diabetes_X, diabetes_Y, test_size=0.3, random_state=0)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

>>> (309, 1)
		(133, 1)
		(309,)
		(133,)
```

## **4. 비어있는 모델 객체 만들기**

```python
# Create linear regression object 
model = linear_model.LinearRegression()
```

## **5. 모델 객체 학습시키기 (on training data)**

```python
# Train the model using the training sets

model.fit(x_train, y_train)

print('Coefficients: ', model.coef_)

>>> Coefficients:  [1013.17358257]
```

## **6. 학습이 끝난 모델 테스트하기 (on test data)**

```python
# Train 데이터에 대한 Model 의 Mean squared error 
print('MSE(Training data) : ', mean_squared_error(model.predict(x_train), y_train))

>>> MSE(Training data) :  3892.7208150824304
```

```python
# Test 데이터에 대한 Model 의 Mean squared error 
print('MSE(Test data) : ', mean_squared_error(model.predict(x_test), y_test))

>>> MSE(Test data) :  3921.3720274248517
```

```python
# Square root of error
np.sqrt( mean_squared_error(model.predict(x_test), y_test) )

>>> 62.62085936351282
```

## 7. 모델 시각화

```python
plt.figure(figsize=(10, 10))

plt.scatter(x_test, y_test, color="black") # Test data
plt.scatter(x_train, y_train, color="red", s=1) # Train data

plt.plot(x_test, model.predict(x_test), color="blue", linewidth=3) # Fitted line

plt.show()
```

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0d630bf6-a690-408f-82ca-eee5f5eacf91/Untitled.png)

# 🐌 전체 코드(학습률 높이기)

- x_data 한 열 말고 전체로 가져다 쓰기
- 선형 회귀 모델에서 GradientBoostingRegressor 모델로 바꾸기

```python
from sklearn import datasets, model_selection, linear_model, ensemble
from sklearn.metrics import mean_squared_error

 

# 1. Prepare the data (array!)
diabetes = datasets.load_diabetes()

# 2. Feature selection
diabetes_X = diabetes.data #[:, 2:3] 
diabetes_Y = diabetes.target

# 3. Train/Test split
x_train, x_test, y_train, y_test = model_selection.train_test_split(diabetes_X, diabetes_Y, test_size=0.3, random_state=0)

# 4. Create model object 
model = ensemble.GradientBoostingRegressor()

# 5. Train the model 
model.fit(x_train, y_train)

# 6. Test the model
print('MSE(Training data) : ', mean_squared_error(model.predict(x_train), y_train))
print('MSE(Test data) : ', mean_squared_error(model.predict(x_test), y_test))

# 7. Visualize the model
# plt.figure(figsize=(10, 10))
# plt.scatter(x_test, y_test, color="black") # Test data
# plt.scatter(x_train, y_train, color="red", s=1) # Train data
# plt.plot(x_test, model.predict(x_test), color="blue", linewidth=3) # Fitted line
# plt.show()
```
