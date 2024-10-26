### 🐘 Gradient Boosting(GBM)?

- 주요 앙상블 알고리즘은 bagging과 boostiong으로 나눌 수 있고, Gradient boosting은 boosting 계열의 앙상블 알고리즘이다.
- Boosting이란 약한 분류기를 결합하여 강한 분류기를 만드는 과정이다.
- Gradient Boosting(GBM) 은 잔차를 이용하여 이전 모형의 약점을 보완한다.
    
    → 하지만 과대적합이 일어날 수 있는 단점을 지니고 있음
    
- GBM은 순차적으로 적합한 뒤 이들을 선형 결합한 모형을 생성한다

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a9943cca-db10-4d1f-a8ca-614738f31bb3/Untitled.png)

ROC Curve: 수신자 조작 특성 곡선

AUC = Area Under the ROC Curve (0.5~1) → ROC Curve 밑의 면적

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, model_selection, linear_model
from sklearn.metrics import mean_squared_error, accuracy_score, roc_curve, auc

# 1. Prepare the data (array!)
df_data = pd.read_excel('/content/drive/MyDrive/카카오 sw ai 중급 2(part5)/실습 파일/2. Scikit-learn (Answer)/boston_house_data.xlsx', index_col=0)
df_target = pd.read_excel('/content/drive/MyDrive/카카오 sw ai 중급 2(part5)/실습 파일/2. Scikit-learn (Answer)/boston_house_target.xlsx')
df_target['Label'] = df_target[0].apply(lambda x: 1 if x > df_target[0].mean() else 0 ) 
boston_data = np.array(df_data)
boston_target = np.array(df_target['Label'])

# 2. Feature selection
boston_X = boston_data[:,5:13] # 주택당 방 수 & 인구 중 하위 계층 비율 
boston_Y = boston_target

# 3. Train/Test split
x_train, x_test, y_train, y_test = model_selection.train_test_split(boston_X, boston_Y, test_size=0.3, random_state=0)

# 4. Create model object 
model = linear_model.LogisticRegression()

# 5. Train the model 
model.fit(x_train, y_train)

# 6. Test the model
print('Accuracy: ', accuracy_score(model.predict(x_test), y_test))

# 7. Visualize the model
pred_test = model.predict_proba(x_test) # Predict 'probability'
fpr, tpr, _ = roc_curve(y_true=y_test, y_score=pred_test[:,1]) # real y & predicted y (based on "Sepal width")
roc_auc = auc(fpr, tpr) # AUC 면적의 값 (수치)

plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.title("ROC curve")
plt.show()
```

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9bbbf53e-8a77-49c5-9793-2dbb3425dd92/Untitled.png)

```python
**plt.plot(fpr, tpr, color='darkorange', 
lw=2, label='ROC curve (area = %0.2f)' % roc_auc)**
```

- **fpr** : False Positive Rate, tpr: True Positive Rate를 x축과 y축으로 하는 ROC 곡선을 그립니다.
- **color** : 곡선의 색상을 지정합니다.
- **lw** : 곡선의 두께를 지정합니다.
- **label** : 곡선에 대한 라벨을 지정합니다. roc_auc는 ROC 곡선 아래 면적(AUC)을 나타냅니다.

```python
**plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')**
```

- [0, 1]과 [0, 1]을 연결하는 대각선을 그립니다.
- color: 대각선의 색상을 지정합니다.
- lw: 대각선의 두께를 지정합니다.
- linestyle: 대각선의 스타일을 지정합니다.

### ***ps**

**TP= 실제 Positive를 모델이 Positive로 옳게 예측한 수**

**TN = 실제 Negative를 모델이 Negative로 옳게 예측한 수**

**FP= 실제 Negative를 모델이 Positive로 잘못 예측한 수**

**FN = 실제 Positive를 모델이 Negative로 잘못 예측한 수**

FPR와 TPR은 Confusion Matrix의 요소들로 다음과 같이 나타낼 수 있습니다.

- FPR은 실제 Negative에서 모델이 Positive라고 예측한 비율을 뜻하며, 다음과 같이 표현합니다.
- FPR = FPTN+FP

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5bc2475b-ba9a-48db-8ef4-e370f27b1e9c/Untitled.png)

- TPR은 실제 Positive에서 모델이 Positive라고 예측한 비율을 뜻하며, 다음과 같이 표현합니다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a4ef7a68-ec54-4c92-a3ce-206cff30c04b/Untitled.png)

[Gradient Boosting regression — scikit-learn 0.20.4 documentation](https://scikit-learn.org/0.20/auto_examples/ensemble/plot_gradient_boosting_regression.html#sphx-glr-auto-examples-ensemble-plot-gradient-boosting-regression-py)
