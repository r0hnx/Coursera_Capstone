# Car Accident Severity Report Analysis with IBM Watson

# **A.** Introduction :fire:
In an effort to reduce the frequency of car collisions in a community, an algorithm must be developed to predict the severity of an accident given the current weather, road and visibility conditions. When conditions are bad, this model will alert drivers to remind them to be more careful.

# **B.** Data Understanding :microscope:
Our predictor or target variable will be `SEVERITYCODE` because it is used measure the severity of an accident from 0 to 5 within the dataset. Attributes used to weigh the severity of an accident are `WEATHER`, `ROADCOND` and `LIGHTCOND`.

Severity codes are as follows:
```
0 : Little to no Probability (Clear Conditions)

1 : Very Low Probability - Chance or Property Damage

2 : Low Probability - Chance of Injury

3 : Mild Probability - Chance of Serious Injury

4 : High Probability - Chance of Fatality
```

# **C.** Extract Dataset & Convert :factory:
In it's original form, this data is not fit for analysis. For one, there are many columns that we will not use for this model. Also, most of the features are of type object, when they should be numerical type.
We must use label encoding to covert the features to our desired data type.

![Alt Text](https://dev-to-uploads.s3.amazonaws.com/i/ermczhhm06ahztqzzcyh.jpg)

With the new columns, we can now use this data in our analysis and ML models!

Now let's check the data types of the new columns in our data. Moving forward, we will only use the new columns for our analysis.

![Alt Text](https://dev-to-uploads.s3.amazonaws.com/i/nsq9b2v5msvnxee5ks27.jpg)

# **D.** Balancing the Dataset :memo:
Our target variable `SEVERITYCODE` is only 42% balanced. In fact, severity code in class 1 is nearly three times the size of class 2.

We can fix this by down-sampling the majority class.
![Alt Text](https://dev-to-uploads.s3.amazonaws.com/i/7bnpn46q49aca8joc4o0.jpg)

Perfectly balanced ( as all things should be! )

# **E.** Methodology :floppy_disk:
Our data is now ready to be fed into machine learning models.

We will use the following models:

* **K-Nearest Neighbor (KNN)**
KNN will help us predict the severity code of an outcome by finding the most similar to data point within k distance.

* **Decision Tree**
A decision tree model gives us a layout of all possible outcomes so we can fully analyze the consequences of a decision. It context, the decision tree observes all possible outcomes of different weather conditions.

* **Logistic Regression**
Because our dataset only provides us with two severity code outcomes, our model will only predict one of those two classes. This makes our data binary, which is perfect to use with logistic regression.

Let's get started!

### Initialization :crown:
* Define X and y
* Normalize the dataset
* Train-Test Split
![Alt Text](https://dev-to-uploads.s3.amazonaws.com/i/pi0ful2zrz1h8encfpdi.jpg)

### Modeling :bar_chart:
* **K-Nearest Neighbor**
Finding the best k value
![Alt Text](https://dev-to-uploads.s3.amazonaws.com/i/nqofr4hxl87erieyyb8g.jpg)
```python
#Train Model & Predict  
k = mean_acc.argmax()+1
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh

Kyhat = neigh.predict(X_test)
Kyhat[0:5]
```
`array([2, 2, 1, 1, 2])`

* **Decision Tree**
```python
# Building the Decision Tree
from sklearn.tree import DecisionTreeClassifier
colDataTree = DecisionTreeClassifier(criterion="entropy", max_depth = 7)
colDataTree
colDataTree.fit(X_train,y_train)
predTree = colDataTree.predict(X_test)
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, predTree))
```
`DecisionTrees's Accuracy:  0.5664365709048206`
```python
# Train Model & Predict
DTyhat = colDataTree.predict(X_test)
print (predTree [0:5])
print (y_test [0:5])
```
`[2 2 1 1 2]
[2 2 1 1 1]`

* **Logistic Regression**
```python
# Building the LR Model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=6, solver='liblinear').fit(X_train,y_train)

# Train Model & Predicr
LRyhat = LR.predict(X_test)

yhat_prob = LR.predict_proba(X_test)
```

# **F.** Summary :books:

Here is the summary of the scores reported in the evaluation step:

| Algorithm | Jaccard | F1 Score | LogLoss|
| :-------- | :------ | :------- | :----- |
| KNN | 0.56 | 0.55 | NA |
| Decision Tree | 0.56 | 0.54 | NA |
| Logistic Regression | 0.52 | 0.51 | 0.68 |

# **G.** Discussion :coffee:
In the beginning of this notebook, we had categorical data that was of type 'object'. This is not a data type that we could have fed through an algorithm, so label encoding was used to created new classes that were of type int8; a numerical data type.

After solving that issue we were presented with another - imbalanced data. As mentioned earlier, class 1 was nearly three times larger than class 2. The solution to this was down-sampling the majority class with sklearn's resample tool. We down-sampled to match the minority class exactly with 58188 values each.

Once we analyzed and cleaned the data, it was then fed through three ML models; K-Nearest Neighbor, Decision Tree and Logistic Regression. Although the first two are ideal for this project, logistic regression made most sense because of its binary nature.

Evaluation metrics used to test the accuracy of our models were jaccard index, f-1 score and log-loss for logistic regression. Choosing different k, max depth and hypermeter C values helped to improve our accuracy to be the best possible.

# **H.** Conclusion :dart:
Based on historical data from weather conditions pointing to certain classes, we can conclude that particular weather conditions have a somewhat impact on whether or not travel could result in property damage (class 1) or injury (class 2).

Thank you for reading! :blush:
