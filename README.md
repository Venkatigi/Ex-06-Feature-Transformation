# Ex-06-Feature-Transformation

# AIM
To read the given data and perform Feature Transformation process and save the data to a file. 

# EXPLANATION
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature Transformation techniques to all the features of the data set
### STEP 4
Save the data to the file

# CODE
```
Develpoed By    : Venkatesh E
Register Number : 212221230119
```
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer

df=pd.read_csv("D:\ggg\clg\Documents\SEM 3\Data Science\Projects\Ex-06-Feature-Transformation/Data_to_Transform.csv")
df.head()

df.info()
df.describe()
df.isnull().sum()
df.shape
df.columns
df.duplicated()

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Highly Negative Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = np.log(df['Highly Positive Skew'])

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Moderate Positive Skew'] = np.log(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = 1/df['Highly Positive Skew']

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = df['Highly Positive Skew']**(1/1.2)

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Moderate Positive Skew_1'], parameters=stats.yeojohnson(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()
```
# OUPUT

![1](https://github.com/Venkatigi/Ex-06-Feature-Transformation/blob/main/images/1.JPG)
![](https://github.com/Venkatigi/Ex-06-Feature-Transformation/blob/main/images/2.JPG)
![](https://github.com/Venkatigi/Ex-06-Feature-Transformation/blob/main/images/3.JPG)
![](https://github.com/Venkatigi/Ex-06-Feature-Transformation/blob/main/images/4.JPG)
![](https://github.com/Venkatigi/Ex-06-Feature-Transformation/blob/main/images/5.JPG)
![](https://github.com/Venkatigi/Ex-06-Feature-Transformation/blob/main/images/6.JPG)
![](https://github.com/Venkatigi/Ex-06-Feature-Transformation/blob/main/images/7.JPG)

![](https://github.com/Venkatigi/Ex-06-Feature-Transformation/blob/main/images/8.JPG)
![](https://github.com/Venkatigi/Ex-06-Feature-Transformation/blob/main/images/9.JPG)
![](https://github.com/Venkatigi/Ex-06-Feature-Transformation/blob/main/images/10.JPG)
![](https://github.com/Venkatigi/Ex-06-Feature-Transformation/blob/main/images/11.JPG)
![](https://github.com/Venkatigi/Ex-06-Feature-Transformation/blob/main/images/12.JPG)
![](https://github.com/Venkatigi/Ex-06-Feature-Transformation/blob/main/images/13.JPG)
![](https://github.com/Venkatigi/Ex-06-Feature-Transformation/blob/main/images/14.JPG)
# RESULT 
Thus the Feature Transformation for the given datasets had been executed successfully
