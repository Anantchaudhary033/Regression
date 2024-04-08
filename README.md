# Regression
Regression is a method for understanding the relationship between independent variables/features and a dependent variable/outcome.
Linear regression involves finding a line that minimizes the difference between predicted and actual output values, typically achieved through the least squares method. Simple linear regression calculators utilize this approach to determine the best-fit line for a given set of paired data, allowing estimation of the dependent variable (X) based on the independent variable (Y).

# Problem Statement
We have a dataset comprising salaries of individuals alongside their respective years of professional experience. Our objective is to develop a linear regression model to accurately predict salaries based on an individual's experience. Additionally, we aim to assess the model's accuracy once established. 

# Variables
Dependent Variable - Salary
Independent Variable - YearsExperience
------------------------------- 
#Importing Libraries
-------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from pandas.core.common import random_state
from sklearn.linear_model import LinearRegression

# Dataset
-------------------------------
df=pd.read_csv('Salary_Data.csv')
df

# Show top and bottom rows in Dataset
Top rows- df.head()
Bottom rows- df.tail()

# Status summary(To define the summary of numeric value,like- mean,median,mode etc.)
df.describe()

# Distribution plot
plt.title('salary Distribution plot')
sns.distplot(df['Salary'])
plt.show()

# Scatter plot
plt.scatter(df['YearsExperience'],df['Salary'],color = 'lightcoral')
plt.title('salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.box(False)
plt.show()

# Find out dependent and independent variable
x = df.loc[:,['YearsExperience']]
x.head()
y = df.loc[:,['Salary']]
y.head()

# Split the data set
x_train,x_test,y_train,y_test =  train_test_split(x,y,test_size=0.3,random_state=42)

# Train and fit the dataset
reg=LinearRegression()
reg.fit(x_train,y_train)

# Predict the dataset
y_pred_test = reg.predict(x_test)
y_pred_train = reg.predict(x_train)

# Accuracy of training and testing data
reg.score(x_test,y_test)

# validation(for x=1.2)
intercept=reg.intercept_
coefficient=reg.coef_
def formula(x):
    salary= coefficient*x+intercept
    print (salary)    
formula(1.2)

# visualization(Best fit line)
plt.scatter(x_train,y_train,color='lightcoral')
plt.plot(x_train,y_pred_train,color='firebrick')
plt.title('Salary vs Experience(Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend(['x_train/pred(y_test)','x_train/y_train'],title= 'Sal/Exp',loc='best')




