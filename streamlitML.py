'''

import streamlit as st  
import pandas as pd
import numpy as np
import pickle
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
#from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


df=pd.read_csv('Loan payments data.csv',usecols=['loan_status','Principal','terms','due_date','paid_off_time','past_due_days','age','education','Gender'])

print(df.head())
#from label encoder from the label encording
effective_date=LabelEncoder()
loan_status=LabelEncoder()
education=LabelEncoder()
gender=LabelEncoder()


df['loan_status']=loan_status.fit_transform(df['loan_status'])
df['education']=education.fit_transform(df['education'])
df['Gender']=gender.fit_transform(df['Gender'])

# convert date columns to numeric timestamps
df['due_date'] = pd.to_datetime(df['due_date']).astype('int64') // 10**9
df['paid_off_time'] = pd.to_datetime(df['paid_off_time']).astype('int64') // 10**9

X=df.iloc[:,2:10]  
y=df.iloc[:,-1]

#from sklearn standard scaler
#scaler=StandardScaler()
#X_scaler=scaler.fit_transform(X)
#=print(X_scaler)

#from sklearn train test data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print(X_train.shape)
print(y_train.shape)

#from sklaern random frorest
work=RandomForestClassifier()
work.fit(X_train,y_train)
y_pred=work.predict(X_test)
accc=accuracy_score(y_test,y_pred)
cmm=confusion_matrix(y_test,y_pred)
R2=r2_score(y_test,y_pred)
mm=mean_squared_error(y_test,y_pred)
print(accc)
print(cmm)
print(mm)
print(R2)
with open('work.pkl','wb')as f:
 pickle.dump(work,f)
print("Model trained and saved as model pkl")'''


from pandas import read_csv
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
df=read_csv('house_data.csv')
print(df.head)
X=df[['size','bedrooms']]
y=df['price']
#from sklearn.linear regression
model=LinearRegression()
model.fit(X,y)
#pickle model
with open("house_model.pkl","wb")as f:
 pickle.dump(model,f)
print("model train and save as house_model.pkl:" )
