#!/usr/bin/env python
# coding: utf-8

# In[35]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import tensorflow as tf
import keras
import seaborn as sn




# In[36]:


df = pd.read_csv("An ANN Model for predicting distance learning students performance.csv")
df.sample(5)


# In[37]:


df.drop('Timestamp', axis='columns', inplace=True)
df.dtypes


# In[38]:


high_fin = df[df.FinancialAssistance=='High'].OnlinePracticalTest
med_fin = df[df.FinancialAssistance=='Medium'].OnlinePracticalTest
low_fin = df[df.FinancialAssistance=='Low'].OnlinePracticalTest

plt.xlabel('Student Performance')
plt.ylabel('Numbers of Students')
plt.title('Students Perfromance Prediction Visualization With Respect To Financial Assistance')

plt.hist([high_fin,med_fin,low_fin],color=['green','yellow','red'],label=['High Finance','Medium Finance','Low Finance'])
plt.legend()


# In[39]:


high_part = df[df.OnlineClassParticipation=='High'].OnlinePracticalTest
med_part = df[df.OnlineClassParticipation=='Medium'].OnlinePracticalTest
low_part = df[df.OnlineClassParticipation=='Low'].OnlinePracticalTest

plt.xlabel('Student Performance')
plt.ylabel('Numbers of Students')
plt.title('Students Perfromance Prediction Visualization With Respect To Online Class Participation')

plt.hist([high_part,med_part,low_part],color=['green','yellow','red'],label=['High Participation','Medium Participation','Low Participation'])
plt.legend()


# In[40]:


high_perf = df[df.PerformanceInSecondarySchool=='High'].OnlinePracticalTest
med_perf = df[df.PerformanceInSecondarySchool=='Medium'].OnlinePracticalTest
low_perf = df[df.PerformanceInSecondarySchool=='Low'].OnlinePracticalTest

plt.xlabel('Student Performance')
plt.ylabel('Numbers of Students')
plt.title('Students Perfromance Prediction Visualization With Respect To Performance In Secondary School')

plt.hist([high_perf,med_perf,low_perf],color=['green','yellow','red'],label=['High Performance','Medium Performance','Low Performance'])
plt.legend()


# In[41]:


urban_area = df[df.StudentLocation=='Urban'].OnlinePracticalTest
rural_area = df[df.StudentLocation=='Rural'].OnlinePracticalTest

plt.xlabel('Student Performance')
plt.ylabel('Numbers of Students')
plt.title('Students Perfromance Prediction Visualization With Respect To Student Location')

plt.hist([urban_area,rural_area],color=['green','red'],label=['Urban','Rural'])
plt.legend()


# In[42]:


fast_int = df[df.FastInternetService=='Yes'].OnlinePracticalTest
slow_int = df[df.FastInternetService=='No'].OnlinePracticalTest

plt.xlabel('Student Performance')
plt.ylabel('Numbers of Students')
plt.title('Students Perfromance Prediction Visualization With Respect To Fast Internet Service')

plt.hist([fast_int,slow_int],color=['green','red'],label=['Fast Internet','Slow Internet'])
plt.legend()


# In[43]:


def print_column_unique_values(df):
    for column in df:
        if df[column].dtypes=='object':
            print(f'{column}: {df[column].unique()}')


# In[44]:


print_column_unique_values(df)


# In[45]:


df['StudentLocation'].replace({'Urban': 1, 'Rural': 0},inplace=True)


# In[46]:


yes_no_cols = ['EarlyRegistration', 'FastInternetService']
    
for col in yes_no_cols:
    df[col].replace({'Yes': 1,'No': 0},inplace=True)


# In[47]:


for column in df:
    print(f'{column}: {df[column].unique()}')


# In[48]:


df = pd.get_dummies(data=df, columns=['OnlineClassParticipation','AssignmentSubmission','ParentGuardianEducation',
                     'PerformanceInSecondarySchool','OnlinePracticalTest','FinancialAssistance'])
df.columns


# In[49]:


# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# df['OnlinePracticalTest'] = scaler.fit_transform(df['OnlinePracticalTest'])


# In[50]:


for column in df:
    print(f'{column}: {df[column].unique()}')


# In[51]:


df.dtypes


# In[52]:


df.sample(5)


# In[53]:


X = df.drop(['OnlinePracticalTest_High','OnlinePracticalTest_Medium','OnlinePracticalTest_Low'],axis='columns')
y = df['OnlinePracticalTest_High']


# In[54]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=5)


# In[55]:


X_train.shape


# In[56]:


X_test.shape


# In[57]:


len(X_train.columns)


# In[58]:


model = keras.Sequential([
    keras.layers.Dense(16, input_shape=(18,), activation='relu'),
    keras.layers.Dense(14, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=100)


# In[59]:


model.evaluate(X_test, y_test)


# In[60]:


y_pred = model.predict(X_test)
y_pred[:10]


# In[61]:


y_test[:10]


# In[62]:


y_pred_output = []
for elem in y_pred:
    if elem >= 0.5:
        y_pred_output.append(1)
    else:
        y_pred_output.append(0)


# In[63]:


y_pred_output[:10]


# In[64]:


from sklearn.metrics import confusion_matrix, classification_report

print(classification_report(y_test,y_pred_output))


# In[65]:


cm = tf.math.confusion_matrix(labels=y_test,predictions=y_pred_output)

plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[66]:


score = accuracy_score(y_test, y_pred_output)
score


# In[67]:


# save trained model
# joblib.dump(model, 'student_performance_predictor.joblib')


# In[68]:


# load trained model
# model = joblib.load('student_performance_predictor.joblib')

