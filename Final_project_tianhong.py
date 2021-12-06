#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 11:37:51 2021

@author: wangtianhong
"""
import streamlit as st
import numpy as np
import seaborn as sns
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress most warnings in TensorFlow
pd.set_option('display.float_format', lambda x: '%.3f' % x)
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import sklearn
from sklearn.linear_model import LinearRegression

reg = LinearRegression(fit_intercept=True)
rng = np.random.default_rng()
st.title("Income Data for people in Tech")
st.markdown("Tianhong Wang")
st.markdown("https://github.com/tianhow1/Data-science-Final-Project")
st.write("The data is from https://www.kaggle.com/jackogozaly/data-science-and-stem-salaries")
df=pd.read_csv('./Levels_Fyi_Salary_Data.csv')

st.write("The overview of the Dataset after simple cleaning")
df = df[~df.isna().any(axis = 1)]
df

# Choice the data that has complete information and the coloums that I intend to use.


st.header("Is a PhD worth it?")

cond_mean = df["totalyearlycompensation"].groupby(df["Education"]).aggregate("mean")
cond_std = df["totalyearlycompensation"].groupby(df["Education"]).aggregate("std")
cond_mean.plot(kind="bar",figsize=(10,4),yerr = cond_std)
fig2,ax=plt.subplots()
ax = sns.boxplot(x="totalyearlycompensation", y="Education", data=df , orient = "h", showfliers = False,ax=ax)
st.pyplot(fig2)
st.write("Refers the code from https://www.kaggle.com/marcomurtinu/is-a-phd-worth-it")
Columns_list= ['basesalary','stockgrantvalue', 'bonus','totalyearlycompensation','yearsofexperience','Masters_Degree','yearsatcompany', 'Bachelors_Degree', 'Doctorate_Degree','Highschool', 'Some_College']



st.write("The increment in the average salary between the class of PhD and the class of other educations")
workerlevel=st.selectbox("Select Education Level",['Master\'s Degree', 'Bachelor\'s Degree', 'Highschool', 'Some College'])

yearly_percent_gain = (cond_mean["PhD"]-cond_mean[workerlevel])/cond_mean[workerlevel]*100

st.write(f"The Phd has an increment yearly income over {workerlevel} is {yearly_percent_gain}%")
st.write("The above result shows that PhD degree do earn more than the other degrees.")
df1=df.loc[:,['timestamp','totalyearlycompensation','basesalary','stockgrantvalue', 'bonus','yearsofexperience'
              ,'yearsatcompany','Education','Masters_Degree', 'Bachelors_Degree', 'Doctorate_Degree','Highschool', 'Some_College']]
st.write("Next, I create a new column named the year after high school Combine education time with working time. Since phD spend about 5 years in education, and masters spend about 2 years in education. And bachelor has 4 years in education.")
# st.write("I compute the mean value and standard derivations")
df['yearsafterhighschool']=df['yearsofexperience']
Master_index=np.array(df1[df1['Masters_Degree']==1].index)
Bachelor_index=np.array(df1[df1['Bachelors_Degree']==1].index)
Doctorate_index=np.array(df1[df1['Doctorate_Degree']==1].index)
df.loc[(Master_index,'yearsafterhighschool')]=df.loc[df['Masters_Degree']==1]['yearsofexperience']+6
df.loc[(Bachelor_index,'yearsafterhighschool')]=df.loc[df['Bachelors_Degree']==1]['yearsofexperience']+4
df.loc[(Doctorate_index,'yearsafterhighschool')]=df.loc[df1['Doctorate_Degree']==1]['yearsafterhighschool']+9
df
st.subheader("The relation betweeen years of experience and total yearly compensation")
i_number=st.slider("Choice the number of data in the graph(the data was chosen randomly)",1,5000)
index=range(21521)
Cindex=rng.choice(index,i_number)
def make_chart(df,d,indexs):
    df1 = df.iloc[index,].copy()
    Y=np.array(df1.loc[:,'totalyearlycompensation']).reshape(-1,1)
    X = np.array(df1.loc[:,d]).reshape(-1, 1)
    reg.fit(X,Y)
    df1["y_pred"] = reg.predict(X)
    chart = alt.Chart(df1).mark_line().encode(
        x = d,
        y = "y_pred",
        color =alt.value("black")
    )
    return chart
st.write("The black line is the first order  linear regression based on these data.")


my_chart1=alt.Chart(df.iloc[Cindex,]).mark_circle().encode(
        alt.X("yearsofexperience"),
        alt.Y("totalyearlycompensation"),
        color='Education')
chart1_pred=make_chart(df,"yearsofexperience",Cindex)

st.altair_chart(my_chart1+chart1_pred, use_container_width=True)
st.subheader("The relation betweeen years after high school and total yearly compensation")

st.write("The black line is the first order  linear regression based on these data.")

my_chart2=alt.Chart(df.iloc[Cindex,]).mark_circle().encode(
        alt.X('yearsafterhighschool'),
        alt.Y("totalyearlycompensation"),
        color='Education')
chart2_pred=make_chart(df,'yearsafterhighschool',Cindex)
    
st.altair_chart(my_chart2+chart2_pred, use_container_width=True)
st.subheader("Next, I generate four linear regression lines with different education levels to show the clear relationship between years after high school and total yearly compensation")

df_PhD = df[df['Doctorate_Degree']==1].copy()
Y=np.array(df_PhD.loc[:,'totalyearlycompensation']).reshape(-1,1)
X = np.array(df_PhD.loc[:,'yearsafterhighschool']).reshape(-1, 1)
reg.fit(X,Y)
df_PhD["y_pred"] = reg.predict(X)
chart_PhD = alt.Chart(df_PhD).mark_line().encode(
    x = 'yearsafterhighschool',
    y = "y_pred",
    color =alt.value("black")
)
df_MS= df[df['Masters_Degree']==1].copy()
MSI=rng.choice(range(9061),5000)
df_MSI=df_MS.iloc[MSI,]
Y=np.array(df_MS.loc[:,'totalyearlycompensation']).reshape(-1,1)
X = np.array(df_MS.loc[:,'yearsafterhighschool']).reshape(-1, 1)
reg.fit(X,Y)
x = np.array(df_MSI.loc[:,'yearsafterhighschool']).reshape(-1, 1)
df_MSI["y_pred"] = reg.predict(x)
chart_MS = alt.Chart(df_MSI).mark_line().encode(
    x = 'yearsafterhighschool',
    y = "y_pred",
    color =alt.value("red")
)
# generate regression with all data but plot with choosen 5000.
df_BS= df[df['Bachelors_Degree']==1].copy()
BSI=rng.choice(range(10902),5000)
df_BSI=df_BS.iloc[BSI,]
Y=np.array(df_BS.loc[:,'totalyearlycompensation']).reshape(-1,1)
X = np.array(df_BS.loc[:,'yearsafterhighschool']).reshape(-1, 1)
x=np.array(df_BSI.loc[:,'yearsafterhighschool']).reshape(-1, 1)
reg.fit(X,Y)
df_BSI["y_pred"] = reg.predict(x)
chart_BS = alt.Chart(df_BSI).mark_line().encode(
    x = 'yearsafterhighschool',
    y = "y_pred",
    color =alt.value("blue")
)
df_HS= df[df['Highschool']==1].copy()
Y=np.array(df_HS.loc[:,'totalyearlycompensation']).reshape(-1,1)
X = np.array(df_HS.loc[:,'yearsafterhighschool']).reshape(-1, 1)
reg.fit(X,Y)
df_HS["y_pred"] = reg.predict(X)
chart_HS = alt.Chart(df_HS).mark_line().encode(
    x = 'yearsafterhighschool',
    y = "y_pred",
    color =alt.value("green")
)
df_CS= df[df['Some_College']==1].copy()
Y=np.array(df_CS.loc[:,'totalyearlycompensation']).reshape(-1,1)
X = np.array(df_CS.loc[:,'yearsafterhighschool']).reshape(-1, 1)
reg.fit(X,Y)
df_CS["y_pred"] = reg.predict(X)
chart_CS = alt.Chart(df_CS).mark_line().encode(
    x = 'yearsafterhighschool',
    y = "y_pred",
    color =alt.value("purple")
)
st.altair_chart(chart_PhD+chart_MS+chart_BS+chart_HS+chart_CS, use_container_width=True)

st.write("Phd fit line=black, Master fit line=red, Bachelor fit line=blue, Highschool fit line =green, Some college fit line= purple")
st.write("From this graph, it is obvious that PhD program is definitly a great investment. Spending same amount time after high school, it always at the highest payment. In the same view the BS and MS degrees don't show their value in the first 40 years after highschool.")
st.subheader("I generate a machine learning training to predict the education")
Train= st.selectbox("Which education level do you want to train",['Masters_Degree', 'Bachelors_Degree', 'Doctorate_Degree','Highschool', 'Some_College'])
df2 = df
scaler = StandardScaler()
scaler.fit(df2[['basesalary','stockgrantvalue', 'bonus','totalyearlycompensation','yearsofexperience','yearsatcompany']])
df2[['basesalary','stockgrantvalue', 'bonus','totalyearlycompensation','yearsofexperience','yearsatcompany']] = scaler.transform(df2[['basesalary','stockgrantvalue', 'bonus','totalyearlycompensation','yearsofexperience','yearsatcompany']])
# if 'x_value' in st.session_state:
#     model=st.session_state['x_value'] 
#     history=st.session_state['y_value']
# else:
    

X_train = df2[['basesalary','stockgrantvalue','bonus','totalyearlycompensation','yearsofexperience','yearsatcompany']]
y_train=df2[Train]
model = keras.Sequential(
    [
     keras.layers.InputLayer(input_shape = (6,)),
     keras.layers.Dense(16, activation="relu"),
     keras.layers.Dense(8, activation="relu"),
     keras.layers.BatchNormalization(),
     keras.layers.Dense(1,activation="sigmoid")
     ]
    )

model.compile(
    loss="binary_crossentropy", 
    optimizer=keras.optimizers.SGD(learning_rate=0.001),
    metrics=["accuracy"],
    )

history = model.fit(X_train,y_train,epochs=20, validation_split = 0.2)

    
    #st.session_state['x_value']=model
    #st.session_state['y_value']=history
fig1, ax = plt.subplots()
ax.plot(history.history['loss'])
ax.plot(history.history['val_loss'])
ax.set_ylabel('loss')
ax.set_xlabel('epoch')
ax.legend(['train', 'validation'], loc='upper right')
st.pyplot(fig1)
st.write("It shows that for PhD degree and High school degree the training isn't overfitting, but this mode is not that good for master.")
st.subheader(f"Try to predict if the worker is {Train}")

basesalary=st.number_input("The basesalary is")
stockgrantvalue=st.number_input("The stockgrantvalue is")
bonus=st.number_input("The bonus is")
totalyearlycompensation=st.number_input("The total yearly compensation is")
yearsofexperience=st.number_input("The years of experience is")
yearsatcompany=st.number_input("The years at company is")
scaler.fit(df2[['basesalary','stockgrantvalue', 'bonus','totalyearlycompensation','yearsofexperience','yearsatcompany']])
a=np.array([basesalary,stockgrantvalue,bonus,totalyearlycompensation,yearsofexperience,yearsatcompany]).reshape(1,6)
a=scaler.transform(a)
T=model.predict(a)
if T>0.5:
    key_word="Yes, it is"
else:
    key_word="No, it isn't"
st.write(f"{key_word} {Train}")




