# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 17:21:57 2020

@author: tejas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as snf
Airlines=pd.read_csv("D:\TEJAS FORMAT\EXCELR ASSIGMENTS\COMPLETED\FORECASTING\AIRLINES\Airlines+Data.csv")
Airlines.head()
Airlines.isnull().sum()
months=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
p=Airlines["Month"][0]
p[0:3]
Airlines["months"]=0
Airlines.head()

for i in range(96):
    p=Airlines["Month"][i]
    Airlines["months"][i]=p[0:3]
    
month_dummies=pd.DataFrame(pd.get_dummies(Airlines["months"]))
month_dummies=month_dummies.iloc[:,[4,3,7,0,8,6,5,1,11,10,9,2]]

Airlines1=pd.concat([Airlines,month_dummies],axis=1)
Airlines1["t"]=np.arange(1,97)
Airlines1["t_Squared"]=Airlines1["t"]*Airlines1["t"]
Airlines1["log_passengers"]=np.log(Airlines["Passengers"])

######Time Plot############
Airlines1.Passengers.plot(style="k")
#########Dividing the dataset into train & test############
Train=Airlines1.head(80)
Test=Airlines1.tail(16)
Test=Test.set_index(np.arange(1,17))

#####Building the Linear model###########
Lin_model=snf.ols("Passengers~t",data=Train).fit()
Lin_pred=pd.Series(Lin_model.predict(pd.DataFrame(Test["t"])))
Lin_rmse=np.sqrt(np.mean((np.array(Test["Passengers"])-np.array(Lin_pred))**2)) ##47.54

####Building the Exponential Model############
Exp_model=snf.ols("log_passengers~t",data=Train).fit()
Exp_pred=pd.Series(Exp_model.predict(pd.DataFrame(Test["t"])))
Exp_rmse=np.sqrt(np.mean((np.array(Test["Passengers"])-np.array(np.exp(Exp_pred)))**2))##43.79

####Building the Quadratic Model##########
Quad_model=snf.ols("Passengers~t+t_Squared",data=Train).fit()
Quad_pred=pd.Series(Quad_model.predict(pd.DataFrame(Test[["t","t_Squared"]])))
Quad_rmse=np.sqrt(np.mean((np.array(Test["Passengers"])-np.array(Quad_pred))**2))##43.65
pwd

#####Building the Additive  Seasonality#########
Add_sea=snf.ols("Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec",data=Train).fit() 
Add_sea_pred=pd.Series(Add_sea.predict(Test[["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]]))
Add_sea_rmse=np.sqrt(np.mean((np.array(Test["Passengers"])-np.array(Add_sea_pred))**2))##129.26

#####Building the Additive Quadratic Seasonality#########
Add_sea_Quad=snf.ols("Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec+t+t_Squared",data=Train).fit() 
Add_sea_Quad_pred=pd.Series(Add_sea_Quad.predict(Test[["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec","t","t_Squared"]]))
Add_sea_Quad_rmse=np.sqrt(np.mean((np.array(Test["Passengers"])-np.array(Add_sea_Quad_pred))**2))##135.32


#####Multiplicative Additive Seasonality#########
Mul_ad_sea=snf.ols("log_passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec",data=Train).fit()
Mul_ad_sea_pred=pd.Series(Mul_ad_sea.predict(Test[["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]]))
Mul_ad_rmse=np.sqrt(np.mean((np.array(Test["Passengers"])-np.array(np.exp(Mul_ad_sea_pred)))**2))##135.32.08



#####Multiplicative Additive Quadratic Seasonality#########
Mul_ad_Quad_sea=snf.ols("log_passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec+t+t_Squared",data=Train).fit()
Mul_ad_Quad_sea_pred=pd.Series(Mul_ad_Quad_sea.predict(Test[["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec","t","t_Squared"]]))
Mul_ad_sea_rmse=np.sqrt(np.mean((np.array(Test["Passengers"])-np.array(np.exp(Mul_ad_Quad_sea_pred)))**2))##23.08

#####Storing the values###########
data = {"MODEL":pd.Series(["Lin_rmse","Exp_rmse","Quad_rmse","Add_sea_rmse","Add_sea_Quad_rmse","Mul_ad_rmse","Mul_ad_sea_rmse"]),"RMSE_Values":pd.Series([Lin_rmse,Exp_rmse,Quad_rmse,Add_sea_rmse,Add_sea_Quad_rmse,Mul_ad_rmse,Mul_ad_sea_rmse])}
table_rmse=pd.DataFrame(data)
# so Mul_ad_sea_rmse has the least value among the models prepared so far 
# Predicting new values 

model_final=snf.ols("log_passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec+t+t_Squared",data=Airlines1).fit()