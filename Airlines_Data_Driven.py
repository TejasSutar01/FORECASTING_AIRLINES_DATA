# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 11:52:16 2020

@author: tejas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime,time
import statsmodels.graphics.tsaplots as tsa_plots
import seaborn as sn
import statsmodels.api as smf

airlines=pd.read_csv("D:\TEJAS FORMAT\EXCELR ASSIGMENTS\COMPLETED\FORECASTING\AIRLINES\Airlines+Data.csv")
airlines["Date"]=pd.to_datetime(airlines["Month"].str.replace(r'-(\d+)$', r'-19\1'))
airlines["month"]=airlines.Date.dt.strftime("%b")
airlines["year"]=airlines.Date.dt.strftime("%Y")

heat_map=pd.pivot_table(airlines,values="Passengers",index="year",columns="month",aggfunc="mean",fill_value=0)
sn.heatmap(heat_map,annot=True,fmt="g")

#######Boxplots#######
sn.boxplot(x="month",y="Passengers",data=airlines)
sn.boxplot(x="year",y="Passengers",data=airlines)
sn.lineplot(x="year",y="Passengers",hue="month",data=airlines)

####Moving average to understand better about trend#######
for i in range(2,24,26):
    airlines["Passengers"].rolling(i).mean().plot(label=str(i))
    plt.legend(loc=4)
    
seasonal_ts_add=smf.tsa.seasonal_decompose(airlines["Passengers"],freq=10)
seasonal_ts_add.plot()
train=airlines.head(92)
test=airlines.tail(4)

#MAPE####
def MAPE(pred,org):
    temp=np.abs((pred-org))*100/org
    return np.mean(temp)

####Simple Exp##########
Exp=SimpleExpSmoothing(train["Passengers"]).fit()
Exp_pred=Exp.predict(start=test.index[0],end=test.index[-1])
Exp_mape=MAPE(Exp_pred,test.Passengers)   ######32.05

###Holt#######
hw=Holt(train["Passengers"]).fit()
hw_pred=hw.predict(start=test.index[0],end=test.index[-1])
hw_mape=MAPE(hw_pred,test.Passengers)#####34.75

# Holts winter exponential smoothing with additive seasonality and additive trend
Exp_add_add=ExponentialSmoothing(train["Passengers"],damped=True,seasonal="add",seasonal_periods=12,trend="add").fit()
Exp_add_add_pred=Exp_add_add.predict(start=test.index[0],end=test.index[-1])
Exp_add_add_Mape=MAPE(Exp_add_add_pred,test.Passengers)#####4.23

#####Holts winter Exp smoothing Multiplicative trend with  add seasonality#########
Exp_mul_add=ExponentialSmoothing(train["Passengers"],damped=True,seasonal="mul",seasonal_periods=12,trend="add").fit()
Exp_mul_add_pred=Exp_mul_add.predict(start=test.index[0],end=test.index[-1])
Exp_mul_add_mape=MAPE(Exp_mul_add_pred,test.Passengers)######4.49

Table={"Model":pd.Series(["Exp_mape","hw_mape","Exp_add_add_Mape","Exp_mul_add_mape"]),"MAPE VAlUES":pd.Series([Exp_mape,hw_mape,Exp_add_add_Mape,Exp_mul_add_mape])}
Table=pd.DataFrame(Table) 

plt.plot(train.index,train["Passengers"],label="Train",color="black") 
plt.plot(test.index,test["Passengers"],label="Test",color="blue")
plt.plot(Exp_pred.index,Exp_pred,label="Simple Exp Smoothing",color="yellow")
plt.plot(hw_pred.index,hw_pred,label="Holts method",color="orange")
plt.plot(Exp_add_add_pred.index,Exp_add_add_pred,label="Exp Smoothing with add trend & add seasonality",color="blue")
plt.plot(Exp_mul_add_pred.index,Exp_mul_add_pred,label="Exp Smoothing with add trend & mul seasonality",color="violet")