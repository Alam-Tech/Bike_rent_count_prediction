#%%
import pandas as pd
import numpy as np
np.set_printoptions(precision=4,suppress=True)
#%%
#Importing the data:
train_data=pd.read_csv('D:\ML_Comps\Bike_sharing_demand_pred\Original_data\TRAIN.csv')
train_len=len(train_data)
test_data=pd.read_csv('D:\ML_Comps\Bike_sharing_demand_pred\Original_data\TEST.csv')
data=pd.concat((train_data,test_data),axis=0)
#%%
#Checking for null values:
# for col in data.columns:
#     extract = data[data[col].isna()]
#     print(f'{col} : {len(extract)}')
#No null values in the data
#%%
#Date functions:
def get_day(date):
    date=date.split('/')
    return int(date[0])

def get_month(date):
    date=date.split('/')
    return int(date[1])

def get_year(date):
    date=date.split('/')
    return int(date[-1])
#%%
#Processing the data:
def process_data(target):
    #Data cleaning:
    target['Day']=target['Date'].apply(lambda x:get_day(x))
    target['Month']=target['Date'].apply(lambda x:get_month(x))
    # target['Year']=target['Date'].apply(lambda x:get_year(x))

    del target['Date']
    del target['Index']

    #encoding the categorical variabes:
    target=pd.get_dummies(target,columns=['Seasons','Holiday','Functioning Day','Month'],drop_first=True)

    #Binning the dates:
    bins=[0,5,10,15,20,25,30,35]
    labels=[0,1,2,3,4,5,6]
    target['Day_bin']=pd.cut(target['Day'],bins=bins,right=True,labels=labels)
    del target['Day']

    #encoding the bin values of the target:
    target=pd.get_dummies(target,columns=['Day_bin'],drop_first=True)
    return target
# %%
# train_data = process_data(train_data)
# test_data = process_data(test_data)
data = process_data(data)
col_order=[ 'Hour', 'Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)',
       'Visibility (10m)', 'Dew point temperature(°C)',
       'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)', 'Seasons_Spring', 'Seasons_Summer',
       'Seasons_Winter', 'Holiday_No Holiday', 'Functioning Day_Yes',
       'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6', 'Month_7',
       'Month_8', 'Month_9','Month_10','Month_11', 'Month_12', 'Day_bin_1', 'Day_bin_2', 'Day_bin_3',
       'Day_bin_4', 'Day_bin_5', 'Day_bin_6','Rented Bike Count']
# train_data = train_data[col_order]
# test_data = test_data[col_order]
data = data[col_order]
#%%
#Preparing the evaluation function:
from sklearn.model_selection import cross_val_score
def evaluate(regressor,ind_vars,dep_vars,name):
    scores=cross_val_score(regressor,X=ind_vars,y=dep_vars,cv=10,n_jobs=-1,scoring='neg_root_mean_squared_error')
    print(f'{name}:')
    print(f'The mean rmse is {scores.mean()}')
    print(f'The std.deviaion is {scores.std()}')
# %%
#Linear Regression:
# Linear Regression:
# The mean rmse is -445.47761253374193
# The std.deviaion is 133.19329943718648

# del data['Month_5'],data['Month_8'],data['Month_9'],data['Month_10'],data['Month_11']

# ind_train=data.iloc[:train_len,:-1].values
# dep_train=data.iloc[:train_len,-1].values

# from sklearn.linear_model import LinearRegression
# regressor=LinearRegression()

# evaluate(regressor,ind_train,dep_train,'Linear Regression')
# %%
#Non-Linear Regression:
# GradBoostReg:
# The mean rmse is -246.0850438852352
# The std.deviaion is 94.68234219235138
# ExtraTreeReg:
# The mean rmse is -257.9805599827714
# The std.deviaion is 106.63336669731588
# AdaBoostReg:
# The mean rmse is -482.84547969358243
# The std.deviaion is 48.65482533112536
# XGBReg:
# The mean rmse is -267.2840329237818
# The std.deviaion is 107.65097442040123
# XGBRFReg:
# The mean rmse is -298.86835074663287
# The std.deviaion is 131.43360636021157

# from sklearn.ensemble import \
#     RandomForestRegressor,GradientBoostingRegressor,ExtraTreesRegressor,AdaBoostRegressor
# from xgboost import XGBRegressor,XGBRFRegressor

ind_train=data.iloc[:train_len,:-1].values
dep_train=data.iloc[:train_len,-1].values

# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# ind_train=scaler.fit_transform(ind_train)

# regressors=[
#     GradientBoostingRegressor(n_estimators=500),
#     ExtraTreesRegressor(n_estimators=500),
#     AdaBoostRegressor(n_estimators=200),
#     XGBRegressor(n_estimators=500),
#     XGBRFRegressor(n_estimators=500)
# ]
# names=['GradBoostReg','ExtraTreeReg','AdaBoostReg','XGBReg','XGBRFReg']

# for reg,name in zip(regressors,names):
#     evaluate(reg,ind_train,dep_train,name)
# %%
# Model tuning:
# For first 12 variable(s)(500 est):
# The mean rmse is -244.48910348356313
# The std.deviaion is 94.91301107844511

# from sklearn.ensemble import GradientBoostingRegressor
# est_list=[100,300,500,1000]
# learning_rates=[1,0.01,0.02,0.001]
# n_estimators: 1000, lr: 0.02:
# The mean rmse is -245.1338961848931
# The std.deviaion is 100.45753981668197

# for est in est_list:
#     for lr in learning_rates:
#         regressor=GradientBoostingRegressor(n_estimators=est,learning_rate=lr)
#         evaluate(regressor,ind_train,dep_train,f'n_estimators: {est}, lr: {lr}')

# indices=np.array(list(range(0,len(ind_train[0]))),dtype=int)
# indices=indices.reshape(-1,1)
# scores=np.array(regressor.feature_importances_).ravel()
# scores=scores.reshape(-1,1)
# score_table=np.append(indices,scores,axis=1)
# score_table=score_table[score_table[:,-1].argsort()[::-1]]

# target_vars=list(map(int,score_table[:15,0]))
# %%
# while len(target_vars)!=0:
#     regressor=GradientBoostingRegressor(n_estimators=500)
#     evaluate(regressor,ind_train[:,target_vars],dep_train,f'For first {len(target_vars)} variable(s)')
#     del target_vars[-1]
# %%
