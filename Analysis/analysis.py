#%%
#Importing the libraries:
import pandas as pd
import numpy as np
#%%
#Importing the data files:
train_data = pd.read_csv('D:\ML_Comps\Bike_sharing_demand_pred\Original_data\TRAIN.csv')
test_data = pd.read_csv('D:\ML_Comps\Bike_sharing_demand_pred\Original_data\TEST.csv')
combined_data = pd.concat((train_data,test_data),axis=0)
train_len = len(train_data)
#%%
#Data prep:

del combined_data['Index'],combined_data['Date']

#encoding the categorical variabes:
combined_data=pd.get_dummies(combined_data,columns=['Seasons','Holiday','Functioning Day'],drop_first=True)

col_order=[ 'Hour', 'Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)',
       'Visibility (10m)', 'Dew point temperature(°C)',
       'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)', 'Seasons_Spring', 'Seasons_Summer',
       'Seasons_Winter', 'Holiday_No Holiday', 'Functioning Day_Yes','Rented Bike Count']
combined_data = combined_data[col_order]
#%%
#Utitility functions:
from sklearn.model_selection import cross_val_score
def evaluate(regressor,ind_vars,dep_vars,name):
    scores=cross_val_score(regressor,X=ind_vars,y=dep_vars,cv=10,n_jobs=-1,scoring='neg_root_mean_squared_error')
    print(f'\n{name}:')
    print(f'The mean rmse is {scores.mean()}')
    print(f'The std.deviaion is {scores.std()}')
#%%
#For linear regression:
# Linear_regression:
# The mean rmse is -459.98413762136335
# The std.deviaion is 151.13333078274482
# del combined_data['Visibility (10m)']

# ind_vars = combined_data.iloc[:train_len,:-1].values
# dep_vars = combined_data.iloc[:train_len,-1].values

# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()

# evaluate(regressor,ind_vars,dep_vars,'Linear_regression')
#%%
#Non-Linear Regression:
# GradBoostReg:
# The mean rmse is -245.73063440011484
# The std.deviaion is 97.29963146417222
# ExtraTreeReg:
# The mean rmse is -249.25570019721226
# The std.deviaion is 105.16360490413412
# AdaBoostReg:
# The mean rmse is -444.35672495302816
# The std.deviaion is 67.36587817100026
# XGBReg:
# The mean rmse is -262.7026644688744
# The std.deviaion is 105.0270663994271
# XGBRFReg:
# The mean rmse is -298.8890634543525
# The std.deviaion is 131.3796336857591

ind_train = combined_data.iloc[:train_len,:-1].values
dep_train = combined_data.iloc[:train_len,-1].values

# from sklearn.ensemble import \
#     RandomForestRegressor,GradientBoostingRegressor,ExtraTreesRegressor,AdaBoostRegressor
# from xgboost import XGBRegressor,XGBRFRegressor

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
#%%
#Model Tuning:
# n_estimators: 300, lr: 0.1:
# The mean rmse is -243.12644426633483
# The std.deviaion is 97.1798465522186

# from sklearn.ensemble import GradientBoostingRegressor
# est_list=[100,300,500,1000]
# learning_rates=[1,0.01,0.1,0.2,0.001]

# for est in est_list:
#     for lr in learning_rates:
#         regressor = GradientBoostingRegressor(n_estimators=est,learning_rate=lr)
#         ind_train = regressor.transform()
#         evaluate(regressor,ind_train,dep_train,f'n_estimators: {est}, lr: {lr}')
#%%
#Feature Selection:

# For first 9 variable(s):
# The mean rmse is -242.3579164205192
# The std.deviaion is 93.84947110549888

from sklearn.ensemble import GradientBoostingRegressor
regressor = GradientBoostingRegressor(n_estimators=300,random_state=0)
regressor.fit(ind_train,dep_train)

indices=np.array(list(range(0,len(ind_train[0]))),dtype=int)
indices=indices.reshape(-1,1)
scores=np.array(regressor.feature_importances_).ravel()
scores=scores.reshape(-1,1)
score_table=np.append(indices,scores,axis=1)
score_table=score_table[score_table[:,-1].argsort()[::-1]]

target_vars=list(map(int,score_table[:,0]))

while len(target_vars)!=0:
    regressor=GradientBoostingRegressor(n_estimators=300,random_state=0)
    evaluate(regressor,ind_train[:,target_vars],dep_train,f'For first {len(target_vars)} variable(s)')
    del target_vars[-1]