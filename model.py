#%%
#Importing the packages:
import pandas as pd
import numpy as np
#%%
#Importing the data:
train_data = pd.read_csv('D:\ML_Comps\Bike_sharing_demand_pred\Original_data\TRAIN.csv')
test_data = pd.read_csv('D:\ML_Comps\Bike_sharing_demand_pred\Original_data\TEST.csv')
combined_data = pd.concat((train_data,test_data),axis=0)
train_len=len(train_data)
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
#Function for processing the data:
def process_data(target):
    #Data cleaning:
    target['Day']=target['Date'].apply(lambda x:get_day(x))
    target['Month']=target['Date'].apply(lambda x:get_month(x))
    
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
#%%
#Processing the data:
combined_data = process_data(combined_data)
col_order = [ 'Hour', 'Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)',
       'Visibility (10m)', 'Dew point temperature(°C)',
       'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)', 'Seasons_Spring', 'Seasons_Summer',
       'Seasons_Winter', 'Holiday_No Holiday', 'Functioning Day_Yes',
       'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6', 'Month_7',
       'Month_8', 'Month_9','Month_10','Month_11', 'Month_12', 'Day_bin_1', 'Day_bin_2', 'Day_bin_3',
       'Day_bin_4', 'Day_bin_5', 'Day_bin_6','Rented Bike Count']
combined_data = combined_data[col_order]
#%%
#Splitting the data:
ind_train = combined_data.iloc[:train_len,:-1].values
dep_train = combined_data.iloc[:train_len,-1].values
ind_test = combined_data.iloc[train_len:,:-1].values
#%%
#Feature Scaling:
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
ind_train = scaler.fit_transform(ind_train)
ind_test = scaler.transform(ind_test)
#%%
#Filtering the variables(Based on the results obtained from analysis):
target_vars = [1, 0, 6, 2, 7, 13, 5, 18, 11, 3, 4, 17]
ind_train = ind_train[:,target_vars]
ind_test = ind_test[:,target_vars]
#%%
#Utilities:
def refine(x):
    return (int(x) if x > 0 else 0)
#%%
#Modelling and Prediction:
from sklearn.ensemble import GradientBoostingRegressor
regressor = GradientBoostingRegressor(n_estimators=500)
regressor.fit(ind_train,dep_train)

results = regressor.predict(ind_test)
results = np.array(list(map(refine,results)),dtype=int)
#%%
#Making submission file:
results = results.reshape(-1,1)
indices = test_data.iloc[:,0:1].values
result_table = np.append(indices,results,axis=1)
result_table = pd.DataFrame(result_table,columns=['Index','Rented Bike Count'])
result_table.to_csv('D:\ML_Comps\Bike_sharing_demand_pred\submissions\submission_1.csv',index=False)
# %%
abnormal_cols=[]
for i in range(len(results)):
    if results[i,0] == 0:
        abnormal_cols.append(i)
ab_data = test_data.iloc[abnormal_cols,:]
# %%
