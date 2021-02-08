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
#Processing the data:

del combined_data['Index'],combined_data['Date']

#encoding the categorical variabes:
combined_data=pd.get_dummies(combined_data,columns=['Seasons','Holiday','Functioning Day'],drop_first=True)

col_order=[ 'Hour', 'Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)',
       'Visibility (10m)', 'Dew point temperature(°C)',
       'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)', 'Seasons_Spring', 'Seasons_Summer',
       'Seasons_Winter', 'Holiday_No Holiday', 'Functioning Day_Yes','Rented Bike Count']
combined_data = combined_data[col_order]
#%%
#Data segregation:
ind_vars = combined_data.iloc[:,:-1].values
dep_train = combined_data.iloc[:train_len,-1].values
#%%
#Filtering the variables(Based on the results obtained from analysis):
# target_vars = [1, 0, 6, 2, 7, 13, 5, 11, 4]
ind_train = ind_vars[:train_len,:]
ind_test = ind_vars[train_len:,:]
#%%
#Utilities:
def refine(x):
    return (int(x) if x > 0 else 0)
#%%
#Modelling and Prediction:
from sklearn.ensemble import GradientBoostingRegressor
regressor = GradientBoostingRegressor(n_estimators=300,random_state=0)
regressor.fit(ind_train,dep_train)

results = regressor.predict(ind_test)
results = np.array(list(map(refine,results)),dtype=int)
#%%
#Making submission file:
results = results.reshape(-1,1)
indices = test_data.iloc[:,0:1].values
result_table = np.append(indices,results,axis=1)
result_table = pd.DataFrame(result_table,columns=['Index','Rented Bike Count'])
result_table.to_csv('D:\ML_Comps\Bike_sharing_demand_pred\submissions\submission_4.csv',index=False)
# %%
