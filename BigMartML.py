# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 20:35:21 2019

@author: Talha.Iftikhar
"""
##Importing Libraries##
import pandas as pd
import numpy as np
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder
from sklearn import cross_validation, metrics
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
################################
##Get data##
train=pd.read_csv('C:\\Users\\Talha.Iftikhar\\Downloads\\BigMartTrain.csv')
test=pd.read_csv('C:\\Users\\Talha.Iftikhar\\Downloads\\BigMartTest.csv')

train['source']='train'## Add a source column for final dataset
test['source']='test'

data = pd.concat([train, test],ignore_index=True)## concating both data sets, we will seperate those later
print(train.shape,test.shape,data.shape)


##Check for missing values
data.apply(lambda x: sum(x.isnull()))## 

data.describe()

## Distinct Values
data.apply(lambda x: len(x.unique()))

categorical_cols=[x for x in data.dtypes.index if data.dtypes[x]=='object']## Seperate Categorical Columns
categorical_cols = [x for x in categorical_cols if x not in ['Item_Identifier','Outlet_Identifier','source']] ##Removing UnNeeded Columns
#Print frequency of categories
for col in categorical_cols:
    print ('\nFrequency of Categories for varible %s'%col)
    print (data[col].value_counts())
    
## Filling Missing data##
# Item Weight contains null rows
item_avg_weight=data.pivot_table(values='Item_Weight',index='Item_Identifier') ## Getting Avg weight for filling missing values
miss_bool = data['Item_Weight'].isnull()
data.loc[miss_bool,'Item_Weight']  = data.loc[miss_bool,'Item_Identifier'].apply(lambda x: item_avg_weight.at[x,'Item_Weight'])## Putting Avg where is null
print ('#missing: %d'% sum(data['Item_Weight'].isnull()))

#Outlet size contains null rows
outlet_size_mode = data.pivot_table(values='Outlet_Size', columns='Outlet_Type',aggfunc=(lambda x:mode(x,axis=None)))
data.head()

data['Outlet_Size']=data['Outlet_Size'].fillna(data['Outlet_Size'].mode()[0], inplace=False)

data.pivot_table(values='Item_Outlet_Sales',index='Outlet_Type')

### Feature  Engineering : Adding New Features###

# Filling item visibility '0' values with mean
item_avg_visibility=data.pivot_table(values='Item_Visibility',index='Item_Identifier')
miss_bool = (data['Item_Visibility']==0)
data.loc[miss_bool,'Item_Visibility']  = data.loc[miss_bool,'Item_Identifier'].apply(lambda x: item_avg_visibility.at[x,'Item_Visibility'])

print ('Number of 0 values after modification: %d'%sum(data['Item_Visibility'] == 0))

#Seperate new feature for Mean Ratio of Visibility 
data['Item_Visibility_MeanRatio'] = data.apply(lambda x: x['Item_Visibility']/item_avg_visibility.at[x['Item_Identifier'],'Item_Visibility'], axis=1)

print (data['Item_Visibility_MeanRatio'].describe())
data.head()

##Categorize Item_type##

data['Item_type_combined']=data['Item_Identifier'].apply(lambda x: x[0:2])
data['Item_type_combined'] =data['Item_type_combined'].map({'FD':'Food',
                                         'NC':'Non-Consumable',
                                                             'DR':'Drinks'})
## Just a backup
data2=data
#Mapping rows to diff values#
data['Item_Fat_Content'].unique() = data['Item_Fat_Content'].replace({'LF':'Low Fat',
                                                             'reg':'Regular',
                                                             'low fat':'Low Fat'})
data.loc[data['Item_type_combined']=="Non-Consumable",'Item_Fat_Content'] = "Non-Edible"

##SKlearn ML##
## Converting Cat to Numerical type for SK Learn train
le = LabelEncoder()

data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])

var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_type_combined','Outlet_Type','Outlet']
for i in var_mod:
    data[i] = le.fit_transform(data[i])

data.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)

train = data.loc[data['source']=="train"]
test = data.loc[data['source']=="test"]
test.describe()

test.drop(['Item_Outlet_Sales','source'],axis=1,inplace=True)
train.drop(['source'],axis=1,inplace=True)

## ML Begins
# Base line algo, just to check how things look
mean_sales = train['Item_Outlet_Sales'].mean()
base1 = test[['Item_Identifier','Outlet_Identifier']]

base1['Item_Outlet_Sales'] = mean_sales

#Export submission file
base1.to_csv("alg0.csv",index=False)


## Function for each ML Algo, We can put multiple Algo using same code##
#Define target and ID columns:
target = 'Item_Outlet_Sales'
IDcol = ['Item_Identifier','Outlet_Identifier']

def modelfit(alg, dtrain, dtest, predictors, target, IDcol, filename):
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target])
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])

    #Perform cross-validation:
    cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], dtrain[target], cv=20, scoring='mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))
    
    #Print model report:
    print ("\nModel Report")
    print ("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(dtrain[target].values, dtrain_predictions)))
    print ("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
    
    #Predict on testing data:
    dtest[target] = alg.predict(dtest[predictors])
    
    #Export submission file:
    IDcol.append(target)
    submission = pd.DataFrame({ x: dtest[x] for x in IDcol})
    submission.to_csv(filename, index=False)
    
##Linear Regression##
predictors=[x for x in train.columns if x not in [target]+IDcol]
alg=LinearRegression(alpha=0.05,normalize=True)
modelfit(alg, train, test, predictors, target, IDcol, 'alg1.csv')
coef1 = pd.Series(alg.coef_, predictors).sort_values()
coef1.plot(kind='bar', title='Model Coefficients')
##Decision tree##

predictors = [x for x in train.columns if x not in [target]+IDcol]
alg = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
modelfit(alg, train, test, predictors, target, IDcol, 'alg2.csv')
coef3 = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
coef3.plot(kind='bar', title='Feature Importances')

##Random forest##

predictors = [x for x in train.columns if x not in [target]+IDcol]
alg5 = RandomForestRegressor(n_estimators=200,max_depth=5, min_samples_leaf=100,n_jobs=4)
modelfit(alg5, train, test, predictors, target, IDcol, 'alg5.csv')
coef5 = pd.Series(alg5.feature_importances_, predictors).sort_values(ascending=False)
coef5.plot(kind='bar', title='Feature Importances')