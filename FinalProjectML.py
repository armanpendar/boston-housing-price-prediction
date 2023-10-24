# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 00:47:49 2023

@author: Arman Pendar
@email : armanpendar@gmail.com
"""

import pandas as pd
import numpy as np
# report dataframe
# create an Empty DataFrame for save report parameters
report_df = pd.DataFrame(columns = ['Name', 'Best_score', 'Best_params'])

# set calculation parameters =====================================================
'''
Number of jobs to run in parallel.
None means 1 unless in a joblib.parallel_backend context. 
-1 means using all processors. See Glossary for more details.
'''
_n_jobs = 1

'''
Controls the verbosity: the higher, the more messages.
>1 : the computation time for each fold and parameter candidate is displayed;
>2 : the score is also displayed;
>3 : the fold and candidate parameter indexes are also displayed together with the starting time of the computation.
'''
_verbos = 3

#==================================================================================

# fetch data from web and preproces and clean data
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
boston_data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
#in data chanta chizi dakhelesh dare
#in ro bznid esme feature ha va vorodi haro mizane k chia hast
#print(boston_data.feature_names)
x=boston_data
y=target

#miayd migid cross validation chijori bashe, yani chijori taghsim b train test kone
from sklearn.model_selection import KFold
fold=KFold(n_splits=5,shuffle=True,random_state=0)

print('1================ LinearRegression')

#Modeleton ro entekhab mikonid --> inja niazi nist setting bnvisid
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# more details :  https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
#oon setting haee k mikhyad bere begarde beyeneshon behtrino vardare ro minvisid
myparams={
          'fit_intercept':[True],
          'copy_X' : [True],          
          'n_jobs':[-1],
          'positive':[False]
          }
 

#in tabeye zir az shoma model ro mifgire, miad az shoma modeletono mige bayad 
#ychizi bname param_grid yani kodom hyperparameter ha va dar ch rangi ro brm test konam
#mige chijori test dataseteto taghsim konm k shoma gfoti oon bala too fold nvshti
#hala mige chijori biad fasele ro hesab kone k too scoring goftin

   
from sklearn.model_selection import GridSearchCV
gs=GridSearchCV(model,param_grid=myparams,cv=fold,scoring='neg_mean_absolute_percentage_error',n_jobs= _n_jobs , verbose=_verbos)
    

#inja fit mikonid
gs.fit(x,y)

#Save parameters in report_df
report_dict = {}
report_dict.update({'Name': 'LinearRegression'})
report_dict.update({'Best_score': gs.best_score_})
report_dict.update({'Best_params': gs.best_params_})
report_df.loc[len(report_df)] = report_dict # only use with a RangeIndex!

# saving report dataframe to csv
report_df.to_csv('Project_Final_report(temp).csv') 

print('2================ KNeighborsRegressor')

# more details : https://scikit-learn.org/0.16/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
#Modeleton ro entekhab mikonid --> inja niazi nist setting bnvisid
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor()

#oon setting haee k mikhyad bere begarde beyeneshon behtrino vardare ro minvisid
myparams={
          'n_neighbors' : [1,2,3,4,5,6,7,8,9,10],
          'algorithm' : ['auto'],# 'ball_tree', 'kd_tree', 'brute'],
          'weights' : ['uniform', 'distance'],
          'leaf_size' : [5],#10,15,20,25,30,35,40,45],
          'metric' : ['minkowski'],#'manhattan', 'euclidean'],
          'p' : [1, 2],
          #'metric_params' : [] #default = None , additional keyword arguments for the metric function.
          }
 

#in tabeye zir az shoma model ro mifgire, miad az shoma modeletono mige bayad 
#ychizi bname param_grid yani kodom hyperparameter ha va dar ch rangi ro brm test konam
#mige chijori test dataseteto taghsim konm k shoma gfoti oon bala too fold nvshti
#hala mige chijori biad fasele ro hesab kone k too scoring goftin

   
from sklearn.model_selection import GridSearchCV
gs=GridSearchCV(model,param_grid=myparams,cv=fold,scoring='neg_mean_absolute_percentage_error',n_jobs= _n_jobs , verbose=_verbos)
    

#inja fit mikonid
gs.fit(x,y)

#Save parameters in report_df
report_dict = {}
report_dict.update({'Name': 'KNeighborsRegressor'})
report_dict.update({'Best_score': gs.best_score_})
report_dict.update({'Best_params': gs.best_params_})
report_df.loc[len(report_df)] = report_dict # only use with a RangeIndex!



#dar asl shoma fek konid 5 ta data darid yani 5 ta x 5 ta y
#in maid 5 ta laptab dre
#yekish miad mige x1 ro mzirim test x2,3,4,5 ro mizrim train
#rooye train yad migire , ba 10 ta hyperparameter yani masalan k ro az 1 ta 10 mzire

#bad maid vase dovomin laptab miad x2 ro mizare test va rooye x1,3,4,5 yad migire 
#bad haminjori k ro az 1 ta 10 mizare

#bad dar akahr baraye harkodom az K (yani hyperparameter hamon) 5 ta test score dre
#ke miangine mishe cross_score --> harkodom paeen tar bashe yani oon hyperparameetr
#Behatrinnnneee
# saving report dataframe to csv
report_df.to_csv('Project_Final_report(temp).csv') 

print('3================ DecisionTreeRegressor')

#Modeleton ro entekhab mikonid --> inja niazi nist setting bnvisid
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()

# more details : https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
#oon setting haee k mikhyad bere begarde beyeneshon behtrino vardare ro minvisid
myparams={
          'max_depth':[5,10],#15,20,25,30],
          'min_samples_leaf':[2],#4,6,8,10],
          'min_samples_split':[2,4],#6,8,10],          
          #'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
          'splitter':['best','random'],
          #'max_features':['auto', 'sqrt', 'log2'],
          #'max_leaf_nodes':[2,4,6,8,10],           
          #'min_weight_fraction_leaf':[], 
          #'random_state':[], 
          #'min_impurity_decrease':[], 
          #'ccp_alpha' : []          
          }
 

#in tabeye zir az shoma model ro mifgire, miad az shoma modeletono mige bayad 
#ychizi bname param_grid yani kodom hyperparameter ha va dar ch rangi ro brm test konam
#mige chijori test dataseteto taghsim konm k shoma gfoti oon bala too fold nvshti
#hala mige chijori biad fasele ro hesab kone k too scoring goftin

   
from sklearn.model_selection import GridSearchCV
gs=GridSearchCV(model,param_grid=myparams,cv=fold,scoring='neg_mean_absolute_percentage_error',n_jobs= _n_jobs , verbose=_verbos)
    

#inja fit mikonid
gs.fit(x,y)

#Save parameters in report_df
report_dict = {}
report_dict.update({'Name': 'DecisionTreeRegressor'})
report_dict.update({'Best_score': gs.best_score_})
report_dict.update({'Best_params': gs.best_params_})
report_df.loc[len(report_df)] = report_dict # only use with a RangeIndex!


# saving report dataframe to csv
report_df.to_csv('Project_Final_report(temp).csv') 
# Start Algorithms

print('4================ RandomForestRegressor')

#Modeleton ro entekhab mikonid --> inja niazi nist setting bnvisid
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()

# more details : https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
#oon setting haee k mikhyad bere begarde beyeneshon behtrino vardare ro minvisid
myparams={
          'n_estimators':[100], # The number of trees in the forest.
          'max_depth':[5,10,15],#20,25,30],
          'min_samples_leaf':[2],#4,6,8,10],
          'min_samples_split':[2,4],#6,8,10],          
          #'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
          #'bootstrap' : [],
          #'ccp_alpha' : [],
          #'max_features':[],
          #'max_leaf_nodes':[], 
          #'max_samples':[], 
          #'min_impurity_decrease':[], 
          #'min_weight_fraction_leaf':[], 
          #'random_state':[],         
          #'n_jobs':[],
          #'oob_score':[],
          #'verbose':[],
          #'warm_start':[]
          }
 

#in tabeye zir az shoma model ro mifgire, miad az shoma modeletono mige bayad 
#ychizi bname param_grid yani kodom hyperparameter ha va dar ch rangi ro brm test konam
#mige chijori test dataseteto taghsim konm k shoma gfoti oon bala too fold nvshti
#hala mige chijori biad fasele ro hesab kone k too scoring goftin

   
from sklearn.model_selection import GridSearchCV
gs=GridSearchCV(model,param_grid=myparams,cv=fold,scoring='neg_mean_absolute_percentage_error',n_jobs= _n_jobs , verbose=_verbos)
    

#inja fit mikonid
gs.fit(x,y)

#Save parameters in report_df
report_dict = {}
report_dict.update({'Name': 'RandomForestRegressor'})
report_dict.update({'Best_score': gs.best_score_})
report_dict.update({'Best_params': gs.best_params_})
report_df.loc[len(report_df)] = report_dict # only use with a RangeIndex!





# saving report dataframe to csv
report_df.to_csv('Project_Final_report(temp).csv')      
  
print('5================ SVR')

#Modeleton ro entekhab mikonid --> inja niazi nist setting bnvisid
from sklearn.svm import SVR
model = SVR()

# more details : https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
#oon setting haee k mikhyad bere begarde beyeneshon behtrino vardare ro minvisid
myparams={
          'kernel':['rbf'],#,'linear', 'poly', 'sigmoid', 'precomputed','callable'],
          #'degree':[2,3,4,5],
          'C' : [1000],#0.0001,0.001,0.01,0.1,10,100,1000],
          'gamma':[0.0001],#0.0001 ,0.001,0.01,0.1,10,100,1000],#'scale', 'auto'],
          'max_iter':[-1],
          #'class_weight':[],            
          'epsilon':[0.2],#0.1,0.2],
          #'intercept_scaling':[], 
          #'coef0':[0],
          'shrinking':[True], 
          'cache_size':[200], 
          'tol':[0.001],
          #'verbose':[False]
          }
 

#in tabeye zir az shoma model ro mifgire, miad az shoma modeletono mige bayad 
#ychizi bname param_grid yani kodom hyperparameter ha va dar ch rangi ro brm test konam
#mige chijori test dataseteto taghsim konm k shoma gfoti oon bala too fold nvshti
#hala mige chijori biad fasele ro hesab kone k too scoring goftin

   
from sklearn.model_selection import GridSearchCV
gs=GridSearchCV(model,param_grid=myparams,cv=fold,scoring='neg_mean_absolute_percentage_error',n_jobs= _n_jobs , verbose=_verbos)
    

#inja fit mikonid
gs.fit(x,y)

#Save parameters in report_df
report_dict = {}
report_dict.update({'Name': 'SVR'})
report_dict.update({'Best_score': gs.best_score_})
report_dict.update({'Best_params': gs.best_params_})
report_df.loc[len(report_df)] = report_dict # only use with a RangeIndex!



# saving report dataframe to csv
report_df.to_csv('Project_Final_report(temp).csv')     

print('6================ MLPRegressor')

#Modeleton ro entekhab mikonid --> inja niazi nist setting bnvisid
from sklearn.neural_network import MLPRegressor
model=MLPRegressor()

# more details : https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
#oon setting haee k mikhyad bere begarde beyeneshon behtrino vardare ro minvisid
myparams={
          'hidden_layer_sizes'  : [(20,)],#(10,10),(20,),(20,20)] ,
          'activation':['relu'],#'tanh' ],#,'identity'],
          'solver':['lbfgs'],#'adam','sgd'],          
          'alpha':[0.0001],
          'batch_size':['auto'],
          'learning_rate':['constant'],#'invscaling','adaptive'],
          'learning_rate_init':[0.001],
          'power_t':[0.5],
          'max_iter':[9000000],
          'shuffle':[True],
          'random_state':[0],
          'tol':[0.0001],
          'verbose':[False],          
          'warm_start':[True],
          'momentum':[0.9],
          'nesterovs_momentum':[True],
          'early_stopping':[True],
          'validation_fraction':[0.1],
          'beta_1':[0.9],
          'beta_2':[0.999],
          'epsilon':[0.00000001],
          'n_iter_no_change':[10],
          'max_fun':[15000]
          }
 

#in tabeye zir az shoma model ro mifgire, miad az shoma modeletono mige bayad 
#ychizi bname param_grid yani kodom hyperparameter ha va dar ch rangi ro brm test konam
#mige chijori test dataseteto taghsim konm k shoma gfoti oon bala too fold nvshti
#hala mige chijori biad fasele ro hesab kone k too scoring goftin

   
from sklearn.model_selection import GridSearchCV
gs=GridSearchCV(model,param_grid=myparams,cv=fold,scoring='neg_mean_absolute_percentage_error',n_jobs= _n_jobs , verbose=_verbos)
    

#inja fit mikonid
gs.fit(x,y)

#Save parameters in report_df
report_dict = {}
report_dict.update({'Name': 'MLPRegressor'})
report_dict.update({'Best_score': gs.best_score_})
report_dict.update({'Best_params': gs.best_params_})
report_df.loc[len(report_df)] = report_dict # only use with a RangeIndex!  

  
#=====================================================================
print('================ Show and Save Report')
#=====================================================================
# Sort report df
report_df = report_df.sort_values(['Best_score'], ascending=[False])
# Show report df
print(report_df[['Name', 'Best_score']])

# saving report dataframe to csv
import datetime
# using now() to get current time
current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
file_name = 'Project_Final_report-'+current_time+'.csv'
report_df.to_csv(file_name)

#===============================================
print('================ Show Plot')
import matplotlib.pyplot as plt
print("Making Report Diagram ...")
#Modeleton ro entekhab mikonid --> inja niazi nist setting bnvisid
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
# more details : https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
#oon setting haee k mikhyad bere begarde beyeneshon behtrino vardare ro minvisid
myparams={
          'n_estimators':[100], # The number of trees in the forest.
          'max_depth':[5,10,15],#20,25,30],
          'min_samples_leaf':[2],#4,6,8,10],
          'min_samples_split':[2,4],#6,8,10],          
          #'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
          #'bootstrap' : [],
          #'ccp_alpha' : [],
          #'max_features':[],
          #'max_leaf_nodes':[], 
          #'max_samples':[], 
          #'min_impurity_decrease':[], 
          #'min_weight_fraction_leaf':[], 
          #'random_state':[],         
          #'n_jobs':[],
          #'oob_score':[],
          #'verbose':[],
          #'warm_start':[]
          } 
from sklearn.model_selection import GridSearchCV
gs=GridSearchCV(model,param_grid=myparams,cv=fold,scoring='neg_mean_absolute_percentage_error',n_jobs= _n_jobs , verbose=1)
    

#inja fit mikonid
gs.fit(x,y)
preddcited_data = gs.predict(x)
#font
font1 = {'family':'serif','color':'blue','size':12}
font2 = {'family':'serif','color':'darkred','size':8}
plt.grid(color = 'green', linestyle = '--', linewidth = 0.3)
plt.title("Boston House Price Prediction by Arman Pendar", fontdict = font1)
plt.xlabel("Index", fontdict = font2)
plt.ylabel("Price", fontdict = font2)

x_shomare_nemone = list(range(0,len(y)))

#plotting data
plt.plot(x_shomare_nemone, y, label="Real Data")
plt.plot(x_shomare_nemone, preddcited_data, label="RandomForestRegressor Prediction")

# placing legend outside plot
#plt.legend(bbox_to_anchor=(1.50, 1.0), loc='center')
plt.legend(loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5)
)
# showing plot
plt.show()
plt.show()

#=================================================
# End
print("All Tasks Done !")
print('See more details in "',file_name,'", file !')

