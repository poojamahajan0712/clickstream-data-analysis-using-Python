# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 11:47:30 2019

@author: Pooja.Mahajan
"""
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost 
from sklearn.ensemble import RandomForestClassifier

df=pd.read_csv("data.csv")

####getting count of missing values
df.isnull().sum(axis=0)


###Target variable distribution
df['dvJun13'].value_counts()

'''Out[269]: 
0    8664
1    5807'''

###missing values  - percentage of missing values 

missing_values=pd.DataFrame(((df.isnull().sum(axis=0))/df.shape[0])*100)
missing_values.reset_index(level=0,inplace=True)
missing_values.sort_values(by=[0],ascending=False,inplace=True)
##types of columns 
df.dtypes.value_counts()
'''Out[284]: 
float64    24
int64      11
object     10'''

###1.Perform imputations for missing data as needed -- basic imputation - median for numerical variables and "None" for categorical
numeric_variables = list(df.select_dtypes(include=['int64', 'float64']).columns.values)
df[numeric_variables] = df[numeric_variables].apply(lambda x: x.fillna(x.median()),axis=0)

categorial_variables = list(df.select_dtypes(exclude=['int64', 'float64',]).columns.values)
df[categorial_variables] = df[categorial_variables].apply(lambda x: x.fillna("None"),axis=0)





#####3.Check for correlations amongst variables- checking with Target Variable 
correlations = pd.DataFrame(df.corr()['dvJun13'].sort_values())
correlations.reset_index(level=0,inplace=True)



#########creating dummy variable ###########
df = pd.get_dummies(df)

####Break the population into test(random 75%) and val(25%)
y=df.dvJun13
X=df.drop(['dvJun13'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

test_label=X_test['user_id'].reset_index()
test_label.drop('index',axis=1,inplace=True)

X_test=X_test.drop(['user_id'],axis=1)
X_train=X_train.drop(['user_id'],axis=1)

###4.Run each of the below models and attach probaility score to a user_id for the val sample.
'''a.Logistic regression
b.Random Forest
c.XGBoost
d.SVM'''
####Logistic Regression

lr = LogisticRegression()
lr.fit(X_train,y_train)
predictions_lr =pd.Series(lr.predict(X_test)).reset_index()
predictions_lr.drop('index',axis=1,inplace=True)

lr_res=pd.concat([test_label,predictions_lr],axis=1)
lr_res.rename(columns={0:'Log_reg'},inplace=True)
print(classification_report(y_test,predictions_lr))
#print(confusion_matrix(y_test,predictions_lr))
#print("Roc AUC: ", roc_auc_score(y_test, predictions_lr))
######################

rf=RandomForestClassifier()
rf.fit(X_train,y_train)
rf.predict(X_test)

confusion_matrix(y_test,rf.predict(X_test))
d1=pd.DataFrame(rf.feature_importances_,index=X_train.columns,columns=['importance'])
d1.sort_values(by='importance',ascending=False)
###################SVM

model = SVC()
model.fit(X_train,y_train)
predictions_svm =pd.Series(model.predict(X_test)).reset_index()
predictions_svm.drop('index',axis=1,inplace=True)
svm_res=pd.concat([test_label,predictions_svm],axis=1)
svm_res.rename(columns={0:'svm'},inplace=True)
print(classification_report(y_test,predictions_svm))
#print("Roc AUC: ", roc_auc_score(y_test, predictions_svm))

##############xgboost 

params = {'learning_rate': [0.01,0.5,0.1],
            'n_estimators': [200,500,600],
            'max_depth': [5,10,12]}
    
xgb = xgboost.XGBClassifier(objective='binary:logistic', silent=False)
gsearch1 =GridSearchCV(estimator = xgb, param_grid = params, scoring='roc_auc', cv=3,refit = True)
gsearch1.fit(X_train,y_train)

predictions_xgb= pd.Series(gsearch1.predict(X_test)).reset_index()
predictions_xgb.drop('index',axis=1,inplace=True)
xgb_res=pd.concat([test_label,predictions_xgb],axis=1)
xgb_res.rename(columns={0:'xgb'},inplace=True)
print(classification_report(y_test,predictions_xgb))
#print("Roc AUC: ", roc_auc_score(y_test, predictions_xgb))

#best_est = gsearch1.best_estimator_
#print(best_est)


##########Random Forest

rf=RandomForestClassifier()
#rf with Hyperparameter tuning - Grid Search##
params = {'n_estimators': [100,200,500],
            'max_depth': [5,10,15],
            'max_features': ['auto', 'sqrt'],}
    
rf = RandomForestClassifier()
gsearch2 =GridSearchCV(estimator = rf, param_grid = params, scoring='roc_auc', cv=5,refit = True)
gsearch2.fit(X_train,y_train)

predictions_rf= pd.Series(gsearch2.predict(X_test)).reset_index()
predictions_rf.drop('index',axis=1,inplace=True)
rf_res=pd.concat([test_label,predictions_rf],axis=1)
rf_res.rename(columns={0:'rf'},inplace=True)
print(classification_report(y_test,predictions_rf))
#print("Roc AUC: ", roc_auc_score(y_test, predictions_rf))


##########ensemble####################
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))
# create the ensemble model
ensemble = VotingClassifier(estimators)
ensemble.fit(X_train,y_train)

predictions_ens= pd.Series(ensemble.predict(X_test)).reset_index()
predictions_ens.drop('index',axis=1,inplace=True)
ens_res=pd.concat([test_label,predictions_ens],axis=1)
ens_res.rename(columns={0:'ens'},inplace=True)
print(classification_report(y_test,predictions_ens))
#print("Roc AUC: ", roc_auc_score(y_test, predictions_ens))


res=pd.merge(lr_res,rf_res,on='user_id',how="left")
res1=pd.merge(svm_res,ens_res,on='user_id',how="left")
fin=pd.merge(res,xgb_res,on='user_id',how="left")
final1=pd.merge(fin,res1,on='user_id',how="left")

final1.to_csv("val_results.csv",index=False)
