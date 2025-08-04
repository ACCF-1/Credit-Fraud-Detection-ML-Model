'''
XGBoost F2 score 0.595

good F2 score: >0.6
aim > 0.8


model audit trails


'''
''' Always compute moving averages AFTER splitting to prevent leakage.
✅ For time-series data, avoid shuffling before splitting.
✅ If using cross-validation, recompute MA per fold in a time-aware manner.'''


#In[0] Libraries
'''Import necessary libraries'''
# data
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None

# ML
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
import xgboost
from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn import metrics
import sklearn.pipeline as skl_pipe
import imblearn.pipeline as imb_pipe
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import model_selection
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit

from functools import partial

import dill

# graphs
import seaborn as sns
import matplotlib.pyplot as plt
import shap

# params
import sys
import os
if __name__ == "__main__":
    parent_dir = os.path.dirname(os.getcwd())
    sys.path.append(os.path.join(parent_dir, 'configurations'))
else:
    sys.path.append(os.path.join(os.getcwd(), 'configurations'))
    sys.path.append(os.path.join(os.getcwd(), 'ml_scripts'))
import config as cfg

# others
import utility_functions as uf
import specific_functions as sf
import importlib


#In[1] setup
'''quick test of model tuning and evaluation'''

def drop_model_irrelevant_cols(df, irrelevant_cols:list):
    df = df.drop(columns=irrelevant_cols, errors='ignore')
    return df

original_df = pd.read_csv('C:/Users/ccfan/Documents/GitHub/Imbalance-Classification-ML/data/processed/credit_fraud_data_transformed.csv') #_cleaned
original_df = original_df.sort_values('transaction_date')

cf_raw_df = pd.read_csv('C:/Users/ccfan/Documents/GitHub/Imbalance-Classification-ML/data/processed/credit_fraud_data_transformed.csv') #_cleaned
cf_raw_df = cf_raw_df.sort_values('transaction_date')

cf_raw_df = drop_model_irrelevant_cols(
    cf_raw_df, 
    [
    'transaction_date',
    'customer_id', 
    'amount',
    'date_in_sec', 
    'No._of_day', 
    'merchant_id', 
    'pos_id', 
    'merchant_established_date', 
    'MCCs',
    'merchant_category',
    'IP_address',
    'type_of_credit_card_used',
    'store_card_merchant_id',
    'IP_address_country_code', #FIXME
    'country_code_date',
    'type_of_credit_card_used_with_mer_id',
    'type_of_credit_card_used',
    ]
)

tgt = 'is_fraud'
feats = cf_raw_df.drop(columns=[tgt]).columns.tolist()

train_X = cf_raw_df[:int(len(cf_raw_df)*0.8)]
test_X = cf_raw_df[int(len(cf_raw_df)*0.8):]

train_X = train_X.drop(columns=['is_fraud'])
test_X = test_X.drop(columns=['is_fraud'])
train_y = cf_raw_df[:int(len(cf_raw_df)*0.8)]['is_fraud']
test_y = cf_raw_df[int(len(cf_raw_df)*0.8):]['is_fraud']

'''X = cf_raw_df[feats]
y = cf_raw_df[tgt]
X = X.drop(columns=['is_fraud'])'''






































#In[2] quick model test
#standardize the data
#normalize the data

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit() #n_splits=5

'''# 6. Train-Test Split (Stratify to preserve class distribution)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 7. Train the XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# 8. Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]'''


train_X = cf_raw_df[:int(len(cf_raw_df)*0.8)]
test_X = cf_raw_df[int(len(cf_raw_df)*0.8):]

train_X = train_X.drop(columns=['is_fraud'])
test_X = test_X.drop(columns=['is_fraud'])
train_y = cf_raw_df[:int(len(cf_raw_df)*0.8)]['is_fraud']
test_y = cf_raw_df[int(len(cf_raw_df)*0.8):]['is_fraud']



'''bl_mdl = DummyClassifier()
bl_mdl.fit(train_X, train_y)
naive_probs = bl_mdl.predict_proba(train_y)[:, 1]
precision, recall, _ = metrics.precision_recall_curve(test_y, naive_probs)
bl_pr_auc_score = metrics.auc(recall, precision)
print('No skill PR AUC: %.3F' % bl_pr_auc_score)'''


bl_mdl = DummyClassifier()
bl_mdl.fit(train_X, train_y)
bl_mdl.predict(test_X)
prediction = bl_mdl.predict(test_X)

dt_mdl = tree.DecisionTreeClassifier()
dt_mdl.fit(train_X, train_y)
prediction = dt_mdl.predict(test_X)

rf_mdl = ensemble.RandomForestClassifier()
rf_mdl.fit(train_X, train_y)
prediction = rf_mdl.predict(test_X)

xgb_mdl = XGBClassifier(scale_pos_weight=13)
xgb_mdl.fit(train_X, train_y)
prediction = xgb_mdl.predict(test_X) #0.6595182138660399


pd.DataFrame(metrics.confusion_matrix(test_y, prediction, labels=[1,0]), 
                                        columns=['Pred:1', 'Pred:2'], 
                                        index=['True:1', 'True:0'])

pd.DataFrame(metrics.classification_report(test_y, prediction, output_dict=True))

metrics.fbeta_score(test_y, prediction, average="binary", beta=2)






#In[9]
if __name__ == '__main__':
    cf_raw_df = pd.read_csv('C:/Users/ccfan/Documents/GitHub/Imbalance-Classification-ML/data/processed/credit_fraud_data_transformed.csv') #_cleaned
    tgt = 'is_fraud'