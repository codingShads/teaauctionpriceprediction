import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import RandomOverSampler

from sklearn.ensemble import RandomForestClassifier

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score

import pickle

import joblib

data = pd.read_csv("AuctionData1.csv")
print(data.head(10))
print(data.dtypes)

data.drop('Unnamed: 11', axis=1, inplace=True)  # remove null column named Unnamed: 12
print(data.head(5))

# Removing column which are not important
list_drop = ['CHEST', 'INV #', 'LOT #', 'M / S', 'BUY/C', 'LimitPrice']
data_new = data.drop(list_drop, axis=1)

print(data_new.dtypes)

insert_mean = data_new.Valuation.mean()
data_new["Valuation"].fillna(insert_mean, inplace=True)

insert_mean = data_new.UpperValuation.mean()
data_new["UpperValuation"].fillna(insert_mean, inplace=True)

insert_mean = data_new.MinPrice.mean()
data_new["MinPrice"].fillna(insert_mean, inplace=True)

insert_mean = data_new.AskingPrice.mean()
data_new["AskingPrice"].fillna(insert_mean, inplace=True)

insert_mode = data_new['WAREHOUSENAME'].mode()[0]
data_new["WAREHOUSENAME"].fillna(insert_mode, inplace=True)

insert_mode = data_new['LEAFStandard'].mode()[0]
data_new["LEAFStandard"].fillna(insert_mode, inplace=True)

insert_mode = data_new['INFUSEDRemarks'].mode()[0]
data_new["INFUSEDRemarks"].fillna(insert_mode, inplace=True)

insert_mode = data_new['LiquorRemark'].mode()[0]
data_new["LiquorRemark"].fillna(insert_mode, inplace=True)

leaf_standard = pd.get_dummies(data_new['LEAFStandard'], drop_first=True)
print(leaf_standard.head(20))

infused_remarks = pd.get_dummies(data_new['INFUSEDRemarks'], drop_first=True)
infused_remarks.head(20)

liquor_remark = pd.get_dummies(data_new['LiquorRemark'], drop_first=True)
liquor_remark.head(20)

counts = data_new['BUYER'].value_counts()
less100Value = counts[counts < 100].index.tolist()
data_new.loc[data_new['BUYER'].isin(less100Value), 'BUYER'] = 'Other'

LB = LabelEncoder()
data_new["BUYER"] = LB.fit_transform(data_new["BUYER"])

insert_mean = data_new['BUYER'].mean()
data_new["BUYER"].fillna(insert_mean, inplace=True)

data_new["SELLINGMARK"] = LB.fit_transform(data_new["SELLINGMARK"])

data_new["GRADE"] = LB.fit_transform(data_new["GRADE"])

data_new2 = pd.concat([data_new, leaf_standard, infused_remarks, liquor_remark], axis=1)

data_new2.drop(['LEAFStandard', 'INFUSEDRemarks', 'LiquorRemark'], axis=1, inplace=True)

data_new3 = data_new2.drop(['WAREHOUSENAME', 'ESTATECODE', 'NET WT.'], axis=1)
data_new3

data_new3.dropna(inplace=True)

LB = LabelEncoder()
data_new3["Valuation"] = LB.fit_transform(data_new3["Valuation"])

LB = LabelEncoder()
data_new3["UpperValuation"] = LB.fit_transform(data_new3["UpperValuation"])

LB = LabelEncoder()
data_new3["MinPrice"] = LB.fit_transform(data_new3["MinPrice"])

LB = LabelEncoder()
data_new3["AskingPrice"] = LB.fit_transform(data_new3["AskingPrice"])

LB = LabelEncoder()
data_new3["PRICE"] = LB.fit_transform(data_new3["PRICE"])

print(data_new3.head())

print(data_new3.shape)

features = data_new3.drop('PRICE', axis=1)
label = data_new3['PRICE']

X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=30)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

print(len([i for i in y_train if i == 1]), len([i for i in y_train if i == 0]))

ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X_train, y_train)

print(len([i for i in y_res if i == 1]), len([i for i in y_res if i == 0]))

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

X_train = X_res
y_train = y_res

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average='micro')
    recall = recall_score(y_test, pred, average='micro')
    f1 = f1_score(y_test, pred, average='micro')
    # tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    cm = confusion_matrix(y_test, pred)
    tn, fp, fn, tp = cm.ravel()[:4]
    print("TN: {}   FP: {}\nFN: {}   TP: {}".format(tn, fp, fn, tp))
    print(classification_report(y_test, pred))
    print('Accuracy: %f' % accuracy)
    print('Precision: %f' % precision)
    print('Recall: %f' % recall)
    print('F1 score: %f' % f1)


# modelRF = RandomForestClassifier(n_estimators=50, criterion='gini', random_state=0)  # bagging model

le = LabelEncoder()
y_train = le.fit_transform(y_train)

# modelRF.fit(X_train, y_train)

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold

cv = RepeatedStratifiedKFold(n_splits=50, n_repeats=2, random_state=15)

param_dict = {
    'n_estimators': [10, 30, 50, 80, 90]
}

randomSearch = GridSearchCV(estimator=RandomForestClassifier(),
                            param_grid=param_dict,
                            scoring='accuracy',
                            cv=cv,
                            n_jobs=1,
                            verbose=10)

randomSearch.fit(X_train, y_train)
# pickle.dump(modelRF.fit, open('modelRF.pkl', 'wb'))

# joblib.dump(modelRF, 'RF2model.pkl')
