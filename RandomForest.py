import matplotlib
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns

from sklearn.preprocessing import MinMaxScaler

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

data = pd.read_csv("AuctionData1.csv")

data.drop('Unnamed: 11', axis=1, inplace=True)  # remove null column named Unnamed: 12

list_drop = ['CHEST', 'INV #', 'LOT #', 'M / S', 'BUY/C', 'LimitPrice', 'ESTATECODE', 'WAREHOUSENAME', 'LEAFStandard', 'INFUSEDRemarks', 'LiquorRemark']
data_new = data.drop(list_drop, axis=1)

# leaf_standard = pd.get_dummies(data_new['LEAFStandard'], drop_first=True)
# leaf_standard.tail(20)
#
# infused_remarks = pd.get_dummies(data_new['INFUSEDRemarks'], drop_first=True)
# infused_remarks.head(20)
#
# liquor_remark = pd.get_dummies(data_new['LiquorRemark'], drop_first=True)
# liquor_remark.head(20)

counts = data_new['BUYER'].value_counts()

less100Value = counts[counts < 100].index.tolist()

data_new.loc[data_new['BUYER'].isin(less100Value), 'BUYER'] = 'Other'

LB = LabelEncoder()
data_new["BUYER"] = LB.fit_transform(data_new["BUYER"])

data_new["SELLINGMARK"] = LB.fit_transform(data_new["SELLINGMARK"])

data_new["GRADE"] = LB.fit_transform(data_new["GRADE"])

# data_new2 = pd.concat([data_new, leaf_standard, infused_remarks, liquor_remark], axis=1)

# data_new2.drop(['LEAFStandard', 'INFUSEDRemarks', 'LiquorRemark'], axis=1, inplace=True)

cols_with_missing = [col for col in data_new.columns if data_new[col].isnull().any()]

imputer = IterativeImputer()

imputer.fit(data_new)

imputed_data = imputer.transform(data_new)

imputed_df = pd.DataFrame(np.round(imputed_data), columns=data_new.columns)

data_final = imputed_df.drop(['SELLINGMARK', 'NET WT.'], axis=1)
print(data_final.tail(50))


features = data_final.drop('PRICE', axis=1)
label = data_final['PRICE']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=30)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

scaler = MinMaxScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test, columns=X_test.columns)

print(X_train_scaled.head(10))





# from imblearn.over_sampling import RandomOverSampler
#
# ros = RandomOverSampler(random_state=42)
# X_res, y_res = ros.fit_resample(X_train, y_train)
#
# X_train = X_res
# y_train = y_res
#
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

from sklearn.ensemble import RandomForestClassifier

modelRF = RandomForestClassifier()

modelRF.fit(X_train_scaled, y_train)

y_pred_RF = modelRF.predict(X_test_scaled)

from sklearn.metrics import accuracy_score

print("RF: ", accuracy_score(y_test, y_pred_RF) * 100)

# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import RepeatedStratifiedKFold
#
# cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=15)
#
# param_dict = {
#     'n_estimators': [200, 500, 700, 900],
#     'max_depth': [4, 5, 6, 8],
#     'max_features': ['sqrt', 'log2'],
#     'criterion': ['gini', 'entropy']
# }
#
# randomSearch = GridSearchCV(estimator=RandomForestClassifier(),
#                             param_grid=param_dict,
#                             scoring='accuracy',
#                             cv=5,
#                             n_jobs=1,
#                             verbose=10)
#
# randomSearch.fit(X_train, y_train)
#
# y_pred_RF = randomSearch.predict(X_test)
# print("RF: ", accuracy_score(y_test, y_pred_RF))
#
# import joblib
# joblib.dump(modelRF, 'modelRF_ideHP.pkl')
