import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold

data = pd.read_csv("AuctionData1.csv")

data.drop('Unnamed: 11', axis=1, inplace=True)  # remove null column named Unnamed: 12

list_drop = ['CHEST', 'INV #', 'LOT #', 'M / S', 'BUY/C', 'LimitPrice', 'ESTATECODE', 'WAREHOUSENAME']
data_new = data.drop(list_drop, axis=1)

print(data_new.head(10))

print(data_new.describe(include='all'))

print(data_new.tail(10))

print(data_new.isnull().sum())



# k-1 one hot encoing
leaf_standard = pd.get_dummies(data_new['LEAFStandard'], dummy_na=False, drop_first=True)
leaf_standard.tail(20)

infused_remarks = pd.get_dummies(data_new['INFUSEDRemarks'], dummy_na=False, drop_first=True)
infused_remarks.head(20)

liquor_remark = pd.get_dummies(data_new['LiquorRemark'], dummy_na=False, drop_first=True)
liquor_remark.head(20)

counts = data_new['BUYER'].value_counts()

less100Value = counts[counts < 100].index.tolist()

data_new.loc[data_new['BUYER'].isin(less100Value), 'BUYER'] = 'Other'

LB = LabelEncoder()
data_new["BUYER"] = LB.fit_transform(data_new["BUYER"])

data_new["SELLINGMARK"] = LB.fit_transform(data_new["SELLINGMARK"])

data_new["GRADE"] = LB.fit_transform(data_new["GRADE"])

data_new2 = pd.concat([data_new, leaf_standard, infused_remarks, liquor_remark], axis=1)

print(data_new2.head(10))

data_new2.drop(['LEAFStandard', 'INFUSEDRemarks', 'LiquorRemark'], axis=1, inplace=True)

# Adding missing values using Multiple imputation
cols_with_missing = [col for col in data_new2.columns if data_new2[col].isnull().any()]

cols_with_missing

imputer = IterativeImputer()

imputer.fit(data_new2)

imputed_data = imputer.transform(data_new2)

imputed_df = pd.DataFrame(np.round(imputed_data), columns=data_new2.columns)
# End of Multiple imputation

data_final = imputed_df.drop(['SELLINGMARK', 'NET WT.'], axis=1)
print(data_final)

print(data_final.dtypes)

features = data_final.drop('PRICE', axis=1)
label = data_final['PRICE']

# converting to array variables
features = np.array(features)
label = np.array(label)



X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=1)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

xgb_model = XGBRegressor(max_depth=7, n_estimators=100, predictor='auto', subsample=0.8, tree_method='hist')
xgb_model.fit(X_train, y_train)
print(xgb_model.score(X_test, y_test))

# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=25)
#
# param_dict = {
#     'max_depth': [1, 3, 5, 7],
#     'subsample': [0.2, 0.5, 0.8],
#     'n_estimators': [100, 300, 500],
#     'predictor': ['auto', 'cpu_predictor'],
#     'tree_method': ['hist']
# }
#
# gridSearch = GridSearchCV(estimator=XGBRegressor(),
#                           param_grid=param_dict,
#                           cv=cv,
#                           n_jobs=1,
#                           verbose=10)



# print(xgb_model.best_params_)

# print(xgb_model.best_score_)

# evaluate(gridSearch, X_test, y_test)

import joblib

joblib.dump(xgb_model, 'xgb_modelHP.pkl')
