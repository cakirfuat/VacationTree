
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
import warnings
import pydotplus

from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier, export_graphviz

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV, cross_validate
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
import random
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.2f" % x)
warnings.simplefilter(action="ignore")
warnings.filterwarnings('ignore')

df = pd.read_excel(".venv/Data/Tatil Tercihleri ve İlgi Alanları (1).xlsx", sheet_name="Veri")
df.head(30)

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    #dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

outlier_thresholds(df, "YIL_TATİL_GÜN")
replace_with_thresholds(df, "YIL_TATİL_GÜN")


cat_cols = [col for col in df.columns if df[col].dtypes == "O"]
label_encoder = LabelEncoder()

for col in cat_cols:
    df[col] = label_encoder.fit_transform(df[col])
df["GELİR"] = df["GELİR"] + 1
df["TATİL_BÜTÇE"] = df["TATİL_BÜTÇE"] + 1
df["GELİR_HARCAMA"] = df["GELİR"] * df["TATİL_BÜTÇE"]

df.head(5)

y = df["AAP"]
X = df.drop("AAP", axis=1)
# Verileri eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
##################################################
# LightGBM Regressor #
##################################################

lgm_model = LGBMRegressor( verbosity=-1)

lgm_model.fit(X_train, y_train)
y_pred = lgm_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}") #0.55
r2 = r2_score(y_test, y_pred)
print(f"R-Squared: {r2}") #-0.11


# Grid Search için denenecek hiperparametrelerin listesi
param_grid = {
    'learning_rate': [0.001],  # Öğrenme oranı
    'n_estimators': [500],
    'colsample_bytree': [0.4, 0.5, 6],
    'feature_fraction': [0.8],
    'num_leaves': [15,20,25],  # Maksimum yaprak sayısı
    'max_depth': [8,12],  # Maksimum derinlik (-1: sınırsız)
    'min_child_samples': [20,22]  # Minimum çocuk örnek sayısı
}

lgm_grid_search = GridSearchCV(lgm_model,
                           param_grid,
                           cv=5,
                           verbose=1)

final_lgm_model = lgm_grid_search.fit(X, y)

# En iyi parametreleri ve skorları gösterelim
print("En iyi parametreler:", dt_best_grid.best_params_)
print("En iyi model:", dt_best_grid.best_estimator_)

parametreler = {'colsample_bytree': 0.5, 'learning_rate': 0.001, 'max_depth': 10, 'min_child_samples': 20, 'n_estimators': 500, 'num_leaves': 15}
#En iyi skor (MSE): 0.45096443352274707
model_lgbm = LGBMRegressor(**parametreler).fit(X_train,y_train)
y_pred = model_lgbm.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Test Verileri MSE: {mse}") # 0.487


import numpy as np

import pandas as pd

# Veri setinizin DataFrame olarak yüklenmesi
df = pd.read_csv('veri.csv')

# Mevcut veri setinin kopyasını oluşturma ve birleştirme
new_data = df.sample(frac=0.5, replace=True)  # Yüzde 50 oranında rastgele örnekler seçme
df = pd.concat([df, new_data], ignore_index=True)  # Yeni veri setini mevcut veri setine eklemek
df.describe()
