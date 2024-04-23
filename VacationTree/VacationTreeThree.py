
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

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.2f" % x)
warnings.simplefilter(action="ignore")

df = pd.read_excel(".venv/Data/Tatil Tercihleri ve İlgi Alanları (1).xlsx", sheet_name="Veri")
df.head(30)

def veri_tanıtım(dataframe):

    print("* İLK 10 GÖZLEM *")
    print("--------------------------------------------------------------------------")
    print(dataframe.head(10))

    print("--------------------------------------------------------------------------")
    print("* DEĞİŞKEN İSİMLERİ *")
    print("--------------------------------------------------------------------------")
    for i in dataframe.columns:
        print(i , "\n")
    print("--------------------------------------------------------------------------")
    print("* BETİMSEL İSTATİSTİK *")
    print("--------------------------------------------------------------------------")
    print(dataframe.describe().T)

    print("--------------------------------------------------------------------------")
    print("* KAYIP,BOŞ GÖZLEM *")
    print("--------------------------------------------------------------------------")
    print(dataframe.isnull().sum())

    print("--------------------------------------------------------------------------")
    print("* DEĞİŞKEN TİPLERİ *")
    print("--------------------------------------------------------------------------")
    print(dataframe.info())

    print("--------------------------------------------------------------------------")
    print("* VERİ BOYUTU *")
    print("--------------------------------------------------------------------------")
    print(dataframe.shape)
    print("Gözlem Birimi :" , dataframe.shape[0])
    print("Değişken Sayısı :", dataframe.shape[1])
veri_tanıtım(df)

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

df.head()

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

# RandomForestRegressor modelini oluşturma
rf_model = RandomForestRegressor()
rf_model.get_params()
# Modeli eğitme
rf_model.fit(X_train, y_train)

# Model performansını değerlendirme
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

user = X.sample(1, random_state=10)
model.predict(user)


# Hiperparametre aralıklarını belirleme
rf_params = {
    'n_estimators': [80, 90, 102, 130],
    'max_depth': [None, 2, 3, 4],
    'min_samples_split': [13, 14, 15, 16, 17],
    'min_samples_leaf': [4, 5, 6],
    'max_features': [ 'sqrt']
}

# GridSearchCV kullanarak hiperparametre optimizasyonu
rf_best_grid = GridSearchCV(rf_model,
                            rf_params,
                            cv=5,
                            verbose=True)
rf_final = rf_best_grid.fit(X,y)
print("En iyi parametreler:", rf_best_grid.best_params_)
print("En iyi model:", rf_best_grid.best_estimator_)


# Optimize edilmiş

model = RandomForestRegressor(max_depth=2,
                               max_features='sqrt',
                               min_samples_leaf=5,
                               min_samples_split=16,
                               n_estimators=90)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
