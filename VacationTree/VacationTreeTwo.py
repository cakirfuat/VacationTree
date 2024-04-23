import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV, cross_validate
import xgboost as xgb
import lightgbm as lgb
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder





# Görüntüleme ayarları
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.2f" % x)
warnings.simplefilter(action="ignore")

#Veri
df = pd.read_excel(".venv/Data/Tatil Tercihleri ve İlgi Alanları.xlsx", sheet_name="Değişken azaltma 2")
df.head(10)
#Aykırı değer düzenleme
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
df["YIL_TATİL_GÜN"].describe().T
df.head()

#Encoding
copy = df.copy()

binary_sutunlar = [col for col in copy.columns if copy[col].nunique() == 2]
labelencoder = LabelEncoder()
for col in binary_sutunlar:
    copy[col] = labelencoder.fit_transform(copy[col])

#Future engineering
copy['YAŞ'] = copy['YAŞ'].replace({'>50': 5, '45-50': 4, '35-44': 3, '25-34': 2, '18-24': 1})
copy['EĞİTİM_DURUMU'] = copy['EĞİTİM_DURUMU'].replace({'İlkokul Mezunu': 1, 'Lise Mezunu': 2, 'Öğrenci': 3, 'Ön Lisans Mezunu': 4,
                                                     'Lisans Mezunu': 5, 'Yüksek Lisans': 6, 'Ortaokul Mezunu': 7,
                                                     'Doktora veya Doktora Adayı': 8})
copy['ÇOCUK_SAYISI'] = copy['ÇOCUK_SAYISI'].replace({'Yok': 1, 'Bir': 2, 'İki': 3, 'İkiden Fazla': 4})
copy['GELİR'] = copy['GELİR'].replace({'Düşük Gelir Düzeyi': 1, 'Orta Gelir Düzeyi': 2, 'Yüksek Gelir Düzeyi': 3,
                                           'Üst Gelir Düzeyi': 4})
copy['TATİL_BÜTÇE'] = copy['TATİL_BÜTÇE'].replace({'%0-5': 1, '%6-10': 2, '%11-15': 3, '%16-20': 4,
                                                       '%21-25': 5, '%21-30': 6, '%26-30': 7, '%31 ve üzeri': 8})

copy['BURÇ'] = labelencoder.fit_transform(copy['BURÇ'])
copy['FAV_BÖLGE'] = labelencoder.fit_transform(copy['FAV_BÖLGE'])
copy['FAV_İL'] = labelencoder.fit_transform(copy['FAV_İL'])

dff = copy.drop(["SON_TATİL_TARİH", "ALKOL_TERCİH"], axis=1)


scaler = MinMaxScaler()
dff['YIL_TATİL_GÜN'] = scaler.fit_transform(dff[['YIL_TATİL_GÜN']])
dff.head(15)
# Atraksiyon katsayısı
dff['AKTV_Ağırlıklı_toplam'] = (dff['AKTV_1'] * 5 +
                                dff['AKTV_2'] * 7 +
                                dff['AKTV_3'] * 2 +
                                dff['AKTV_4'] * 4 +
                                dff['AKTV_5'] * 8 +
                                dff['AKTV_6'] * 1 +
                                dff['AKTV_7'] * 3 +
                                dff['AKTV_8'] * 10 +
                                dff['AKTV_9'] * 6 +
                                dff['AKTV_10'] * 9)
dff.head(15)
dff = df.copy()
# İlgilendiğiniz sütunlara göre Clustering
secilen_sutunlar = ['CİNSİYET',
                       'EĞİTİM_DURUMU',
                       'İLİŞKİ DURUMU',
                       'ÇOCUK_SAYISI',
                       'EH_DURUM',
                       'BURÇ',
                       'GELİR',
                       'TATİL_BÜTÇE',
                       'AKTV_Ağırlıklı_toplam',
                       'YAŞ']

secilen_sutunlar_tek = ['AKTV_Ağırlıklı_toplam']

scaler = MinMaxScaler()
dff['AKTV_Ağırlıklı_toplam'] = scaler.fit_transform(dff[['AKTV_Ağırlıklı_toplam']])

# Elbow Yöntemi ile hip küme belirleme
kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2,20))
elbow.fit(dff[secilen_sutunlar])
elbow.show()

kmeans = KMeans(n_clusters=8).fit(dff[secilen_sutunlar])
kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_

clusters = kmeans.labels_
####

dff.groupby('cluster').count()
#Modelleme
y = dff["cluster"]
X = dff[secilen_sutunlar]
#Cart
cart_model = DecisionTreeClassifier().fit(X,y)
cv_result = cross_validate(cart_model, X, y,
                           cv=5,
                           scoring=["accuracy", "precision", "recall","f1", "roc_auc"])

cv_result["test_accuracy"].mean() # 0.238
cv_result["test_precision"].mean()
cv_result["test_recall"].mean()
cv_result["test_f1"].mean()
cv_result["test_roc_auc"].mean()

#RandomForest
rf_model = RandomForestClassifier().fit(X,y)
cv_result_rf = cross_validate(rf_model, X, y,
                           cv=5,
                           scoring=["accuracy", "precision", "recall","f1", "roc_auc"])

cv_result_rf["test_accuracy"].mean()# 0.34
cv_result_rf["test_precision"].mean()
cv_result_rf["test_recall"].mean()
cv_result_rf["test_f1"].mean()
cv_result_rf["test_roc_auc"].mean()

# K en yakın komşu
knn_model = KNeighborsClassifier().fit(X, y)
cv_result_knn = cross_validate(rf_model, X, y,
                           cv=5,
                           scoring=["accuracy", "precision", "recall","f1", "roc_auc"])

cv_result_knn["test_accuracy"].mean() # 0.331
cv_result_knn["test_precision"].mean()
cv_result_knn["test_recall"].mean()
cv_result_knn["test_f1"].mean()
cv_result_knn["test_roc_auc"].mean()

# GradientBoosting

gb_model = GradientBoostingClassifier().fit(X, y)
cv_result_gb = cross_validate(rf_model, X, y,
                           cv=5,
                           scoring=["accuracy", "precision", "recall","f1", "roc_auc"])

cv_result_gb["test_accuracy"].mean() # 0.331
cv_result_gb["test_precision"].mean()
cv_result_gb["test_recall"].mean()
cv_result_gb["test_f1"].mean()
cv_result_gb["test_roc_auc"].mean()

# XGBoost

xgb_model = xgb.XGBClassifier().fit(X,y)
cv_result_xgb = cross_validate(xgb_model, X, y,
                           cv=5,
                           scoring=["accuracy", "precision", "recall","f1", "roc_auc"])

cv_result_xgb["test_accuracy"].mean() #0.331
cv_result_xgb["test_precision"].mean()
cv_result_xgb["test_recall"].mean()
cv_result_xgb["test_f1"].mean()
cv_result_xgb["test_roc_auc"].mean()

# LGBM

lgb_model = lgb.LGBMClassifier().fit(X,y)
cv_result_xgb = cross_validate(xgb_model, X, y,
                           cv=5,
                           scoring=["accuracy", "precision", "recall","f1", "roc_auc"])

cv_result_xgb["test_accuracy"].mean() #0.331
cv_result_xgb["test_precision"].mean()
cv_result_xgb["test_recall"].mean()
cv_result_xgb["test_f1"].mean()
cv_result_xgb["test_roc_auc"].mean()