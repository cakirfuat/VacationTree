import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.2f" % x)
warnings.simplefilter(action="ignore")

df = pd.read_excel(".venv/Data/Tatil Tercihleri ve İlgi Alanları.xlsx", sheet_name="Değişken azaltma 2")
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

### AYKIRI DEĞER İNCELEMESİ ###
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

### VERİ ÖZET ###
#Adım 4:Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.

# ANALYSIS OF NUMERICAL VARIABLES
##################################

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col, plot=True)

# ANALYSIS OF CATEGORICAL VARIABLES
##################################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col)
cat_summary(df,"MSZoning", plot=True )
#street



# ANALYSIS OF NUMERICAL VARIABLES BY TARGET
##################################
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "SalePrice", col)


# ANALYSIS OF CATEGORICAL VARIABLES BY TARGET
##################################

def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "SalePrice", col)



### VERİ ENCODİNG ###

def grab_col_names(dataframe, cat_th=10, car_th=27):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car, num_but_cat

cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)

copy = df.copy()
copy.head(15)

binary_sutunlar = [col for col in copy.columns if copy[col].nunique() == 2]

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

labelencoder = LabelEncoder()

for col in binary_sutunlar:
    copy[col] = labelencoder.fit_transform(copy[col])

cat_col = ["EĞİTİM_DURUMU", "YAŞ", "ÇOCUK_SAYISI",
           "BURÇ", "GELİR", "TATİL_BÜTÇE", "FAV_BÖLGE", "FAV_İL"]

for col in cat_col:
    print(copy[col].value_counts())

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

copy.head(15)
dff = copy.drop(["SON_TATİL_TARİH", "ALKOL_TERCİH"], axis=1)
dff.head()



# 'YIL_TATİL_GÜN' sütununu ölçeklendir

scaler = MinMaxScaler()
dff['YIL_TATİL_GÜN'] = scaler.fit_transform(dff[['YIL_TATİL_GÜN']])


###   DENETİMSİZ ÖĞRENME    ###

kmeans = KMeans(n_clusters=4).fit(dff)
kmeans.get_params()

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.inertia_
kmeans.labels_


## optimum küme sayısı belirleme

k_means = KMeans()
ssd = []
K = range(1, 30)

for k in K:
    k_means = kmeans = KMeans(n_clusters=k).fit(dff)
    ssd.append(k_means.inertia_)

plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık SSE / SSR / SSD")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show()

# Elbow Yöntemi ile hip küme belirleme
kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2,20))
elbow.fit(dff)
elbow.show()

kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(dff)
kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_

clusters = kmeans.labels_
df["cluster"] = clusters
df.head()

## Hiyerarşik Kümeleme
dff.head()
hc_average = linkage(dff, "average" )

# Dendrogramı çizdirelim
plt.figure(figsize=(10, 5))
dendrogram(hc_average, truncate_mode="lastp",
           p=10,
           show_leaf_counts=True,
           leaf_font_size=10)
plt.title('Average Bağlantı Yöntemi ile Hiyerarşik Kümeleme Dendrogramı')
plt.xlabel('Veri Noktaları')
plt.ylabel('Uzaklık')
plt.show()


dff["cluster"] = clusters

dff.head()


y = dff["cluster"]
X = dff.drop(["cluster"], axis=1)  # Drop 'customerID' as it's not a feature.
user = X.sample(1,random_state=23)
# Define a list of models to evaluate.
dff.head(5)
models = [
    #('KNN', KNeighborsClassifier()),
    #('CART', DecisionTreeClassifier(random_state=12345)),
    #('RF', RandomForestClassifier(random_state=12345)),
    #('RF', RandomForestRegressor)
    #('XGB', XGBClassifier(random_state=12345)),
]

# Loop through models to train and evaluate them using cross-validation.
# Scoring metrics include accuracy, F1 score, ROC AUC, precision, and recall.
for name, model in models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")

cart_model = DecisionTreeClassifier(random_state=17).fit(X,y)
cv_result = cross_validate(cart_model, X, y,
                           cv=2,
                           scoring=["accuracy", "precision", "recall","f1", "roc_auc"])

cv_result["test_accuracy"].mean()
cv_result["test_precision"].mean()
cv_result["test_recall"].mean()
cv_result["test_f1"].mean()
cv_result["test_roc_auc"].mean()

# Her bir küme için AKTV puanlarının toplamlarını hesapla
dff['AKTV_toplam'] = dff.loc[:, 'AKTV_1':'AKTV_10'].sum(axis=1)
dff.head()


# Atraksiyona göre ağırlıklı toplamlar chatGPT önerisi için girdi değerler
cluster_akts_toplam = dff.groupby('cluster')['AKTV_toplam'].mean()

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
dff.head()
#
dff.drop('cluster',inplace=True, axis=1)
dff.drop('AKTV_toplam',inplace=True, axis=1)

y = dff["cluster"]
X = dff.drop(["cluster"], axis=1)

# İlgilendiğiniz sütunları seçin
secilen_sutunlar = dff[['CİNSİYET',
                       'EĞİTİM_DURUMU',
                       'İLİŞKİ DURUMU',
                       'ÇOCUK_SAYISI',
                       'EH_DURUM',
                       'BURÇ',
                       'GELİR',
                       'TATİL_BÜTÇE',
                       'P_YALNIZ',
                       'P_CİFT',
                       'P_AİLE',
                       'P_ARKADAŞ',
                       'P_TUR',
                       'P_ARKADAŞ',
                       'AKTV_Ağırlıklı_toplam',
                       'YAŞ']]

# KMeans modelini oluşturun (örneğin 3 küme için)
kmeans = KMeans(n_clusters=5)

# Modeli seçilen sütunlar üzerinde eğitin
kmeans.fit(secilen_sutunlar)

# Küme etiketlerini elde edin
dff['cluster'] = kmeans.labels_



clusters = kmeans.labels_
dff["cluster"] = clusters
dff.head()
# Cluster'a göre gruplama yapıp AKTV_Ağırlıklı_toplam ortalamalarını hesaplama
cluster_aktv_ortalamaları = dff.groupby('cluster')['AKTV_Ağırlıklı_toplam'].mean()

# Ortalamaları artan sıralama ile sıralama
sıralı_ortalamalar = cluster_aktv_ortalamaları.sort_values()



























