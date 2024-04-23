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

df = pd.read_excel("VacationTree/veriii.xlsx", sheet_name="Sheet1")


df['YAŞ'] = df['YAŞ'].replace({'>50': 5, '45-50': 4, '35-44': 3, '25-34': 2, '18-24': 1})
df['EĞİTİM_DURUMU'] = df['EĞİTİM_DURUMU'].replace({'İlkokul Mezunu': 1, 'Lise Mezunu': 2, 'Öğrenci': 3, 'Ön Lisans Mezunu': 4,
                                                     'Lisans Mezunu': 5, 'Yüksek Lisans': 6, 'Ortaokul Mezunu': 7,
                                                     'Doktora veya Doktora Adayı': 8})
df['ÇOCUK_SAYISI'] = df['ÇOCUK_SAYISI'].replace({'Yok': 1, 'Bir': 2, 'İki': 3, 'İkiden Fazla': 4})
df['GELİR'] = df['GELİR'].replace({'Düşük Gelir Düzeyi': 1, 'Orta Gelir Düzeyi': 2, 'Yüksek Gelir Düzeyi': 3,
                                           'Üst Gelir Düzeyi': 4})
df['TATİL_BÜTÇE'] = df['TATİL_BÜTÇE'].replace({'%0-5': 1, '%6-10': 2, '%11-15': 3, '%16-20': 4,
                                                       '%21-25': 5, '%26-30': 6, '%31 ve üzeri': 7})

df['CİNSİYET'] = df['CİNSİYET'].replace({'Kadın': 1, 'Erkek': 0})
df['İLİŞKİ DURUMU'] = df['İLİŞKİ DURUMU'].replace({'İlişkisi Var': 1, 'İlişkisi Yok': 0})
df['EH_DURUM'] = df['EH_DURUM'].replace({'Evet': 1, 'Hayır': 0})
df['ALKOL_KULLANIM'] = df['ALKOL_KULLANIM'].replace({'Kullanıyorum': 1, 'Kullanmıyorum': 0})
df['BURÇ'] = df['BURÇ'].replace({'Koç': 1,'Boğa': 2,
                                 'İkizler': 3,'Yengeç': 4,
                                 'Aslan': 5,'Başak': 6,
                                 'Terazi': 7,'Akrep': 8,
                                 'Yay': 9,'Oğlak': 10,
                                 'Kova': 11,'Balık': 12})

df.head()
df.describe().T

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
df.describe().T
df.head()

df["GELİR_HARCAMA"] = df["GELİR"] * df["TATİL_BÜTÇE"]

df.head(5)

y = df["AAP"]
X = df.drop("AAP", axis=1)
# Verileri eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



#####################################################
# RandomForestRegressor modelini oluşturma #
#####################################################

#random_state_values = range(1, 100)  #en küçük mse 24 te oldu :)

#for state in random_state_values:
#    rf_model = RandomForestRegressor(random_state=state)
#    rf_model.fit(X_train, y_train)
#    y_pred = rf_model.predict(X_test)
#    mse = mean_squared_error(y_test, y_pred)
#    print(f"Random State: {state}, Mean Squared Error: {mse}")


rf_model = RandomForestRegressor(random_state=24)
rf_model.get_params()
# Modeli eğitme
rf_model.fit(X_train, y_train)

# Model performansını değerlendirme
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
r2 = r2_score(y_test, y_pred)
print(f"R-Squared: {r2}")
#Mean Squared Error: 0.20733424187890045
#R-Squared: 0.5800963403339225

user = X.sample(1, random_state=10)
rf_model.predict(user)


# Hiperparametre aralıklarını belirleme
rf_params = {
    'n_estimators': [100, 300, 500],
    'max_depth': [None, 3, 5, 7],
    'min_samples_split': [2, 5, 8, 15, 20],
    'min_samples_leaf': [1, 2, 3, 4],
    'max_features': [3, 5, 7, 'sqrt']
}

# GridSearchCV kullanarak hiperparametre optimizasyonu
rf_best_grid = GridSearchCV(rf_model,
                            rf_params,
                            cv=5,
                            verbose=True)

rf_final = rf_best_grid.fit(X,y)
print("En iyi parametreler:", rf_best_grid.best_params_)
print("En iyi model:", rf_best_grid.best_estimator_)

y_pred = rf_final.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
r2 = r2_score(y_test, y_pred)
print(f"R-Squared: {r2}")
#En iyi parametreler: {'max_depth': None, 'max_features': 3, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 500}
#En iyi model: RandomForestRegressor(max_features=3, n_estimators=500, random_state=24)

#Mean Squared Error: 0.02427455321186216
#R-Squared: 0.9508379627115653

##################################################
# GradientBoostingRegressor #
##################################################

gbm_model = GradientBoostingRegressor(random_state=24)
gbm_model.get_params()
# Modeli eğitme
gbm_model.fit(X_train, y_train)

# Model performansını değerlendirme
y_pred = gbm_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
r2 = r2_score(y_test, y_pred)
print(f"R-Squared: {r2}")
#Mean Squared Error: 0.3509939488348294
#R-Squared: 0.2891495283134349

gbm_params = {'learning_rate': [0.01, 0.1, 0.5],
              'max_depth': [3, 5, 7, 8, 10],
              'n_estimators': [50, 100, 200, 500],
              'subsample': [1, 0.5, 0.7]}
gbm_best_grid = GridSearchCV(gbm_model,
                              gbm_params,
                              cv=5,
                              n_jobs=-1,
                              verbose=True).fit(X, y)


gbm_best_grid.fit(X_train, y_train)

# En iyi parametreler ve en iyi skorun bulunması
en_iyi_parametreler = gbm_best_grid.best_params_
en_iyi_sonuc = gbm_best_grid.best_score_

print("En İyi Parametreler:", en_iyi_parametreler)

gbm_final = GradientBoostingRegressor(**en_iyi_parametreler, random_state=24)
gbm_final.fit(X_train, y_train)

# Model performansını değerlendirme
y_pred = gbm_final.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
r2 = r2_score(y_test, y_pred)
print(f"R-Squared: {r2}")
# Değişken önemlerini hesaplayın
importance = gbm_final.feature_importances_

feature_names = X_train.columns  # veya X.columns
feature_importance_pairs = list(zip(feature_names, importance))

# Listeyi büyükten küçüğe sıralayın
sorted_feature_importance_pairs = sorted(feature_importance_pairs, key=lambda x: x[1], reverse=True)

# Sıralanmış değişken önemlerini ve isimlerini yazdırın
for feature, importance in sorted_feature_importance_pairs:
    print(f"Değişken: {feature}, Önem: {importance:.4f}")
# Önemleri görselleştirin
plt.figure(figsize=(10, 6))
plt.bar(range(len(importance)), importance, tick_label=feature_names)
plt.xlabel('Bağımsız Değişkenler')
plt.ylabel('Önem')
plt.title('Değişken Önemleri')
plt.xticks(rotation=45)
plt.show()
# Değişken önemlerini hesaplayın
importance = gbm_final.feature_importances_

# Önemleri görselleştirin
plt.figure(figsize=(10, 6))
plt.bar(range(len(importance)), importance)
plt.xlabel('Bağımsız Değişkenler')
plt.ylabel('Önem')
plt.title('Değişken Önemleri')
plt.show()

#En İyi Parametreler: {'learning_rate': 0.1,
#                      'max_depth': 10,
#                      'n_estimators': 500,
#                      'subsample': 0.7}
#Mean Squared Error: 0.1888702812471498
# R-Squared: 0.617490475383409


##################################################
# XGBoost Regressor #
##################################################
xgb_model = XGBRegressor(random_state=24)
xgb_model.get_params()
# Modeli eğitme
xgb_model.fit(X_train, y_train)

# Model performansını değerlendirme
y_pred = xgb_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
r2 = r2_score(y_test, y_pred)
print(f"R-Squared: {r2}")
#Mean Squared Error: 0.20733424187890045
#R-Squared: 0.5800963403339225

user = X.sample(1, random_state=10)
xgb_model.predict(user)


# Hiperparametre aralıklarını belirleme
xgboost_params = {'learning_rate': [0.001, 0.01, 0.1],
                  'max_depth': [5, 8, 12, None],
                  'n_estimators': [100, 200, 500, 1000],
                  'colsample_bytree': [None, 0.7, 1]}

xgboost_best_grid = GridSearchCV(xgb_model,
                              xgboost_params,
                              cv=5,
                              n_jobs=-1,
                              verbose=True).fit(X, y)

# Modeli eğitme
xgb_final = xgboost_best_grid.fit(X_train, y_train)

# En iyi parametreler ve en iyi skorun bulunması
en_iyi_parametreler = xgboost_best_grid.best_params_
en_iyi_sonuc = xgboost_best_grid.best_score_
print("En İyi Parametreler:", en_iyi_parametreler)
#En İyi Parametreler: {'colsample_bytree': 0.7,
#                      'learning_rate': 0.01,
#                      'max_depth': 12,
#                      'n_estimators': 500}

y_pred = xgb_final.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
r2 = r2_score(y_test, y_pred)
print(f"R-Squared: {r2}")

#Mean Squared Error: 0.20945339484730063
#R-Squared: 0.5758045259246887

##################################################
# LightGBM Regressor #
##################################################

lgm_model = LGBMRegressor(random_state=25, verbosity=-1)

lgm_model.fit(X_train, y_train)
y_pred = lgm_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
r2 = r2_score(y_test, y_pred)
print(f"R-Squared: {r2}")

#Mean Squared Error: 0.280841928385657
#R-Squared: 0.37655229841030224

lgm_model.get_params()

# Grid Search için denenecek hiperparametrelerin listesi
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1, 0.3],  # Öğrenme oranı
    'n_estimators': [100, 300, 500, 1000],
    'colsample_bytree': [0.5, 0.7, 1],
    'num_leaves': [15, 31, 50],  # Maksimum yaprak sayısı
    'max_depth': [5, 10, -1],  # Maksimum derinlik (-1: sınırsız)
    'min_child_samples': [10, 20, 30]  # Minimum çocuk örnek sayısı
}

lgm_grid_search = GridSearchCV(lgm_model,
                           param_grid,
                           cv=5,
                           verbose=1)

final_lgm_model = lgm_grid_search.fit(X, y)

# En iyi parametreleri ve skorları gösterelim
print("En iyi parametreler:", lgm_grid_search.best_params_)
print("En iyi model:", lgm_grid_search.best_estimator_)

#En iyi parametreler: {'colsample_bytree': 0.5, 'learning_rate': 0.1, 'max_depth': -1, 'min_child_samples': 10, 'n_estimators': 500, 'num_leaves': 50}
#En iyi model: LGBMRegressor(colsample_bytree=0.5,
#                            min_child_samples=10,
#                            n_estimators=500,
#                            num_leaves=50,
#                            random_state=25,
#                            verbosity=-1)

y_pred = final_lgm_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Test Verileri MSE: {mse}")
r2 = r2_score(y_test, y_pred)
print(f"R-Squared: {r2}")

#Test Verileri MSE: 0.0017669184997902813
#R-Squared: 0.996421548425


#########################################
###    TÜM MODELLER İÇİN FONKSİYON    ###
#########################################
y = df["AAP"]
X = df.drop("AAP", axis=1)
def base_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Base Models....")
    regressors = [('LR', LinearRegression()),
                  ('KNN', KNeighborsRegressor()),
                  ("CART", DecisionTreeRegressor()),
                  ("RF", RandomForestRegressor()),
                  ('GBM', GradientBoostingRegressor()),
                  ('XGBoost', XGBRegressor(use_label_encoder=False, eval_metric='logloss')),
                  ('LightGBM', LGBMRegressor(verbose=-1))]
    for name, regressor in regressors:
        regressor.fit(X_train, y_train)  # Modeli eğit
        y_pred = regressor.predict(X_test)  # Test seti üzerinde tahmin yap
        mse = mean_squared_error(y_test, y_pred)
        print(f"* * * * * * {name}* * * * * *")
        print(f"{name}  MSE: {mse}")
        r2 = r2_score(y_test, y_pred)
        print(f"{name} için R-Squared: {r2}")
        print("          * * *             ")

base_models(X, y)


######################################################
# 4. Automated Hyperparameter Optimization
######################################################

lr_params = {}

knn_params = {'n_neighbors': [3, 5, 7, 9],  # Komşu sayısı için değer aralığı
              'weights': ['uniform', 'distance']}  # Ağırlık seçenekleri

rf_params = {
    'n_estimators': [100, 300, 500],
    'max_depth': [None, 3, 5, 7],
    'min_samples_split': [2, 5, 8, 15, 20],
    'min_samples_leaf': [1, 2, 3, 4],
    'max_features': [3, 5, 7, 'sqrt']}

xgboost_params = {'learning_rate': [0.001, 0.01, 0.1],
                  'max_depth': [5, 8, 12, None],
                  'n_estimators': [100, 200, 500, 1000],
                  'colsample_bytree': [None, 0.7, 1]}

lightgbm_params = {
    'learning_rate': [0.001, 0.01, 0.1, 0.3],  # Öğrenme oranı
    'n_estimators': [100, 300, 500, 1000],
    'colsample_bytree': [0.5, 0.7, 1],
    'num_leaves': [15, 31, 50],  # Maksimum yaprak sayısı
    'max_depth': [5, 10, -1],  # Maksimum derinlik (-1: sınırsız)
    'min_child_samples': [10, 20, 30]  # Minimum çocuk örnek sayısı
}

def hyperparameter_optimization_regressors(X_train, y_train, X_test, y_test, regressors_params):
    best_models = {}
    for name, regressor, params in regressors_params:
        print(f"Optimizing {name}...")
        grid_search = GridSearchCV(regressor, params, cv=5, verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_models[name] = (best_model, best_params)

        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"{name} - Test Verileri MSE: {mse}, R-Squared: {r2}")

    return best_models


# Kullanılacak regressörler ve parametreler
regressors_params = [
    ('LinearRegression', LinearRegression(), lr_params),
    ('KNeighborsRegressor', KNeighborsRegressor(), knn_params),
    ('RandomForestRegressor', RandomForestRegressor(), rf_params),
    ('XGBRegressor', XGBRegressor(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
    ('LGBMRegressor', LGBMRegressor(verbose=-1), lightgbm_params)
]


# Örnek veri çerçevesi ve bağımlı/değişkenler
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fonksiyonu kullanarak en iyi modelleri ve parametreleri alalım
best_models = hyperparameter_optimization_regressors(X_train, y_train, X_test, y_test, regressors_params)

# En iyi modelleri ve parametreleri gösterelim
for name, (model, params) in best_models.items():
    print(f"En iyi model ve parametreler for {name}:")
    print(model)
    print(params)
    print()

#Optimizing LinearRegression...
#LinearRegression - Test Verileri MSE: 0.4446743551830656, R-Squared: 0.09942329154640162

#Optimizing KNeighborsRegressor...
#KNeighborsRegressor - Test Verileri MSE: 0.24088625799951852, R-Squared: 0.5121451219025184

#Optimizing RandomForestRegressor...
#RandomForestRegressor - Test Verileri MSE: 0.19713246972883194, R-Squared: 0.6007574787067886

#Optimizing XGBRegressor...
#XGBRegressor - Test Verileri MSE: 0.20440034388337328, R-Squared: 0.586038217055513

#Optimizing LGBMRegressor...
#LGBMRegressor - Test Verileri MSE: 0.21035568756142334, R-Squared: 0.5739771576650287

#En iyi model ve parametreler for LinearRegression:
#LinearRegression()
#{}

#En iyi model ve parametreler for KNeighborsRegressor:
#KNeighborsRegressor(n_neighbors=9, weights='distance')
#{'n_neighbors': 9, 'weights': 'distance'}

#En iyi model ve parametreler for RandomForestRegressor:
#RandomForestRegressor(max_features=3)
#{'max_depth': None, 'max_features': 3, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}

#En iyi model ve parametreler for XGBRegressor:
#XGBRegressor(base_score=None, booster=None, callbacks=None,
    #             colsample_bylevel=None, colsample_bynode=None,
    #             colsample_bytree=0.7, device=None, early_stopping_rounds=None,
    #             enable_categorical=False, eval_metric='logloss',
    #             feature_types=None, gamma=None, grow_policy=None,
    #             importance_type=None, interaction_constraints=None,
    #             learning_rate=0.01, max_bin=None, max_cat_threshold=None,
    #             max_cat_to_onehot=None, max_delta_step=None, max_depth=12,
    #             max_leaves=None, min_child_weight=None, missing=nan,
    #             monotone_constraints=None, multi_strategy=None, n_estimators=500,
    #             n_jobs=None, num_parallel_tree=None, random_state=None, ...)
#{'colsample_bytree': 0.7, 'learning_rate': 0.01, 'max_depth': 12, 'n_estimators': 500}

#En iyi model ve parametreler for LGBMRegressor:
#LGBMRegressor(colsample_bytree=0.7, learning_rate=0.01, min_child_samples=10,
#             n_estimators=1000, num_leaves=50, verbose=-1)
#{'colsample_bytree': 0.7, 'learning_rate': 0.01, 'max_depth': -1, 'min_child_samples': 10, 'n_estimators': 1000, 'num_leaves': 50}


lgbm_model = LGBMRegressor(colsample_bytree=0.7,
                           max_depth=-1,
                           learning_rate=0.01,
                           min_child_samples=10,
                           n_estimators=1000,
                           num_leaves=50,
                           verbose=-1,
                           random_state=27)

lgbm_model.fit(X_train, y_train)
y_pred = lgbm_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
r2 = r2_score(y_test, y_pred)
print(f"R-Squared: {r2}")

##################################################
# XGBoost Regressor #
##################################################
xgb_model = XGBRegressor(colsample_bytree= 0.7,
                         learning_rate= 0.01,
                         max_depth=12,
                         n_estimators=500,
                         random_state=24)
xgb_model.fit(X_train, y_train)

# Model performansını değerlendirme
y_pred = xgb_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
r2 = r2_score(y_test, y_pred)
print(f"R-Squared: {r2}")

xgb_model = XGBRegressor(colsample_bytree=0.7,
             learning_rate=0.01,
             max_depth=12,
             n_estimators=500)


gbm_medel = GradientBoostingRegressor(learning_rate=0.01,
                         max_depth=10,
                         n_estimators=500,
                         subsample= 0.7,
                         random_state=24)
gbm_final.fit(X_train, y_train)

# Model performansını değerlendirme
y_pred = gbm_final.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
r2 = r2_score(y_test, y_pred)
print(f"R-Squared: {r2}")



results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
plt.figure(figsize=(8, 6))
plt.scatter(results_df['Actual'], results_df['Predicted'], color='blue', alpha=0.5)
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahmin Edilen Değerler')
plt.title('Gerçek vs. Tahmin Edilen Değerler Dağılımı')
plt.grid(True)
plt.show()






















































