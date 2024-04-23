import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def train_and_evaluate_model(user=None,konaklama='Otel'):
    # Veri setini okuma
    df = pd.read_excel("VacationTree/veriii.xlsx", sheet_name="Sheet1")

    # Veri setini ön işleme
    df['YAŞ'] = df['YAŞ'].replace({'>50': 5, '45-50': 4, '35-44': 3, '25-34': 2, '18-24': 1})
    df['EĞİTİM_DURUMU'] = df['EĞİTİM_DURUMU'].replace(
        {'İlkokul Mezunu': 1, 'Lise Mezunu': 2, 'Öğrenci': 3, 'Ön Lisans Mezunu': 4,
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
    df['BURÇ'] = df['BURÇ'].replace({'Koç': 1, 'Boğa': 2,
                                     'İkizler': 3, 'Yengeç': 4,
                                     'Aslan': 5, 'Başak': 6,
                                     'Terazi': 7, 'Akrep': 8,
                                     'Yay': 9, 'Oğlak': 10,
                                     'Kova': 11, 'Balık': 12})

    def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
        quartile1 = dataframe[col_name].quantile(q1)
        quartile3 = dataframe[col_name].quantile(q3)
        interquantile_range = quartile3 - quartile1
        up_limit = quartile3 + 1.5 * interquantile_range
        low_limit = quartile1 - 1.5 * interquantile_range
        return low_limit, up_limit

    def replace_with_thresholds(dataframe, variable):
        low_limit, up_limit = outlier_thresholds(dataframe, variable)
        dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

    outlier_thresholds(df, "YIL_TATİL_GÜN")
    replace_with_thresholds(df, "YIL_TATİL_GÜN")


    y = df["AAP"]
    X = df.drop("AAP", axis=1)

    # Verileri eğitim ve test setlerine ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Gradient Boosting Regressor modelini oluşturma
    gbm_model = GradientBoostingRegressor(learning_rate=0.01,
                                          max_depth=10,
                                          n_estimators=500,
                                          subsample=0.7,
                                          random_state=24)
    gbm_model.fit(X_train, y_train)

    # Model performansını değerlendirme
    y_pred = gbm_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(mse)
    print(r2)

    # Değişken önemlerini hesaplayın
    importance = model.feature_importances_

    # Önemleri görselleştirin
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importance)), importance)
    plt.xlabel('Bağımsız Değişkenler')
    plt.ylabel('Önem')
    plt.title('Değişken Önemleri')
    plt.show()

    # Tahmin edilen ve gerçek değerlerin dağılımını görselleştirme
    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    plt.figure(figsize=(8, 6))
    plt.scatter(results_df['Actual'], results_df['Predicted'], color='blue', alpha=0.5)
    plt.xlabel('Gerçek Değerler')
    plt.ylabel('Tahmin Edilen Değerler')
    plt.title('Gerçek vs. Tahmin Edilen Değerler Dağılımı')
    plt.grid(True)
    plt.show()

    if user is not None:
        # Dışarıdan gelen veriyi kullanarak tahmin yapma
        user_pred = gbm_model.predict(user)
        print("Dışarıdan Verilen Örneğin Tahmini:", user_pred)
    dfTatil = pd.read_excel("VacationTree/veriTatil.xlsx", sheet_name="TÜMÜ")
    dfTatil['TATIL_MIN'] = dfTatil['TATIL_MIN'].astype(float)
    dfTatil['TATIL_MAX'] = dfTatil['TATIL_MAX'].astype(float)
    filtre = (dfTatil['KONAKLAMA_TUR'].str.contains(konaklama)) & (dfTatil['TATIL_MIN'] <= float(user_pred) ) & (dfTatil['TATIL_MAX']  >= float(user_pred))
    filtrelenmis = dfTatil[filtre].head(5)
    rastgele_secilen_veri = dfTatil.loc[filtre, ['SEHIR', 'ACIKLAMA']].sample(n=5)
    return rastgele_secilen_veri


# Modeli eğitme ve performansı değerlendirme


# Selectbox'ın seçenekleri ve valueları
dfGender = pd.DataFrame({'options' : ['Kadın', 'Erkek'],'CİNSİYET' : [1,0]})
dfPets = pd.DataFrame({'pets' : ['Evet','Hayır'],'EH_DURUM': [1,0]})
dfEduc = pd.DataFrame({"education" :  ['İlkokul Mezunu','Ortaokul Mezunu', 'Lise Mezunu','Öğrenci', 'Ön Lisans Mezunu','Lisans Mezunu',
             'Yüksek Lisans','Doktora veya Doktora Adayı'],'EĞİTİM_DURUMU': [1,2,3,4,5,6,7,8]})
dfAge = pd.DataFrame({'age': ['18-24','25-34','35-44','45-50','>50'], 'YAŞ': [1,2,3,4,5]})
dfMarital = pd.DataFrame({'maritalStatus':['EVET <3','Yalnızım '], 'İLİŞKİ DURUMU': [1,0]})
dfChild = pd.DataFrame({'child':['Yok','Bir','İki','İkiden Fazla'],'ÇOCUK_SAYISI': [1,2,3,4] })

dfHoroscope = pd.DataFrame({'horoscope': ['Koç','Boğa','İkizler','Yengeç','Aslan','Başak','Terazi','Akrep',
             'Yay','Oğlak','Kova','Balık'], 'BURÇ': [1,2,3,4,5,6,7,8,9,10,11,12]})
dfSalary = pd.DataFrame({'salary': ['Daha az - 300.000 TL','300.000 - 500.000 TL','500.000 - 700.000 TL','700.000 - Daha çok'],
                         'GELİR': [1,2,3,4]})

dfBudget = pd.DataFrame({'budget': ['%0-5','%6-10','%11-15','%16-20','%21-25','%26-30','%31 ve üzeri'],
                         'TATİL_BÜTÇE': [1,2,3,4,5,6,7]})
dfAlcohol = pd.DataFrame({'alcohol':['Seve seve','Kullanmıyorum'],'ALKOL_KULLANIM': ["1","0"] })

accomodation =  ['Otel','Kiralık Ev','Kamp','Bungalov','Pansiyon','Butik Otel']


st.markdown(
    """
    <style>
        /* Selectbox'ın arka plan rengini değiştir */
          div[data-baseweb="select"] > div:first-child {
            background-color: #B7F3E6 !important;
             width: 200px;
             
        }
          .stNumberInput .decrement, .stNumberInput .increment {
            display: none;
        }
         input {
         background-color: #B7F3E6 !important;
            width: 200px;
        }
        .stApp  {
           background-color: #F0EEDE !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Selectbox oluşturma

st.image("VacationTree/VacationTree.png", width=200)

# CSS ile resmin konumunu ayarlayın
st.markdown(
    """
    <style>
    /* Resmin konumunu belirleme */
    .stImage > img {
         display: flex;
        justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Seçilen değere göre değeri belirleme
# value = 1 if selected_gender == 'Kadın' else 0

# Seçilen cinsiyeti ve değeri görüntüleme
st.write(
    f'<h1 style="text-align:left; color:#FF5733; font-family: Broadway , sans-serif;font-size:32px; font-weight:bold; margin-bottom:20px">Harika öneriler için bana kendinden bahset.</h1>',
    unsafe_allow_html=True
)

sol_sutun,  orta_sutun, sag_sutun = st.columns([3,3,4])

# Sol sütun için içerik
with sol_sutun:

    sdg = st.selectbox("Cinsiyet?", dfGender['options'], key="unique_key_gender")
    sdm = st.selectbox("Hayatında kimse var mı?", dfMarital['maritalStatus'], key="unique_key_status")
    sdh = st.selectbox("Burcun ne?", dfHoroscope['horoscope'], key="unique_key_horoscope")
    user_input = st.text_input("Yılda kaç gün kendine ayırırsın?")
    # Kullanıcı sadece sayı girişi yapabilsin
    if user_input:
        try:
            users_int_input = int(user_input)
        except ValueError:
            st.write("Lütfen Sadece Sayı Giriniz.")

    # Sol sütunda istediğiniz içeriği oluşturabilirsiniz
with orta_sutun:
    sde = st.selectbox("Eğitim?", dfEduc['education'], key="unique_key_education")
    sdc = st.selectbox("Çoluk çocuk?", dfChild['child'], key="unique_key_child")
    sds = st.selectbox("Ne kadar kazanıyorsun? (Yıllık)", dfSalary['salary'], key="unique_key_salary")
    sdac = st.selectbox("Nerede mutlu olursun?", accomodation, key="unique_key_accomodation")
#Sağ sütun için içerik
with sag_sutun:
    sda = st.selectbox("Kaç oldun?", dfAge['age'], key="unique_key_age")
    sdp = st.selectbox("Evcil hayvan?", dfPets['pets'], key="unique_key_pets")
    sdb = st.selectbox("Tatil bütçen yıllık kazancının...", dfBudget['budget'], key="unique_key_budget")
    sdal = st.selectbox("İki tek atar mıyız?", dfAlcohol['alcohol'], key="unique_key_alcohol")

button_clicked = st.button("Rotaları gör")
st.markdown(
    """
    <style>
        .stButton>button {
            background-color: #FAA278;
            color: black;
            margin-left: 250px;
        }
        .stButton>button:hover {
        background-color: #FCD7C5; /* Yeşil renk */
        color: black; /* Beyaz renk */
    }
    </style>
    """,
    unsafe_allow_html=True
)

selected_gender = dfGender[(dfGender['options'] == sdg)]
selected_gender_values = selected_gender['CİNSİYET']
selected_education = dfEduc[(dfEduc['education'] == sde)]
selected_education_values = selected_education['EĞİTİM_DURUMU']
selected_yas = dfAge[(dfAge['age'] == sda)]
selected_yas_values = selected_yas['YAŞ']
selected_marital = dfMarital[(dfMarital['maritalStatus'] == sdm)]
selected_marital_values = selected_marital['İLİŞKİ DURUMU']
selected_child = dfChild[(dfChild['child'] == sdc)]
selected_child_values = selected_child['ÇOCUK_SAYISI']
selected_pets = dfPets[(dfPets['pets'] == sdp)]
selected_pets_values = selected_pets['EH_DURUM']
selected_horoscope = dfHoroscope[(dfHoroscope['horoscope'] == sdh)]
selected_horoscope_values = selected_horoscope['BURÇ']
selected_salary = dfSalary[(dfSalary['salary'] == sds)]
selected_salary_values = selected_salary['GELİR']
dfInput = pd.DataFrame()
dfInput['YIL_TATİL_GÜN'] = [user_input]
selected_user_input = dfInput['YIL_TATİL_GÜN']
selected_budget = dfBudget[(dfBudget['budget'] == sdb)]
selected_budget_values = selected_budget['TATİL_BÜTÇE']
selected_alcohol = dfAlcohol[(dfAlcohol['alcohol'] == sdal)]
selected_alcohol_values = selected_alcohol['ALKOL_KULLANIM']

# Birleştirilmiş DataFrame'in sütun isimlerini orijinal isimlerine geri çevirelim

# Buton tıklandığında bir mesaj gösterme
if button_clicked:
    st.title("***Mutlu olacağını düşündüğüm rotalar:***")
    st.write("--------------------------------------")
    merged_df = pd.concat([selected_gender_values.reset_index(drop=True),
                           selected_education_values.reset_index(drop=True),
                           selected_yas_values.reset_index(drop=True),
                           selected_marital_values.reset_index(drop=True),
                           selected_child_values.reset_index(drop=True),
                           selected_pets_values.reset_index(drop=True),
                           selected_horoscope_values.reset_index(drop=True),
                           selected_salary_values.reset_index(drop=True),
                           selected_user_input.reset_index(drop=True),
                           selected_budget_values.reset_index(drop=True),
                           selected_alcohol_values.reset_index(drop=True)], axis=1)
    oneri = train_and_evaluate_model(merged_df,sdac)
    for i, (etkinlik_yeri, aciklama) in enumerate(zip(oneri['SEHIR'], oneri['ACIKLAMA'])):

        if i % 2 == 0:
            st.success(f"**{i+1}-{etkinlik_yeri}** : {aciklama}")
        else:
            st.warning(f"**{i+1}-{etkinlik_yeri}** : {aciklama}")



