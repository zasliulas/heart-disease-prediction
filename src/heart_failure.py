import pandas as pd # Veri işleme ve analiz için
import numpy as np # Sayısal hesaplamalar için
import seaborn as sns # Görselleştirme için
import matplotlib.pyplot as plt # Grafik çizimleri için

# Makine öğrenmesi modelleri
import xgboost as xgb  #XGBoost modeli
from sklearn.svm import SVC  # Support Vector Machine
from sklearn.tree import DecisionTreeClassifier # Karar ağacı
from sklearn.neighbors import KNeighborsClassifier  # K-en yakın komşu
from sklearn.ensemble import RandomForestClassifier  # Rastgele orman
from sklearn.linear_model import LogisticRegression  # Lojistik regresyon

# Ön işleme (kategorik verileri sayısal verilere dönüştürmek için ve
# Kategorik verileri sayısal verilere dönüştürmek için
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Veriyi eğitim ve test olarak ayırma
from sklearn.model_selection import train_test_split

# Model değerlendirme metrikleri
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve


# In[23]:


df = pd.read_csv("C:\\Users\\Huawei\\PycharmProjects\\heart-disease-prediction\\data\\heart.csv") #Veri setini okumak için read komutu kullandık
df.head() #ilk 5 satırını görüntüler


# In[24]:


(df.info()) #Veri setinin genel yapısı hakkında bilgi


# In[25]:


print(df.isnull().sum()) #Eksik Veri Kontrolü


# In[26]:


df.dtypes #Veri setindeki her sütunun veri tiplerini görüntüler


# In[27]:


#Kategorik Değişkenler
kategorik = df.select_dtypes(include='object')

print("Kategorik değişkenler:", kategorik.columns.tolist())


# In[28]:


#Sayısal Değişkenler
sayisal = df.select_dtypes(include='number')
print("Sayısal değişkenler:", sayisal.columns.tolist())


# In[29]:


#HeartDisease Değişkenini Label Encode Etme
#Yes' ve 'No' değerlerini 1 ve 0 olarak dönüştürüyoruz
lb = LabelEncoder()  #LabelEncoder nesnesi oluşturuluyor
df['HeartDisease'] = lb.fit_transform(df['HeartDisease'])  


# Kategorik Verileri Encode Etme 
df = pd.get_dummies(df, columns=kategorik.columns.drop(["Sex","ChestPainType","RestingECG","ExerciseAngina","ST_Slope"]), drop_first=True)

# Sayısal Verileri Ölçekleme
scaler = StandardScaler() 
scaled_columns = ["Age","RestingBP","Cholesterol","FastingBS",
                  "MaxHR","Oldpeak"]
df[scaled_columns] = scaler.fit_transform(df[scaled_columns])# Seçilen sütunlara ölçekleme işlemi uygulanır


# In[30]:


df  #Bütün işlemeri yaptıktan sonra veri setimizi kontrol ediyoruz


# In[31]:


X = df.drop(columns=["HeartDisease"]) #(HeartDisease) dışındaki tüm sütunlar özellik olarak seçilir
y = df['HeartDisease'] #(HeartDisease) seçilir


# In[32]:


# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)


# In[33]:


reports = [] #Daha sonrasında modelleri karşılaştırmak için liste oluşturduk


# In[34]:


# One-Hot Encoding
df_encoded = pd.get_dummies(df, drop_first=True)

# Özellik ve hedef değişkenlerini ayırma
X=df_encoded.drop("HeartDisease", axis=1)
y=df_encoded["HeartDisease"]

# Eğitim ve test verisine ayırma
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)  # Sadece eğitim verisine fit edilmeli

# Modelin daha doğru öğrenmesini sağlıyoruz
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Modeli eğitmek, tahmin yapmak, değerlendirmek (XGBoost)

# Modeli Eğitme
xgb_model = xgb.XGBClassifier()  # XGBoost sınıflandırıcı nesnesi oluşturuluyor
xgb_model.fit(X_train, y_train)  # Eğitim verisi ile model eğitiliyor

# Test Verisinde Tahmin Yapma
y_pred = xgb_model.predict(X_test_scaled) # Test verisi ile tahmin yapılıyor

# Değerlendirme
confusionMatrix = confusion_matrix(y_test, y_pred)  # Karışıklık matrisi hesaplanıyor
# Karışıklık matrisi, TP,FP,TN ve FN sonuçları içerir.

classReport = classification_report(y_test, y_pred)  # Sınıflandırma raporu çıkarılıyor
# Precision, recall, F1 skoru ve doğruluk gibi metrikleri gösterir.

reports.append(classification_report(y_test, y_pred, output_dict=True)) 
 # Sonuçlar karşılaştırma listesine ekleniyor

y_pred_prob = xgb_model.predict_proba(X_test)[:, 1]  # Pozitif sınıf için olasılık değerleri alınıyor
# Test verisinde her bir örneğin pozitif sınıf olma olasılığ (1)hesaplanıyor.

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)  # ROC eğrisi için gerekli metrikler hesaplanıyor
# False Positive Rate ve True Positive Rate hesaplanarak ROC eğrisi çizilecektir.

rocScore = roc_auc_score(y_test, y_pred_prob)  # ROC AUC skoru hesaplanıyor
# ROC eğrisinin altındaki alanı hesaplar, yüksek AUC skoru iyi performansı gösterir.

# Confusion Matrix Çizme
plt.figure(figsize=(8,4)) # Grafik boyutunu ayarla
sns.heatmap(confusionMatrix,annot=True,fmt="d",cmap="Greens") # Isı haritası çiz, değerleri göster, tam sayı formatında, yeşil tonlarında
plt.title("Confusion Matrix (XGBoost)") # Grafik başlığı
plt.xlabel("Tahmin Edilen") # X ekseni etiketi
plt.ylabel("Gerçek") # Y ekseni etiketi
plt.show() # Grafiği göster

# ROC Eğrisini Çizme
plt.figure(figsize=(8,4))  # Grafik boyutu 8x4 olarak ayarlanır
plt.plot(fpr, tpr, label=f'XGBoost (AUC = {rocScore:.2f})')  
# ROC eğrisi çizilir ve etiket olarak AUC değeri eklenir
plt.plot([0, 1], [0, 1], 'k--')  
# Rastgele tahmin eğrisini temsil eden diyagonal çizilir
plt.xlabel('False Positive Rate')  # X ekseni adı: Yanlış Pozitif Oranı
plt.ylabel('True Positive Rate')  # Y ekseni adı: Doğru Pozitif Oranı
plt.title('ROC Eğrisi (XGBoost)')  # Grafik başlığı belirlenir
plt.legend()  # Eğri etiketi (AUC değeri) gösterilir
plt.show()  # Grafiği ekrana çizdir

print(f"ROC AUC: {rocScore:.4f}")  
# AUC skoru terminale yazdırılır, modelin genel başarı seviyesi özetlenir


# En Önemli Özellikleri Belirtme
importances = xgb_model.feature_importances_ # Eğitilmiş XGBoost modelinden özellik önem skorlarını al
features = X.columns # Orijinal veri setindeki özellik (sütun) isimlerini al
feat_importance = pd.Series(importances, index=features).sort_values(ascending=False).head(5)
 # Özellik önem skorlarını bir Pandas Serisine dönüştür,
 # özellik isimleriyle eşleştir, azalan sıraya göre sırala ve en önemli ilk 10 özelliği seç
feat_importance.plot(kind='barh',cmap="summer") 
# En önemli 5 özelliği yatay çubuk grafik olarak çiz, "summer" renk haritasını kullan
plt.title(f"En Önemli Özellikler (XGBoost)") # Grafik başlığını ayarla
plt.xlabel("Önem") # X ekseni etiketini ayarla
plt.show() # Grafiği göster

#Raporu Gösterme
print(classReport) # Sınıflandırma raporunu (precision, recall, f1-score vb.) konsola yazdır


# In[35]:


#Modeli eğitmek, tahmin yapmak, değerlendirmek (SVC)

# Modeli Eğitme
svc_model = SVC(probability=True) 
#(Support Vector Classifier) modelini oluşturuyoruz 
#ve olasılık tahmini yapılabilmesi için 'probability=True' ekliyoruz.
svc_model.fit(X_train, y_train) # Modeli eğitim verisiyle eğitiyoruz.

# Test Verisinde Tahmin Yapma
y_pred = svc_model.predict(X_test)

# Değerlendirme
confusionMatrix = confusion_matrix(y_test, y_pred)
classReport = classification_report(y_test, y_pred,zero_division=True) #Sınıflandırma raporu
reports.append(classification_report(y_test, y_pred,zero_division=True,output_dict=True))
#raporu sözlük formatında listeye ekle
y_proba = svc_model.predict_proba(X_test)[:, 1]  # Pozitif sınıfın olasılıkları
fpr, tpr, thresholds = roc_curve(y_test, y_proba)#roc eğrisi için gerekli olan fpr,tpr,eşik değer
rocScore = roc_auc_score(y_test, y_proba) #roc auc hesaplaması

# Confusion Matrix Çizme
plt.figure(figsize=(8,4))
sns.heatmap(confusionMatrix,annot=True,fmt="d",cmap="Greens")
plt.title("Confisuon Matrix (SVC)")
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.show()

# ROC Eğrisini Çizme
plt.plot(fpr, tpr, label=f'SVC (AUC = {rocScore:.2f})')#Roc eğrisi çizimi
plt.plot([0, 1], [0, 1], 'k--')# Rastgele sınıflandırıcıya ait referans çizgisi
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Eğrisi (SVC)')
plt.legend()
plt.show()

print(f"ROC AUC: {rocScore:.4f}") #skoru yazdırıyoruz

#Raporu Gösterme
print(classReport)


# In[36]:


# Modeli eğitmek, tahmin yapmak, değerlendirmek (Decision Tree)

# Modeli Eğitme
decT_model = DecisionTreeClassifier(max_depth=5,min_samples_split=10) # Performansı arttırmak için
decT_model.fit(X_train, y_train)

# Test Verisinde Tahmin Yapma
y_pred = decT_model.predict(X_test)

# Değerlendirme
confusionMatrix = confusion_matrix(y_test, y_pred)
classReport = classification_report(y_test, y_pred)
reports.append(classification_report(y_test, y_pred,output_dict=True))
y_pred_prob = decT_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
rocScore = roc_auc_score(y_test, y_pred_prob)

# Confusion Matrix Çizme
plt.figure(figsize=(8,4))
sns.heatmap(confusionMatrix,annot=True,fmt="d",cmap="Greens")
plt.title("Confusion Matrix (Decision Tree)")
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.show()

# ROC Eğrisini Çizme
plt.figure(figsize=(8,4))
plt.plot(fpr, tpr, label=f'Decision Tree (AUC = {rocScore:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Eğrisi (Decision Tree)')
plt.legend()
plt.show()

print(f"ROC AUC: {rocScore:.4f}")

# En Önemli Özellikleri Belirtme
importances = decT_model.feature_importances_
features = X.columns
feat_importance = pd.Series(importances, index=features).sort_values(ascending=False).head(10)
feat_importance.plot(kind='barh',cmap="summer")
plt.title(f"En Önemli Özellikler (Decision Tree)")
plt.xlabel("Önem")
plt.show()

#Raporu Gösterme
print(classReport)


# In[37]:


# Modeli eğitmek, tahmin yapmak, değerlendirmek (K-Nearest Neighbors)

# Modeli Eğitme
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)

# Test Verisinde Tahmin Yapma
y_pred = knn_model.predict(X_test)

# Değerlendirme
confusionMatrix = confusion_matrix(y_test, y_pred)
classReport = classification_report(y_test, y_pred, zero_division=True)
reports.append(classification_report(y_test, y_pred, zero_division=True,output_dict=True))
y_proba = knn_model.predict_proba(X_test)[:, 1]  # Pozitif sınıfın olasılıkları
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
rocScore = roc_auc_score(y_test, y_proba)

# Confusion Matrix Çizme
plt.figure(figsize=(8,4))
sns.heatmap(confusionMatrix, annot=True, fmt="d", cmap="Greens")
plt.title("Confusion Matrix (K-Nearest Neighbors)")
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.show()

# ROC Eğrisini Görselleştirme
plt.figure(figsize=(8,4))
plt.plot(fpr, tpr, label=f'K-NN (AUC = {rocScore:.2f})', color='darkorange')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Eğrisi (K-Nearest Neighbors)')
plt.legend()
plt.show()

print(f"\nROC AUC: {rocScore:.4f}")

# Raporu Gösterme
print(classReport)


# In[38]:


# Modeli eğitmek, tahmin yapmak, değerlendirmek (Random Forest)

# Modeli Eğitme
rndF_model = RandomForestClassifier()
rndF_model.fit(X_train, y_train)

# Test Verisinde Tahmin Yapma
y_pred = rndF_model.predict(X_test)

# Değerlendirme
confusionMatrix = confusion_matrix(y_test, y_pred)
classReport = classification_report(y_test, y_pred)
reports.append(classification_report(y_test, y_pred,output_dict=True))
y_pred_prob = rndF_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
rocScore = roc_auc_score(y_test, y_pred_prob)

# Confusion Matrix Çizme
plt.figure(figsize=(8,4))
sns.heatmap(confusionMatrix,annot=True,fmt="d",cmap="Greens")
plt.title("Confusion Matrix (Random Forest)")
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.show()

# ROC Eğrisini Çizme
plt.figure(figsize=(8,4))
plt.plot(fpr, tpr, label=f'Random Forest (AUC = {rocScore:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Eğrisi (Random Forest)')
plt.legend()
plt.show()

print(f"ROC AUC: {rocScore:.4f}")

# En Önemli Özellikleri Belirtme
importances = rndF_model.feature_importances_
features = X.columns
feat_importance = pd.Series(importances, index=features).sort_values(ascending=False).head(10)
feat_importance.plot(kind='barh',cmap="summer")
plt.title(f"En Önemli Özellikler (Random Forest)")
plt.xlabel("Önem")
plt.show()

#Raporu Gösterme
print(classReport)


# In[39]:


# Modeli eğitmek, tahmin yapmak, değerlendirmek (Logistic Regression)

# Modeli Eğitme
log_model = LogisticRegression(max_iter=8500,solver="saga",fit_intercept=True)
# model.n_iter_ ile kontrol yaptık 7930 çıkıyor %10da pay verdik o yüzden 8500 iter yaptık
# solver = "saga" ise saga daha büyük veri setlerinde olduğu için
log_model.fit(X_train, y_train)

# Test Verisinde Tahmin Yapma
y_pred = log_model.predict(X_test)

# Değerlendirme
confusionMatrix = confusion_matrix(y_test, y_pred)
classReport = classification_report(y_test, y_pred,zero_division=True)
reports.append(classification_report(y_test, y_pred,zero_division=True,output_dict=True))
y_pred_prob = log_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
rocScore = roc_auc_score(y_test, y_pred_prob)

# Confusion Matrix Çizme
plt.figure(figsize=(8,4))
sns.heatmap(confusionMatrix,annot=True,fmt="d",cmap="Greens")
plt.title("Confusion Matrix (Logistic Regression)")
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.show()

# ROC Eğrisini Çizme
plt.figure(figsize=(8,4))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {rocScore:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Eğrisi (Logistic Regression)')
plt.legend()
plt.show()

print(f"ROC AUC: {rocScore:.4f}")

#Raporu Gösterme
print(classReport)


# In[40]:


def predict_attrition(model, input_df, expected_columns, scaler, scaled_columns):
    input_encoded = pd.get_dummies(input_df)

    for col in expected_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    input_encoded = input_encoded[expected_columns]

    for col in scaled_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    to_scale = input_encoded[scaled_columns]

    if to_scale.shape[1] != len(scaler.mean_):
        raise ValueError(f"Scaler {len(scaler.mean_)} özellik bekliyor ama {to_scale.shape[1]} özellik geldi.\n"
                         f"scaled_columns: {scaled_columns}\nto_scale.columns: {to_scale.columns.tolist()}")

    scaled_array = scaler.transform(to_scale)
    scaled_df = pd.DataFrame(scaled_array, columns=scaled_columns, index=input_encoded.index)

    input_encoded[scaled_columns] = scaled_df

    input_encoded = input_encoded.astype(float)

    prob = model.predict_proba(input_encoded)[0][1]
    prediction = model.predict(input_encoded)[0]
    result = "Kalp hastalığı VAR" if prediction == 1 else "Kalp hastalığı YOK"
    return result, prob

# scaled_columns, X'in içinden gelen sayısal sütunlara göre belirlenmeli:
scaled_columns = list(X.columns)

# Modelleri tanımla
models = [xgb_model, svc_model, decT_model, knn_model, rndF_model, log_model]
model_names = ['XGBoost', 'SVC', 'Decision Tree', 'KNN', 'Random Forest', 'Logistic Regression']

# Örnek kişi
heart_input = pd.DataFrame([{
    'Age': 60,
    'Sex': 'M',
    'ChestPainType': 'ASY',
    'RestingBP': 140,
    'Cholesterol': 200,
    'FastingBS': 0,
    'RestingECG': 'Normal',
    'MaxHR': 150,
    'ExerciseAngina': 'Y',
    'Oldpeak': 1.5,
    'ST_Slope': 'Flat'
}])



print("🩺 Tahmin Sonuçları:\n")
for name, model in zip(model_names, models):
    result, prob = predict_attrition(model, heart_input, X.columns, scaler, scaled_columns)
    print(f"{name} ➤ Sonuç: {result} — Olasılık: %.2f" % (prob * 100) + "%\n")




# In[41]:


# Tüm modellerin sınıflandırma raporlarını tutmak için boş bir liste oluşturuyoruz
processed_reports = []
# Her model ve ona karşılık gelen classification_report çıktısını eşleştirerek dönüyoruz
for model_name, report in zip(models, reports):
    # Bu modelin sonuçlarını tabloya çeviriyoruz
    df_temp = pd.DataFrame(report).transpose().reset_index()
    # Sütun isimlerini düzenle
    df_temp.columns = ['Class', 'Precision', 'Recall', 'F1-Score', 'Support'] 
    # Model ismini ekle
# Tüm modellerin sınıflandırma raporlarını tutmak için boş bir liste oluşturuyoruz
processed_reports = []
# Her model ve ona karşılık gelen classification_report çıktısını eşleştirerek dönüyoruz
for model_name, report in zip(models, reports):
    # Bu modelin sonuçlarını tabloya çeviriyoruz
    df_temp = pd.DataFrame(report).transpose().reset_index()
    # Sütun isimlerini düzenle
    df_temp.columns = ['Class', 'Precision', 'Recall', 'F1-Score', 'Support'] 
    # Model ismini ekle


# In[42]:


df_temp.insert(0, 'Model', model_name.__class__.__name__)
# Hazırladığımız tabloyu genel listeye ekliyoruz
processed_reports.append(df_temp)
# Tüm raporları tek DataFrame'de birleştir
final_df_temp = pd.concat(processed_reports)
# Sonucu göster
final_df_temp

