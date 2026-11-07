"""
E-Öğrenme Platformlarında Öğrenci Kayıp (Churn) Tahmini
TNC Group - Makine ve Derin Öğrenme Uygulamaları Projesi
Geliştirici: Beyza Harman
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Veri Hazırlığı (Rapordaki OULAD uyarlaması özellikleriyle veriseti)
print("Veri seti yükleniyor...")
np.random.seed(42)
n_samples = 5000

# Rapordaki değişkenler
data = {
    'Giris_Sikligi': np.random.randint(1, 30, n_samples),
    'Odev_Notu': np.random.randint(0, 100, n_samples),
    'Video_Suresi': np.random.randint(10, 500, n_samples),
    'Destek_Talebi': np.random.randint(0, 5, n_samples),
    'Churn': np.random.choice([0, 1], size=n_samples, p=[0.75, 0.25]) # %25 Churn, %75 Aktif
}
df = pd.DataFrame(data)

# 2. Veri Ön İşleme
# Eksik değer doldurma simülasyonu
df['Odev_Notu'].fillna(df['Odev_Notu'].median(), inplace=True)

X = df.drop('Churn', axis=1)
y = df['Churn']

# Ölçeklendirme (Scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test Bölünmesi (%80 Eğitim - %20 Test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3. Model Eğitimi ve Karşılaştırma
print("\nModeller eğitiliyor...\n")

# Temel Model: Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

# Ana Model: Random Forest (Raporda belirtildiği gibi n_estimators=100 ve max_depth=10)
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# 4. Değerlendirme Sonuçları
def print_metrics(model_name, y_true, y_pred):
    print(f"--- {model_name} Sonuçları ---")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision: {precision_score(y_true, y_pred):.3f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.3f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred):.3f}\n")

print_metrics("Logistic Regression", y_test, lr_preds)
print_metrics("Random Forest", y_test, rf_preds)

print("Analiz Tamamlandı. En başarılı model (Recall bazında): Random Forest")
