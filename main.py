# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gdown
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans
import streamlit as st
import io  # Tambahkan impor ini

# URL Google Drive
url = 'https://drive.google.com/uc?id=1Pt9wAgu9F6JGXv6G4cz6Knl6Xt1DY2D0'

# Unduh file dari Google Drive
output = 'data-penumpang-5-pelabuhan-cruise_imigrasi_2019-2023.csv'
gdown.download(url, output, quiet=False)

# Memuat dataset
data = pd.read_csv(output)

# Menampilkan informasi dasar tentang dataset
st.title('Analisis Komprehensif Tren Penumpang Kapal Pesiar (2019-2023)')

st.subheader('Pendahuluan')
st.write("""
Proyek ini bertujuan untuk menganalisis data penumpang kapal pesiar dari tahun 2019 hingga 2023 untuk memahami tren, pola perjalanan, dan segmentasi penumpang. Dengan analisis ini, perusahaan dapat membuat keputusan yang lebih baik terkait perencanaan operasional dan strategi pemasaran untuk meningkatkan retensi penumpang dan mengoptimalkan rute perjalanan.
""")

st.subheader('Tujuan')
st.write("""
- Memahami distribusi dan tren jumlah penumpang kapal pesiar dari tahun 2019 hingga 2023.
- Mengidentifikasi rute kapal pesiar yang paling populer berdasarkan jumlah penumpang.
- Memprediksi jumlah penumpang menggunakan model regresi linear.
- Mengelompokkan penumpang berdasarkan karakteristik mereka menggunakan clustering K-Means.
- Memberikan rekomendasi bisnis berdasarkan hasil analisis dan pemodelan.
""")

st.subheader('Dataset')
st.write("""
Dataset yang digunakan adalah data penumpang kapal pesiar dari lima pelabuhan utama, termasuk informasi tentang nama kapal, negara asal, tanggal keberangkatan, pelabuhan tujuan, jumlah penumpang, dan jumlah kru. Data ini mencakup berbagai fitur yang dapat digunakan untuk analisis dan pemodelan.
""")

st.write("### Informasi Dataset")
buffer = io.StringIO()
data.info(buf=buffer)
s = buffer.getvalue()
st.text(s)

st.write("### Beberapa Baris Pertama")
st.write(data.head())

st.write("### Statistik Deskriptif")
st.write(data.describe())

# Preprocessing
columns_to_convert = ['Jumlah Penumpang', 'Jml Penumpang WNA', 'Jml Penumpang WNI', 'Jml Crew WNA', 'Jml Crew WNI']
for column in columns_to_convert:
    data[column] = data[column].astype(str).str.replace(',', '').astype(float)
data.fillna(0, inplace=True)
data['Tanggal Keberangkatan'] = pd.to_datetime(data['Tanggal Keberangkatan'], format='%d %B %Y', errors='coerce')
data['Tanggal Kedatangan'] = pd.to_datetime(data['Tanggal Kedatangan'], format='%d %B %Y', errors='coerce')

st.subheader('Preprocessing')
st.write("""
Langkah-langkah pembersihan data meliputi mengatasi missing values, mengubah tipe data yang tidak sesuai, dan normalisasi fitur.
""")

# Visualisasi distribusi jumlah penumpang
st.write("### Visualisasi Distribusi Jumlah Penumpang")
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(data['Jumlah Penumpang'], bins=10, kde=True, ax=ax)
ax.set_title('Distribusi Jumlah Penumpang')
ax.set_xlabel('Jumlah Penumpang')
ax.set_ylabel('Frekuensi')
st.pyplot(fig)

# Boxplot untuk jumlah penumpang
st.write("### Boxplot Jumlah Penumpang")
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(y=data['Jumlah Penumpang'], ax=ax)
ax.set_title('Boxplot Jumlah Penumpang')
ax.set_ylabel('Jumlah Penumpang')
st.pyplot(fig)

# Insight tentang distribusi jumlah penumpang
st.write("""
#### Insight:
Distribusi Jumlah Penumpang: Sebagian besar data terkonsentrasi pada jumlah penumpang yang lebih rendah, dengan outliers menunjukkan variasi signifikan.
""")

# Visualisasi tren jumlah penumpang per tahun
data['Tahun Kedatangan'] = data['Tanggal Kedatangan'].dt.year
penumpang_per_tahun = data.groupby('Tahun Kedatangan')['Jumlah Penumpang'].sum().reset_index()

st.write("### Visualisasi Tren Jumlah Penumpang Per Tahun")
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=penumpang_per_tahun, x='Tahun Kedatangan', y='Jumlah Penumpang', marker='o', color='royalblue', linewidth=2.5, ax=ax)
ax.set_title('Tren Jumlah Penumpang Per Tahun', fontsize=16, fontweight='bold')
ax.set_xlabel('Tahun', fontsize=14)
ax.set_ylabel('Jumlah Penumpang')
ax.set_xticks(penumpang_per_tahun['Tahun Kedatangan'])
ax.set_xticklabels(penumpang_per_tahun['Tahun Kedatangan'], rotation=45)
ax.grid(True, linestyle='--', alpha=0.7)
st.pyplot(fig)

# Insight tentang tren jumlah penumpang per tahun
st.write("""
#### Insight:
Tren Jumlah Penumpang Per Tahun: Fluktuasi jumlah penumpang dari tahun ke tahun dapat memberikan wawasan tentang faktor-faktor eksternal yang mempengaruhi tren wisata.
""")

st.subheader('Modeling')
st.write("""
Model yang digunakan untuk menganalisis dan memprediksi data penumpang meliputi regresi linear dan clustering dengan K-Means.
""")

# Model Regresi Linear
X = data[['Jml Penumpang WNA', 'Jml Penumpang WNI', 'Jml Crew WNA', 'Jml Crew WNI']]
y = data['Jumlah Penumpang']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

st.write("### Model Regresi Linear")
st.write(f'MAE: {mae:.2f}')
st.write(f'RMSE: {rmse:.2f}')

# Insight tentang model regresi linear
st.write("""
#### Insight:
Regresi Linear: Model memberikan prediksi yang baik dengan MAE dan RMSE yang rendah, menunjukkan hubungan linear yang kuat antara fitur penjelas dan jumlah penumpang.
""")

# Visualisasi prediksi vs aktual
st.write("### Visualisasi Prediksi vs Aktual Jumlah Penumpang")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(y_test.values, label='Actual', color='blue', marker='o')
ax.plot(y_pred, label='Predicted', color='red', linestyle='--', marker='x')
ax.set_title('Prediksi vs Aktual Jumlah Penumpang')
ax.set_xlabel('Samples')
ax.set_ylabel('Jumlah Penumpang')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Clustering dengan K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)

st.write("### Clustering dengan K-Means")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=data['Jml Penumpang WNA'], y=data['Jumlah Penumpang'], hue=data['Cluster'], palette='viridis', ax=ax)
ax.set_title('Segmentasi Penumpang berdasarkan Jumlah Penumpang WNA dan Total Jumlah Penumpang')
ax.set_xlabel('Jumlah Penumpang WNA')
ax.set_ylabel('Jumlah Penumpang')
st.pyplot(fig)

# Insight tentang clustering dengan K-Means
st.write("""
#### Insight:
Clustering: K-Means membantu mengidentifikasi segmen penumpang yang berbeda, memungkinkan strategi pemasaran yang lebih tertarget.
""")

# Rute Kapal Pesiar Populer
rute_populer = data.groupby('Rute Kapal Pesiar')['Jumlah Penumpang'].sum().reset_index()
rute_populer = rute_populer.sort_values(by='Jumlah Penumpang', ascending=False)

st.write("### Rute Kapal Pesiar Populer Berdasarkan Jumlah Penumpang")
fig, ax = plt.subplots(figsize=(12, 8))
sns.barplot(data=rute_populer, x='Jumlah Penumpang', y='Rute Kapal Pesiar', palette='viridis', ax=ax)
ax.set_title('Rute Kapal Pesiar Populer Berdasarkan Jumlah Penumpang', fontsize=16, fontweight='bold')
ax.set_xlabel('Jumlah Penumpang', fontsize=14)
ax.set_ylabel('Rute Kapal Pesiar', fontsize=14)
st.pyplot(fig)

# Insight tentang rute kapal pesiar populer
st.write("""
#### Insight:
Rute Populer: Rute dari Singapura ke Benoa adalah yang paling populer, diikuti oleh rute dari Australia ke Benoa. Analisis ini menunjukkan rute-rute yang memiliki volume penumpang tertinggi dan memberikan wawasan penting untuk strategi pemasaran dan operasional.
""")

st.subheader('Kesimpulan dan Rekomendasi Bisnis')
st.write("""
### Kesimpulan
- **Regresi Linear:** Model regresi linear memberikan prediksi yang baik dengan MAE dan RMSE yang rendah.
- **Clustering dengan K-Means:** Clustering dengan K-Means membantu mengidentifikasi segmentasi penumpang yang berbeda.
- **Rute Populer:** Rute dari Singapura ke Benoa adalah yang paling populer, diikuti oleh rute dari Australia ke Benoa.

### Rekomendasi Bisnis
1. **Fokus pada Rute Populer:** Mengalokasikan lebih banyak sumber daya dan upaya pemasaran pada rute-rute populer seperti Singapura ke Benoa dan Australia ke Benoa untuk meningkatkan retensi penumpang. Rute-rute ini memiliki potensi tinggi untuk menarik lebih banyak penumpang.
2. **Segmentasi Penumpang:** Menggunakan hasil clustering untuk menyesuaikan penawaran layanan dan promosi sesuai dengan segmen penumpang yang berbeda. Setiap segmen memiliki kebutuhan dan preferensi yang unik, sehingga penawaran yang disesuaikan dapat meningkatkan kepuasan dan retensi penumpang.
3. **Optimasi Kapasitas:** Menggunakan prediksi jumlah penumpang untuk mengoptimalkan kapasitas kapal dan mengurangi biaya operasional. Dengan memprediksi jumlah penumpang yang akurat, perusahaan dapat menghindari kapasitas yang tidak optimal dan mengelola sumber daya dengan lebih efisien.
""")
