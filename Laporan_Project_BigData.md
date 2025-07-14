# Laporan Project Big Data

## a. Data Collection
Data yang digunakan pada project ini adalah "Top 1000 World University.csv" yang berisi 1000 universitas terbaik dunia beserta berbagai metrik performa, lokasi, dan peringkat. Data diimpor ke dalam aplikasi dashboard menggunakan library pandas.

**Kode:**
```python
import pandas as pd

def load_data():
    df = pd.read_csv('Top 1000 World University.csv', sep=';')
    # Konversi tipe data numerik
    df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
    # ... (konversi kolom lain)
    return df

df = load_data()
```
**Penjelasan:**
Kode di atas digunakan untuk membaca file CSV dan melakukan konversi tipe data agar siap dianalisis.
- `pd.read_csv`: Membaca file CSV menjadi DataFrame.
- `pd.to_numeric`: Mengubah kolom menjadi tipe numerik.
- `load_data()`: Fungsi untuk memuat dan membersihkan data.

---

## b. EDA dan Visualisasi Data
EDA dilakukan dengan berbagai visualisasi interaktif menggunakan Plotly dan Streamlit.

**Kode (contoh bar chart Top 20 Skor):**
```python
import plotly.express as px

top_20 = df.nlargest(20, 'Score')
fig_top20 = px.bar(
    top_20, 
    x='Score', 
    y='Institution',
    orientation='h',
    title="Top 20 Universitas Berdasarkan Skor",
    color='Score',
    color_continuous_scale='viridis'
)
st.plotly_chart(fig_top20, use_container_width=True)
```
**Penjelasan:**
Kode ini menampilkan 20 universitas dengan skor tertinggi dalam bentuk bar chart horizontal.
- `df.nlargest`: Mengambil N baris dengan nilai terbesar pada kolom tertentu.
- `px.bar`: Membuat bar chart dengan Plotly Express.
- `st.plotly_chart`: Menampilkan grafik Plotly di aplikasi Streamlit.

---

## c. Analisis Korelasi
Analisis korelasi dilakukan dengan membuat matriks korelasi dan divisualisasikan dalam bentuk heatmap.

**Kode:**
```python
metrics_cols = ['Quality of Education', 'Alumni Employment', 'Quality of Faculty',
               'Research Output', 'Quality Publications', 'Influence', 'Citations', 'Score']
correlation_matrix = df[metrics_cols].corr()
fig_heatmap = px.imshow(
    correlation_matrix,
    text_auto=True,
    aspect="auto",
    color_continuous_scale='RdBu',
    title="Matriks Korelasi Metrik Kinerja"
)
st.plotly_chart(fig_heatmap, use_container_width=True)
```
**Penjelasan:**
Kode ini menghitung korelasi antar metrik performa dan menampilkannya dalam bentuk heatmap interaktif.
- `df[metrics_cols].corr()`: Menghitung matriks korelasi antar kolom numerik.
- `px.imshow`: Membuat heatmap dari matriks.
- `st.plotly_chart`: Menampilkan heatmap di Streamlit.

---

## d. Membuat Model Regresi Linier
Model regresi linier dibuat menggunakan scikit-learn, dengan fitur yang dapat dipilih pengguna.

**Kode:**
```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df[selected_features]
y = df['Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LinearRegression()
model.fit(X_train_scaled, y_train)
```
**Penjelasan:**
Kode ini membagi data menjadi data latih dan uji, melakukan standarisasi fitur, lalu melatih model regresi linier.
- `train_test_split`: Membagi data menjadi data latih dan uji.
- `StandardScaler`: Melakukan standarisasi fitur agar memiliki mean 0 dan std 1.
- `fit_transform`: Melatih scaler dan mentransformasi data latih.
- `transform`: Mentransformasi data uji dengan scaler yang sama.
- `LinearRegression()`: Membuat objek model regresi linier.
- `fit`: Melatih model pada data latih.

---

## e. Evaluasi Model Linier
Evaluasi model dilakukan dengan menghitung metrik R², RMSE, MAE, dan visualisasi hasil prediksi.

**Kode:**
```python
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

y_pred_test = model.predict(X_test_scaled)
test_r2 = r2_score(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_mae = mean_absolute_error(y_test, y_pred_test)

# Visualisasi prediksi vs aktual
test_fig = px.scatter(x=y_test, y=y_pred_test, title="Data Uji: Aktual vs Prediksi",
                      labels={'x': 'Skor Aktual', 'y': 'Skor Prediksi'})
test_fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()],
                              mode='lines', name='Prediksi Sempurna', line=dict(dash='dash', color='red')))
st.plotly_chart(test_fig, use_container_width=True)
```
**Penjelasan:**
Kode ini menghitung metrik evaluasi model dan menampilkan scatter plot antara skor aktual dan prediksi pada data uji.
- `predict`: Menghasilkan prediksi skor dari model.
- `r2_score`, `mean_squared_error`, `mean_absolute_error`: Menghitung metrik evaluasi model.
- `px.scatter`: Membuat scatter plot antara nilai aktual dan prediksi.
- `add_trace`: Menambahkan garis referensi ke plot.
- `st.plotly_chart`: Menampilkan plot di Streamlit. 