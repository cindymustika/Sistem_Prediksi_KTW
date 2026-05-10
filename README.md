# 🎓 Prediksi Kelulusan Tepat Waktu Mahasiswa

Aplikasi ini merupakan sistem berbasis **Machine Learning** yang digunakan untuk memprediksi apakah mahasiswa akan **lulus tepat waktu (KTW)** atau **tidak**, berdasarkan data akademik.

Aplikasi ini dibangun menggunakan **Python** dan **Streamlit**, serta dapat diakses secara online melalui Streamlit Cloud.


## Demo Aplikasi 
https://sistemprediksiktw.streamlit.app


## Fitur Utama

- 🔹 Input data mahasiswa secara manual
- 🔹 Prediksi kelulusan (KTW / Non-KTW)
- 🔹 Tampilan interaktif berbasis web
- 🔹 Menu validasi data (10% data uji)
- 🔹 Menggunakan model Machine Learning terlatih


## Model Machine Learning

Model yang digunakan merupakan hasil eksperimen dari beberapa algoritma, yaitu:

- Decision Tree
- Random Forest
- Naive Bayes

Model terbaik dipilih berdasarkan evaluasi:

- Accuracy
- Precision
- Recall
- F1-Score
- AUC-ROC


## Input Fitur

Pengguna diminta mengisi data berikut:

- **Status**
  - 0 = Aktif
  - 1 = Lulus
- **IPK**
- **Semester Proposal**
- **Semester Hasil**
- **Gender**
  - 1 = Laki-laki
  - 2 = Perempuan


## Output

Hasil prediksi akan menampilkan:

- **KTW (Kelulusan Tepat Waktu)**
- **Non-KTW (Tidak Tepat Waktu)**


##  Struktur Repository

```
Sistem_Prediksi_KTW/
├── app.py
├── model_terbaik (1).joblib
├── fitur_model1 (1).joblib
├── X_val.joblib
├── y_val.joblib
├── requirements.txt
└── README.md
```

###  Keterangan

- `app.py` : Aplikasi utama Streamlit
- `model_terbaik (1).joblib` : Model Machine Learning terbaik
- `fitur_model1 (1).joblib` : Daftar fitur yang digunakan model
- `X_val.joblib` : Data validasi (fitur)
- `y_val.joblib` : Label data validasi
- `requirements.txt` : Daftar library yang dibutuhkan
- `README.md` : Dokumentasi proyek


