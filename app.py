import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="Sistem Prediksi KTW", layout="wide")

@st.cache_resource
def load_all_files():
    try:
        # Nama file dibuat bersih tanpa angka kurung agar tidak membingungkan GitHub
        model = joblib.load('model_terbaik.joblib')
        features = joblib.load('fitur_model.joblib')
        X_val = joblib.load('X_val.joblib')
        y_val = joblib.load('y_val.joblib')
        return model, features, X_val, y_val
    except Exception as e:
        st.error(f"Gagal memuat file model: {e}")
        return None, None, None, None

model, features, X_val, y_val = load_all_files()

# --- SIDEBAR NAVIGASI ---
st.sidebar.title("Menu Utama")
choice = st.sidebar.radio("Pilih Halaman:", ["Input Prediksi Manual", "Daftar Data Validasi (10%)"])

if model is not None:
    if choice == "Input Prediksi Manual":
        st.title("🎓 Prediksi Kelulusan Tepat Waktu Mahasiswa")
        st.write("Masukkan data akademik mahasiswa di bawah ini:")

        with st.form("my_form"):
            col1, col2 = st.columns(2)
            with col1:
                status = st.selectbox("Status [0:Aktif, 1:Lulus]", [0, 1])
                ipk = st.number_input("IPK", 0.0, 4.0, 3.5, step=0.01)
                gender = st.radio("Gender [1:L, 2:P]", [1, 2])
            with col2:
                smt_prop = st.number_input("Semester Proposal", 0, 14, 7)
                smt_hasil = st.number_input("Semester Hasil", 0, 14, 8)
            
            submit = st.form_submit_button("Cek Prediksi")

        if submit:
            if smt_hasil > 0 and smt_prop > 0 and smt_hasil < smt_prop:
                st.error("⚠️ Error: Seminar Hasil tidak boleh mendahului Seminar Proposal!")
            else:
                input_data = [[status, smt_hasil, ipk, smt_prop, gender]]
                prediction = model.predict(input_data)[0]
                proba = model.predict_proba(input_data)[0]

                st.divider()
                if prediction == 1: # KTW
                    st.success(f"HASIL: **KTW (Lulus Tepat Waktu)**")
                    st.info("💡 **Analisis:** Mahasiswa memiliki rekam jejak akademik yang baik dan mengikuti linimasa tugas akhir dengan tepat waktu.")
                else: # Non-KTW
                    st.warning(f"HASIL: **Non-KTW (Lulus Terlambat)**")
                    
                    st.write("🔍 **Analisis Faktor Penyebab Potensial:**")
                    alasan = []
                    if smt_hasil > 8:
                        alasan.append(f"- **Semester Seminar Hasil Terlambat:** Mahasiswa baru menempuh Seminar Hasil pada semester {int(smt_hasil)} (melebihi standar masa KTW maksimal semester 8).")
                    if smt_prop > 7:
                        alasan.append(f"- **Semester Seminar Proposal Terlambat:** Pengerjaan skripsi baru dimulai atau baru diseminarkan pada semester {int(smt_prop)}.")
                    if ipk < 3.00:
                        alasan.append(f"- **IPK Kurang Optimal:** Nilai IPK saat ini ({ipk:.2f}) berada di bawah 3.00, yang berpotensi memperlambat kelulusan.")
                    if smt_prop == 0 and smt_hasil == 0:
                        alasan.append("- **Belum Memulai Tugas Akhir:** Mahasiswa belum melaksanakan Seminar Proposal maupun Seminar Hasil.")
                    
                    if not alasan:
                        alasan.append("- **Kombinasi Tren Akademik:** Berdasarkan data historis, kombinasi nilai IPK dan semester aktif mahasiswa ini mengarah pada kelulusan Non-KTW.")
                    
                    for p in alasan:
                        st.write(p)
                
                st.write(f"**Probabilitas Kelas** → KTW: {proba[1]*100:.2f}% | Non-KTW: {proba[0]*100:.2f}%")

    elif choice == "Daftar Data Validasi (10%)":
        st.title("📊 Data Validasi")
        st.write("Ini adalah 10% dataset yang digunakan untuk menguji akurasi model.")
        
        try:
            df_X = pd.DataFrame(X_val)
            df_y = pd.DataFrame(y_val)
            df_val = pd.concat([df_X, df_y], axis=1)
            
            df_val.columns = ['Status', 'Smt Hasil', 'IPK', 'Smt Prop', 'L(1)/P(2)', 'Kelas']
            
            for col in ['Status', 'Smt Hasil', 'Smt Prop', 'L(1)/P(2)', 'Kelas']:
                df_val[col] = df_val[col].astype(int)
            
            df_val['Keterangan'] = df_val['Kelas'].map({1: 'KTW', 0: 'Non-KTW'})
            
            kolom_pilihan = ['Status', 'L(1)/P(2)', 'IPK', 'Smt Prop', 'Smt Hasil', 'Kelas', 'Keterangan']
            df_val_reordered = df_val[kolom_pilihan]
            
            df_display = df_val_reordered.rename(columns={
                'Status': 'Status (0=Aktif; 1=Lulus)',
                'L(1)/P(2)': 'Gender L[1]/P[2]',
                'IPK': 'IPK',
                'Smt Prop': 'Semester Seminar Proposal',
                'Smt Hasil': 'Semester Seminar Hasil'
            })
            
            st.dataframe(df_display, use_container_width=True)
            st.info("💡 **Tips:** Kamu bisa ambil data dari tabel ini untuk mencoba input manual di menu sebelah kiri.")
            
        except Exception as table_error:
            st.error(f"Gagal memproses struktur tabel data validasi: {table_error}")

else:
    st.error("Model gagal dimuat. Pastikan file joblib sudah di-upload ke GitHub dengan benar.")
