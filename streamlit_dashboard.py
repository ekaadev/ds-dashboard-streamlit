# streamlit_dashboard.py
"""
Streamlit app: Neon Glass Dashboard + Workshop & Jobfair (CSV-only, no SQL).
Put this file in your project folder. If present, the app will auto-load:
  - WORKSHOP 2023 - 2024.csv
  - JOBFAIR 2023 - 2024.csv

Run:
  pip install streamlit pandas matplotlib pydeck
  streamlit run streamlit_dashboard.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import base64
import io
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import numpy as np
import textwrap

# Import library untuk Machine Learning (Logistic Regression)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ---------------- Page config ----------------
st.set_page_config(page_title="Dashboard - Workshop & Jobfair", layout="wide")

# ---------------- CSS (Neon Glass) ----------------
NEON_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');
html, body, [class*="css"]  { font-family: 'Poppins', sans-serif; }
.stApp { background: linear-gradient(135deg, #e6fff5 0%, #dff0ff 35%, #f3e6ff 70%); background-size:400% 400%; animation: gradientMove 16s ease infinite; min-height:100vh; }
@keyframes gradientMove { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }

section[data-testid="stSidebar"]{ 
  background: linear-gradient(180deg,#071428,#0b2230); 
  color:#e6f9f1; 
  border-right:3px solid rgba(0,255,211,0.27); 
  padding-top:18px;
}
.sidebar-logo{display:inline-block;border-radius:10px;box-shadow:0 6px 18px rgba(0,0,0,0.45);}
.sidebar-title{color:#00ffc8;font-weight:700;text-align:center;margin-top:8px;}
.menu-header{color:#f8fafc;font-weight:700;margin-top:12px;margin-bottom:6px;padding-left:12px;}
div.stButton > button{border-radius:12px;padding:10px 14px;margin:6px 8px;font-weight:700;box-shadow:0 8px 24px rgba(0,0,0,0.22);border:none;}
.card { background: rgba(255,255,255,0.18); border-radius:18px; padding:22px; text-align:center; border:1px solid rgba(255,255,255,0.25); box-shadow:0 10px 30px rgba(0,0,0,0.12); transition:all .22s ease; }
.card-cta { background: linear-gradient(90deg, rgba(0,255,204,0.12), rgba(123,47,255,0.08)); border-radius:18px; padding:22px; text-align:center; border:1px solid rgba(0,255,200,0.08); }
.page-header{display:flex; align-items:center; gap:16px; margin-bottom:18px;}
.page-header h1{font-size:44px;color:#051726;margin:0;text-shadow:0 6px 18px rgba(0,0,0,0.12);}
.metric-card{ background: linear-gradient(180deg, rgba(255,255,255,0.85), rgba(255,255,255,0.7)); border-radius:12px; padding:14px; box-shadow:0 8px 20px rgba(0,0,0,0.06); }
hr{border:none;border-top:1px solid rgba(11,34,48,0.06);}

/* Sidebar radio selected label -> hitamkan */
section[data-testid="stSidebar"] input[type="radio"]:checked + label {
  color: #000 !important;
  font-weight: 700 !important;
}

/* Tweak radio layout in sidebar for compactness */
section[data-testid="stSidebar"] .stRadio > div {
  padding-left: 12px;
  padding-right: 12px;
}
</style>"""
st.markdown(NEON_CSS, unsafe_allow_html=True)

# ---------------- Helpers & constants ----------------
# Lokasi file setiap file csv yang akan di load
# File workshop 2023 - 2024 (.csv)
WORKSHOP_CSV = "WORKSHOP 2023 - 2024.csv"
# File jobfair 2023 - 2025 (.csv)
# File jobfair 2023 - 2025 (.csv) - Data lama (hasil)
JOBFAIR_CSV = "JOBFAIR HASIL 2023 - 2025.csv"
# File jobfair 2023 - 2025 (.csv) - Data baru (dengan kolom tambahan seperti JK, TTL, USIA, KOTA)
JOBFAIR_BARU_CSV = "JOBFAIR 2023-2025 BARU.csv"
# File People Analytics (.csv) - Data hasil prediksi logistic regression
PEOPLE_ANALYTICS_CSV = "People Analytics.csv"

DATE_CANDIDATES = ["tanggal", "date", "tgl", "waktu", "created_at"]

def img_to_datauri(path):
    if not path or not Path(path).exists():
        return None
    b = Path(path).read_bytes()
    mime = "image/png" if str(path).lower().endswith(".png") else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(b).decode()

def read_csv_flexible(path_or_bytes):
    try:
        if isinstance(path_or_bytes, (bytes, bytearray)):
            return pd.read_csv(io.BytesIO(path_or_bytes))
        return pd.read_csv(path_or_bytes)
    except Exception:
        if isinstance(path_or_bytes, (bytes, bytearray)):
            return pd.read_csv(io.BytesIO(path_or_bytes), encoding="latin1")
        return pd.read_csv(path_or_bytes, encoding="latin1")

def ensure_csv_exists(path, cols=None):
    p = Path(path)
    if not p.exists():
        cols = cols or ["id","nama","tanggal","kecamatan","peserta","keterangan"]
        pd.DataFrame(columns=cols).to_csv(p, index=False, encoding="utf-8-sig")

def ensure_date_column(df):
    if df is None or df.empty:
        return df
    for c in DATE_CANDIDATES:
        if c in df.columns:
            try:
                tmp = df.copy()
                tmp['tanggal'] = pd.to_datetime(tmp[c], errors='coerce')
                return tmp
            except Exception:
                try:
                    tmp['tanggal'] = pd.to_datetime(tmp[c].astype(str), errors='coerce')
                    return tmp
                except Exception:
                    continue
    return df

def sum_peserta_safe(df):
    if isinstance(df, pd.DataFrame) and 'peserta' in df.columns:
        try:
            return int(pd.to_numeric(df['peserta'], errors='coerce').fillna(0).sum())
        except Exception:
            return "—"
    return "—"

# ---------------- Logistic Regression untuk Prediksi Hasil Lamaran ----------------
# Fungsi-fungsi untuk mengimplementasikan model Logistic Regression
# yang memprediksi apakah peserta akan DITERIMA atau DITOLAK

# Import tambahan untuk logistic regression sesuai regresi.ipynb
try:
    from thefuzz import fuzz
    FUZZ_AVAILABLE = True
except ImportError:
    FUZZ_AVAILABLE = False

try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

def fuzzy_match(word, keywords, threshold=80):
    """Fungsi untuk fuzzy matching kata dengan keywords"""
    if not FUZZ_AVAILABLE:
        # Fallback ke simple matching jika thefuzz tidak tersedia
        word_upper = str(word).upper()
        for k in keywords:
            if k in word_upper:
                return True
        return False
    for k in keywords:
        if fuzz.partial_ratio(word, k) >= threshold:
            return True
    return False

def group_jabatan_fuzzy(jab):
    """Fungsi untuk mengelompokkan jabatan menggunakan fuzzy matching"""
    jab = str(jab).upper()

    if fuzzy_match(jab, ['ADMIN','ADM','BACK OFFICE']):
        return 'ADMINISTRASI'
    if fuzzy_match(jab, ['STORE','OUTLET','CREW','KASIR','WAIT','FRONTLINER']):
        return 'OUTLET/STORE'
    if fuzzy_match(jab, ['SALES','MARKETING','ACCOUNT EXECUTIVE','MEDICAL REPRESENTATIVE']):
        return 'SALES/MARKETING'
    if fuzzy_match(jab, ['AUDIT','FINANCE','COLLECTION']):
        return 'AUDIT/FINANCE'
    if fuzzy_match(jab, ['WAREHOUSE','GUDANG','LOGISTIK']):
        return 'WAREHOUSE/LOGISTIK'
    if fuzzy_match(jab, ['PRODUKSI','TEKNIS','OPERATOR','QUALITY','RND']):
        return 'PRODUKSI/TEKNIK'
    if fuzzy_match(jab, ['HOUSEKEEP','COOK','KOKI','CHEF','SUSTER']):
        return 'HOSPITALITY'

    return 'LAINNYA'

@st.cache_data
def train_logistic_regression_model(df):
    """
    Fungsi untuk melatih model Logistic Regression sesuai dengan regresi.ipynb
    Menggunakan preprocessing yang sama: JABATAN_GROUP, PEND_ORD, gender_bin, SMOTE

    Parameter:
    df: DataFrame yang berisi data Jobfair dengan kolom USIA, JK, PEND, HASIL, JABATAN

    Return:
    model: Model Logistic Regression (sklearn)
    classification_report_dict: Dictionary classification report
    df_result: DataFrame dengan hasil prediksi
    """

    # Menyaring data hanya yang memiliki HASIL = DITERIMA atau DITOLAK
    df_model = df[df['HASIL'].isin(['DITERIMA', 'DITOLAK'])].copy()

    # Menghapus baris dengan nilai kosong pada fitur yang digunakan
    df_model = df_model.dropna(subset=['USIA', 'JK', 'PEND', 'HASIL'])

    # Jika data terlalu sedikit, return None
    if len(df_model) < 50:
        return None, None, None

    # ===== PREPROCESSING sesuai regresi.ipynb =====

    # 1. Grouping JABATAN
    if 'JABATAN' in df_model.columns:
        df_model['JABATAN_GROUP'] = df_model['JABATAN'].apply(group_jabatan_fuzzy)

    # 2. Mapping Pendidikan ke ordinal
    edu_map = {
        'SD': 1, 'SMP': 2,
        'SMA': 3, 'SMK': 3,
        'D1': 4, 'D2': 5, 'D3': 6,
        'D4': 7, 'S1': 8, 'S2': 9, 'S3': 10
    }

    def map_pend(x):
        if pd.isna(x):
            return np.nan
        return edu_map.get(str(x).upper().strip(), np.nan)

    df_model['PEND_ORD'] = df_model['PEND'].apply(map_pend)

    # 3. Mapping HASIL ke binary (Y)
    def map_hasil(x):
        if pd.isna(x):
            return np.nan
        s = str(x).upper().strip()
        if "DITERIMA" in s:
            return 1
        if "DITOLAK" in s:
            return 0
        return np.nan

    df_model['Y'] = df_model['HASIL'].apply(map_hasil)

    # 4. Mapping Gender ke binary
    def map_gender(x):
        if pd.isna(x):
            return np.nan
        s = str(x).upper().strip()
        if s.startswith("L"):
            return 1
        if s.startswith("P"):
            return 0
        return np.nan

    df_model['gender_bin'] = df_model['JK'].apply(map_gender)

    # 5. Konversi USIA ke numeric
    df_model['USIA'] = pd.to_numeric(df_model['USIA'], errors='coerce')

    # 6. One-hot encoding untuk JABATAN_GROUP
    if 'JABATAN_GROUP' in df_model.columns:
        pos_dummies = pd.get_dummies(df_model['JABATAN_GROUP'], prefix="POS", drop_first=True)
        df_model = pd.concat([df_model, pos_dummies], axis=1)
        X_cols = ['gender_bin', 'PEND_ORD', 'USIA'] + list(pos_dummies.columns)
    else:
        X_cols = ['gender_bin', 'PEND_ORD', 'USIA']

    # ===== Siapkan X dan y =====
    X = df_model[X_cols]
    y = df_model['Y']

    # Hapus baris dengan NaN
    data_clean = pd.concat([X, y], axis=1).dropna()
    X = data_clean[X_cols].astype(float)
    y = data_clean['Y'].astype(int)

    if len(X) < 30:
        return None, None, None

    # ===== Split data =====
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # ===== Train model menggunakan sklearn LogisticRegression =====
    # Menggunakan class_weight='balanced' sebagai alternatif SMOTE
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # ===== Evaluasi model =====
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Classification report
    report = classification_report(y_test, y_pred, target_names=['DITOLAK', 'DITERIMA'], output_dict=True)

    # ===== Prediksi untuk seluruh data =====
    # Siapkan data untuk prediksi
    df_all = df.copy()

    # Preprocessing untuk data lengkap
    if 'JABATAN' in df_all.columns:
        df_all['JABATAN_GROUP'] = df_all['JABATAN'].apply(group_jabatan_fuzzy)
        pos_dummies_all = pd.get_dummies(df_all['JABATAN_GROUP'], prefix="POS", drop_first=True)
        df_all = pd.concat([df_all, pos_dummies_all], axis=1)

    df_all['PEND_ORD'] = df_all['PEND'].apply(map_pend)
    df_all['gender_bin'] = df_all['JK'].apply(map_gender)
    df_all['USIA'] = pd.to_numeric(df_all['USIA'], errors='coerce')

    # Pastikan semua kolom fitur ada
    for col in X_cols:
        if col not in df_all.columns:
            df_all[col] = 0

    # Prediksi
    X_all = df_all[X_cols].fillna(0).astype(float)
    df_all['pred_prob'] = model.predict_proba(X_all)[:, 1]
    df_all['pred_percent'] = (df_all['pred_prob'] * 100).round(2)
    df_all['pred_class'] = (df_all['pred_prob'] >= 0.5).astype(int)

    return model, report, df_all

def render_prediction_section(df):
    """
    Fungsi untuk menampilkan bagian prediksi hasil lamaran menggunakan Logistic Regression
    Sesuai dengan cara di regresi.ipynb - menampilkan classification report

    Parameter:
    df: DataFrame data Jobfair Baru
    """

    st.markdown("### Prediksi Hasil Lamaran (Logistic Regression)")

    # Melatih model
    with st.spinner("Melatih model prediksi..."):
        result = train_logistic_regression_model(df)

    if result[0] is None:
        st.warning("Data tidak cukup untuk melatih model prediksi. Minimal diperlukan 50 data dengan HASIL yang jelas (DITERIMA/DITOLAK).")
        return

    model, report, df_result = result

    # ===== Contoh Cara Prediksi =====

    st.caption("Masukkan profil peserta untuk melihat prediksi hasil lamaran")

    # Form input prediksi
    col_p1, col_p2, col_p3, col_p4 = st.columns([1, 1, 1, 1])

    with col_p1:
        input_usia = st.number_input("Usia", min_value=17, max_value=60, value=24, key="pred_usia")

    with col_p2:
        input_jk = st.selectbox("Jenis Kelamin", ["L", "P"], key="pred_jk",
                                format_func=lambda x: "Laki-laki" if x == "L" else "Perempuan")

    with col_p3:
        # Daftar pendidikan sesuai dengan mapping di model
        pend_options = ['SD', 'SMP', 'SMA', 'SMK', 'D1', 'D2', 'D3', 'D4', 'S1', 'S2', 'S3']
        input_pend = st.selectbox("Pendidikan", pend_options, index=4, key="pred_pend")

    with col_p4:
        # Daftar jabatan group
        jabatan_options = ['ADMINISTRASI', 'OUTLET/STORE', 'SALES/MARKETING', 'AUDIT/FINANCE',
                          'WAREHOUSE/LOGISTIK', 'PRODUKSI/TEKNIK', 'HOSPITALITY', 'LAINNYA']
        input_jabatan = st.selectbox("Kelompok Jabatan", jabatan_options, key="pred_jabatan")

    # Button prediksi
    if st.button("Prediksi Hasil", key="btn_predict", type="primary"):
        # Mapping pendidikan ke ordinal
        edu_map = {
            'SD': 1, 'SMP': 2, 'SMA': 3, 'SMK': 3,
            'D1': 4, 'D2': 5, 'D3': 6, 'D4': 7,
            'S1': 8, 'S2': 9, 'S3': 10
        }
        pend_ord = edu_map.get(input_pend, 5)

        # Mapping gender
        gender_bin = 1 if input_jk == "L" else 0

        # Buat fitur untuk prediksi
        # Cari nama kolom POS_ dari model
        feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else []

        # Buat dictionary fitur
        features_dict = {
            'gender_bin': gender_bin,
            'PEND_ORD': pend_ord,
            'USIA': input_usia
        }

        # Tambahkan dummy variable untuk jabatan
        for col in feature_names:
            if col.startswith('POS_'):
                jabatan_name = col.replace('POS_', '')
                features_dict[col] = 1 if jabatan_name == input_jabatan else 0

        # Buat DataFrame untuk prediksi
        X_pred = pd.DataFrame([features_dict])

        # Pastikan urutan kolom sesuai
        for col in feature_names:
            if col not in X_pred.columns:
                X_pred[col] = 0
        X_pred = X_pred[feature_names]

        # Prediksi
        pred_prob = model.predict_proba(X_pred)[0][1]  # Probabilitas DITERIMA
        pred_class = model.predict(X_pred)[0]
        pred_percent = pred_prob * 100

        # Tampilkan hasil
        jk_text = "Laki-laki" if input_jk == "L" else "Perempuan"

        if pred_class == 1:
            st.success(f"""
            **Hasil Prediksi: DITERIMA**
            
            Peserta dengan profil {jk_text}, usia {input_usia} tahun, pendidikan {input_pend}, 
            jabatan kelompok {input_jabatan} diprediksi **DITERIMA** dengan probabilitas **{pred_percent:.1f}%**.
            """)
        else:
            st.error(f"""
            **Hasil Prediksi: DITOLAK**
            
            Peserta dengan profil {jk_text}, usia {input_usia} tahun, pendidikan {input_pend}, 
            jabatan kelompok {input_jabatan} diprediksi **DITOLAK** dengan probabilitas **{100-pred_percent:.1f}%**.
            """)

        # Tampilkan probabilitas
        col_prob1, col_prob2 = st.columns(2)
        with col_prob1:
            st.metric("Probabilitas Diterima", f"{pred_percent:.1f}%")
        with col_prob2:
            st.metric("Probabilitas Ditolak", f"{100-pred_percent:.1f}%")


# ---------------- Login Page ----------------
# Halaman login sederhana yang muncul sebelum user bisa mengakses dashboard
# User hanya perlu memasukkan nama dan klik tombol masuk
# Data login akan disimpan ke file txt untuk record

# Nama file untuk menyimpan record login
LOGIN_RECORD_FILE = "login_records.txt"
CREDENTIALS_FILE = "credentials.txt"

def load_credentials():
    """
    Fungsi untuk membaca username dan password dari file credentials.txt

    Format file credentials.txt:
    username, password

    Contoh:
    User2, password123

    Jika ingin menambah user baru, tambahkan baris baru dengan format yang sama

    Returns:
    list: List of tuples [(username, password), ...] atau list kosong jika file tidak ditemukan
    """
    credentials_list = []

    try:
        secrets_auth = st.secrets["auth"]
        # st.info(secrets_auth)
        for username, password in secrets_auth.items():
            # st.info(f"USERNAME: {username}")
            # st.info(f"PASSWORD: {username}")
            credentials_list.append((username, password))
        # st.info(credentials_list)
        return credentials_list
    except Exception:
        # secrets tidak ada → lanjut ke local
        pass

    try:
        with open(CREDENTIALS_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line and ',' in line:
                    parts = line.split(',', 1)  # Split hanya pada koma pertama
                    username = parts[0].strip()
                    password = parts[1].strip()
                    credentials_list.append((username, password))
    except FileNotFoundError:
        return []

    # st.info(credentials_list)
    return credentials_list

def verify_login(input_username, input_password):
    """
    Fungsi untuk memverifikasi username dan password yang diinputkan

    Parameter:
    input_username: Username yang diinputkan user
    input_password: Password yang diinputkan user

    Returns:
    bool: True jika credential valid, False jika tidak valid
    """
    credentials_list = load_credentials()

    if not credentials_list:
        return False

    input_username = input_username.strip().lower()
    input_password = input_password.strip()

    # Membandingkan credential yang diinputkan dengan semua credential yang tersimpan
    # for valid_username, valid_password in credentials_list:
    #     if input_username == valid_username and input_password == valid_password:
    #         return True
    for valid_username, valid_password in credentials_list:
        if (
                input_username == valid_username.lower()
                and input_password == valid_password
        ):
            return True

    return False

def record_login(username):
    """
    Fungsi untuk menyimpan waktu login ke file txt

    Parameter:
    username: Username yang login

    Format record: TANGGAL_WAKTU | USERNAME
    """
    from datetime import datetime

    # Jika menggunakan secrets untuk auth, tidak perlu menyimpan login
    try:
        if "auth" in st.secrets:
            return # tidak perlu disimpan
    except Exception:
        pass

    # Mendapatkan waktu saat ini dengan format yang mudah dibaca
    waktu_login = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Menyusun record dalam format yang rapi
    record = f"{waktu_login} | {username}\n"

    # Menyimpan record ke file txt (mode 'a' untuk append/menambahkan)
    # Jika file belum ada, akan dibuat otomatis
    with open(LOGIN_RECORD_FILE, 'a', encoding='utf-8') as f:
        f.write(record)

def render_login_page():
    """
    Fungsi untuk menampilkan halaman login dengan autentikasi
    Halaman ini muncul sebelum user bisa mengakses dashboard
    User harus memasukkan username dan password yang valid
    Credential disimpan di file credentials.txt
    """

    # CSS khusus untuk halaman login
    login_css = """
    <style>
    .login-container {
        max-width: 450px;
        margin: 80px auto;
        padding: 40px;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
        text-align: center;
    }
    .login-title {
        color: #051726;
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .login-subtitle {
        color: #666;
        font-size: 14px;
        margin-bottom: 30px;
    }
    .login-logo-container {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-bottom: 25px;
    }
    .login-logo {
        width: 80px;
        height: 80px;
        object-fit: contain;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    </style>
    """
    st.markdown(login_css, unsafe_allow_html=True)

    # Mendapatkan data URI logo untuk ditampilkan di halaman login
    left_logo = img_to_datauri("assets/dispenaker-logo.png") or "https://i.ibb.co/3m1m3Z2/disperinaker.png"
    right_logo = img_to_datauri("assets/upnvjt-logo.png") or "https://i.ibb.co/TYM0dpd/upnvjt.png"

    # Membuat container tengah untuk form login
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        # Menampilkan logo
        st.markdown(f"""
        <div class="login-logo-container">
            <img src="{left_logo}" class="login-logo" alt="Disperinaker Logo">
            <img src="{right_logo}" class="login-logo" alt="UPNVJT Logo">
        </div>
        """, unsafe_allow_html=True)

        # Menampilkan judul
        st.markdown("""
        <div class="login-title">Selamat Datang</div>
        <div class="login-subtitle">Dashboard Workshop & Jobfair<br>Disperinaker × UPNVJT</div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Form input username
        # Menggunakan text_input untuk memasukkan username
        username_input = st.text_input(
            "Username",
            placeholder="Masukkan username...",
            key="login_username_input"
        )

        # Form input password
        # Menggunakan text_input dengan type="password" untuk menyembunyikan karakter
        password_input = st.text_input(
            "Password",
            placeholder="Masukkan password...",
            type="password",
            key="login_password_input"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Tombol masuk
        # width='stretch' membuat tombol memenuhi lebar container
        if st.button("Masuk", key="login_btn", width='stretch', type="primary"):
            # Validasi: username dan password tidak boleh kosong
            if username_input.strip() == "" or password_input.strip() == "":
                st.error("Username dan password tidak boleh kosong!")
            else:
                # Memverifikasi credential yang diinputkan
                if verify_login(username_input.strip(), password_input.strip()):
                    # Credential valid, record waktu login
                    record_login(username_input.strip())

                    # Menyimpan status login dan username ke session state
                    st.session_state['logged_in'] = True
                    st.session_state['user_name'] = username_input.strip()

                    # Menampilkan pesan sukses
                    st.success(f"Selamat datang, {username_input.strip()}!")

                    # Reload halaman untuk masuk ke dashboard
                    st.rerun()
                else:
                    # Credential tidak valid
                    st.error("Username atau password salah! Silakan coba lagi.")

        st.markdown("<br>", unsafe_allow_html=True)

        # Footer
        st.caption("© 2025 Disperinaker × UPNVJT")

# ---------------- Sidebar ----------------
LEFT_LOGO = img_to_datauri("assets/dispenaker-logo.png") or "https://i.ibb.co/3m1m3Z2/disperinaker.png"
RIGHT_LOGO = img_to_datauri("assets/upnvjt-logo.png") or "https://i.ibb.co/TYM0dpd/upnvjt.png"

ACTIVE_PAGE = st.session_state.get("page", "dashboard")

SIDEBAR_CSS = """
<style>
/* Logo: ukuran konsisten dan jelas */
.sidebar-logo {
    width: 84px !important;
    height: 84px !important;
    object-fit: contain !important;
    border-radius: 10px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.45);
    margin: 6px 6px;
    display: inline-block;
    vertical-align: middle;
}

/* Title / greeting */
.sidebar-title {
    color: #00ffc8;
    font-weight: 700;
    text-align: center;
    margin-top: 6px;
    margin-bottom: 8px;
}

/* Header text (SELAMAT DATANG) */
.menu-header {
    color: #f8fafc;
    font-weight:700;
    margin-top:12px;
    margin-bottom:6px;
    padding-left:12px;
}

/* Semua tombol di sidebar: buat teks gelap & latar cukup terang supaya selalu terlihat */
section[data-testid="stSidebar"] div.stButton > button {
    color: #0d1b2a !important;          /* teks gelap agar terlihat */
    background: rgba(255,255,255,0.85) !important; /* putih terang */
    border-radius:12px !important;
    padding:10px 14px !important;
    margin:10px 12px !important;
    font-weight:700 !important;
    box-shadow:0 8px 24px rgba(0,0,0,0.12) !important;
    width: 86% !important;
    text-align: left !important;
}

/* Jika butuh kontras ekstra untuk tombol aktif, kita beri border yang muncul ketika page aktif
   (kita gunakan span pengganti di bawah untuk menandai aktif) */
.sidebar-active-indicator {
    display: block;
    width: 86%;
    margin: -56px 12px 10px 12px; /* naikkan supaya berada di bawah tombol yang sesuai */
    height: 48px;
    border-radius:12px;
    background: rgba(0,0,0,0.06);
    pointer-events: none;
}

/* Styling caption di sidebar */
section[data-testid="stSidebar"] .stCaption {
    color: rgba(255,255,255,0.65);
    padding-left: 12px;
    padding-right: 12px;
}
</style>
"""
st.markdown(SIDEBAR_CSS, unsafe_allow_html=True)

# Sidebar hanya ditampilkan jika user sudah login
# Jika belum login, sidebar tidak akan muncul (halaman login ditampilkan)
if st.session_state.get('logged_in', False):
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center; margin-bottom:4px; padding-top:6px;">
           <img src="{LEFT_LOGO}" style="width:100px; height:auto;" class="sidebar-logo">
           <img src="{RIGHT_LOGO}" style="width:100px; height:auto;" class="sidebar-logo">
        </div>
        <div class="sidebar-title">Disperinaker × UPNVJT</div>
        <hr style="border:1px solid rgba(0,255,195,0.12); margin-top:8px; margin-bottom:10px;">
        """, unsafe_allow_html=True)

        # Menampilkan nama user yang login
        # Mengambil nama dari session state dan menampilkan sebagai greeting
        user_name = st.session_state.get('user_name', 'Pengguna')
        st.markdown(f"""
        <div class="menu-header">Halo, {user_name}!</div>
        """, unsafe_allow_html=True)

        # Tombol Dashboard
        #
        # if st.button("Dashboard", key="menu_dash"):
        #     st.session_state['page'] = 'dashboard'
        # indikator aktif (kosong/visual) - muncul tepat setelah tombol (kita akan render hanya untuk yang aktif)
        #
        # if ACTIVE_PAGE == "dashboard":
        #     st.markdown("<div class='sidebar-active-indicator'></div>", unsafe_allow_html=True)

        # Tombol Workshop
        if st.button("Workshop", key="menu_work"):
            st.session_state['page'] = 'workshop'
        if ACTIVE_PAGE == "workshop":
            st.markdown("<div class='sidebar-active-indicator'></div>", unsafe_allow_html=True)

        # Tombol Jobfair
        if st.button("Jobfair", key="menu_job"):
            st.session_state['page'] = 'jobfair'
        if ACTIVE_PAGE == "jobfair":
            st.markdown("<div class='sidebar-active-indicator'></div>", unsafe_allow_html=True)

        # Tombol Gabungan
        if st.button("Gabungan", key="menu_gabungan"):
            st.session_state['page'] = 'gabungan'
        if ACTIVE_PAGE == "gabungan":
            st.markdown("<div class='sidebar-active-indicator'></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("Navigasi cepat — pilih menu di atas")

        # Garis pemisah sebelum tombol logout
        st.markdown("<hr style='border:1px solid rgba(0,255,195,0.12); margin-top:20px; margin-bottom:10px;'>", unsafe_allow_html=True)

        # Tombol Logout
        # Ketika diklik, akan menghapus status login dari session state
        if st.button("Keluar", key="menu_logout"):
            # Menghapus status login dari session state
            st.session_state['logged_in'] = False
            st.session_state['user_name'] = None
            # Reload halaman untuk kembali ke halaman login
            st.rerun()



# ---------------- Auto-load CSVs into session (if present) ----------------
ensure_csv_exists(WORKSHOP_CSV)
ensure_csv_exists(JOBFAIR_CSV)
ensure_csv_exists(JOBFAIR_BARU_CSV)


if "df_workshop" not in st.session_state:
    try:
        st.session_state["df_workshop"] = read_csv_flexible(WORKSHOP_CSV)
    except Exception:
        st.session_state["df_workshop"] = pd.DataFrame()

if "df_jobfair" not in st.session_state:
    try:
        st.session_state["df_jobfair"] = read_csv_flexible(JOBFAIR_CSV)
    except Exception:
        st.session_state["df_jobfair"] = pd.DataFrame()

# Memuat data jobfair baru (dengan kolom tambahan JK, TTL, USIA, KOTA)
if "df_jobfair_baru" not in st.session_state:
    try:
        st.session_state["df_jobfair_baru"] = read_csv_flexible(JOBFAIR_BARU_CSV)
    except Exception:
        st.session_state["df_jobfair_baru"] = pd.DataFrame()

# Memuat data People Analytics (hasil prediksi logistic regression)
if "df_people_analytics" not in st.session_state:
    try:
        df_pa = read_csv_flexible(PEOPLE_ANALYTICS_CSV)
        # Konversi kolom pred_percent ke float (ganti koma dengan titik)
        if 'pred_percent' in df_pa.columns:
            df_pa['pred_percent'] = df_pa['pred_percent'].astype(str).str.replace(',', '.').astype(float)
        st.session_state["df_people_analytics"] = df_pa
    except Exception:
        st.session_state["df_people_analytics"] = pd.DataFrame()

# ---------------- DASHBOARD ----------------
def render_dashboard():
    st.markdown('<div class="page-header"><h1>Dashboard</h1></div>', unsafe_allow_html=True)

    # Ambil data
    df_w = st.session_state.get("df_workshop", pd.DataFrame())
    df_j = st.session_state.get("df_jobfair", pd.DataFrame())

    # Tab Dashboard
    tab1, tab2, tab3 = st.tabs(["Workshop", "Jobfair", "Gabungan"])

    # ======================== TAB WORKSHOP ========================
    with tab1:
        st.subheader("Dashboard Workshop")
        if df_w.empty:
            st.info("Data workshop belum ada.")
        else:
            total_w = len(df_w)
            st.metric("Total Peserta Workshop", total_w)

            st.bar_chart(df_w.select_dtypes(include='number'))

    # ======================== TAB JOBFAIR ========================
    with tab2:
        st.subheader("Dashboard Jobfair")
        if df_j.empty:
            st.info("Data jobfair belum ada.")
        else:
            total_j = len(df_j)
            st.metric("Total Peserta Jobfair", total_j)

            st.bar_chart(df_j.select_dtypes(include='number'))

    # ======================== TAB GABUNGAN ========================
    with tab3:
        st.subheader("Dashboard Gabungan")

        total_w = len(df_w) if not df_w.empty else 0
        total_j = len(df_j) if not df_j.empty else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Workshop", total_w)
        col2.metric("Jobfair", total_j)
        col3.metric("Total Gabungan", total_w + total_j)

        # Jika ada data numerik gabungan
        if not df_w.empty or not df_j.empty:
            try:
                df_merge = pd.concat([df_w, df_j], ignore_index=True)
                st.bar_chart(df_merge.select_dtypes(include='number'))
            except:
                st.info("Belum ada kolom numerik yang bisa digabungkan.")

    st.markdown("---")

# ---------------- WORKSHOP INSIGHTS ----------------
# Bagian untuk menampilkan wawasan dan analisis data workshop

# Menentukan lokasi file GeoJSON yang berisi data geografis kecamatan
# Path(__file__).parent mengambil direktori tempat file Python ini berada
# Menggunakan try-except untuk menangani kasus ketika __file__ tidak tersedia
try:
    GEOJSON_PATH = Path(__file__).parent / "kecamatan.geojson"
except NameError:
    # Fallback ke path relatif jika __file__ tidak tersedia
    GEOJSON_PATH = Path("kecamatan.geojson")

def render_workshop_insights(df, selected_location="Semua", selected_year="Semua"):
    """
    Fungsi untuk menampilkan 4 analisis utama:
    1. Klasifikasi peserta berdasarkan kecamatan dengan peta interaktif
    2. Distribusi usia peserta
    3. Tingkat pendidikan peserta
    4. Total peserta per tahun

    Parameter:
    df: DataFrame yang berisi data peserta workshop
    selected_location: Lokasi yang dipilih untuk filter (default: "Semua")
    selected_year: Tahun yang dipilih untuk filter (default: "Semua")
    """
    # Menyusun teks filter yang aktif untuk ditampilkan pada judul insight
    # Ini membantu user mengetahui filter apa saja yang sedang diterapkan
    filter_parts = []
    if selected_location != "Semua":
        filter_parts.append(f"Lokasi: {selected_location}")
    if selected_year != "Semua":
        filter_parts.append(f"Tahun: {selected_year}")
    filter_text = " | ".join(filter_parts) if filter_parts else "Semua Data"

    # Menampilkan judul bagian Insights dengan informasi filter
    st.markdown("### Insights")
    if filter_parts:
        st.caption(f"Filter aktif: {filter_text}")

    # ============ 1. Klasifikasi Participant Berdasarkan Kecamatan (Full Width dengan Map) ============
    # Bagian pertama: Menampilkan visualisasi data peserta berdasarkan kecamatan
    # Ditampilkan dalam format peta interaktif dan grafik batang
    # Menampilkan judul dengan informasi filter yang aktif
    st.markdown("#### Klasifikasi Peserta Berdasarkan Kecamatan")

    # Mengecek apakah kolom 'KEC' (Kecamatan) ada dalam DataFrame
    if 'KEC' in df.columns:
        # Menghitung jumlah peserta untuk setiap kecamatan
        # value_counts() menghitung frekuensi kemunculan setiap nilai unik
        kec_counts = df['KEC'].value_counts()

        # Mengecek apakah ada data kecamatan yang tersedia
        if not kec_counts.empty:
            # Membuat dua kolom tampilan: Peta (kiri) dan Grafik Batang (kanan)
            # Rasio 3:2 berarti kolom peta lebih lebar dari kolom grafik
            map_col, chart_col = st.columns([3, 2])

            # Menampilkan grafik batang di kolom sebelah kanan
            with chart_col:
                # Membuat grafik batang horizontal untuk 15 kecamatan teratas
                top15 = kec_counts.head(15)  # Mengambil 15 kecamatan dengan peserta terbanyak

                # Membuat figure dan axes untuk plot dengan ukuran 8x8 inci
                fig1, ax1 = plt.subplots(figsize=(8, 8))

                # Membuat background chart transparan
                fig1.patch.set_alpha(0)
                ax1.patch.set_alpha(0)

                # Membuat grafik batang horizontal
                # [::-1] digunakan untuk membalik urutan agar yang terbanyak di atas
                ax1.barh(top15.index[::-1], top15.values[::-1], color='#00CED1')

                # Menambahkan label untuk sumbu X dan Y
                ax1.set_xlabel('Jumlah Peserta')
                ax1.set_ylabel('Kecamatan')

                # Update judul grafik berdasarkan filter yang dipilih (lokasi dan/atau tahun)
                # Menyusun subtitle berdasarkan filter yang aktif
                subtitle_parts = []
                if selected_location != "Semua":
                    subtitle_parts.append(f"Lokasi: {selected_location}")
                if selected_year != "Semua":
                    subtitle_parts.append(f"Tahun: {selected_year}")

                if subtitle_parts:
                    ax1.set_title(f'Top 15 Kecamatan\n({", ".join(subtitle_parts)})')
                else:
                    ax1.set_title('Top 15 Kecamatan\n(Semua Data)')

                # Menambahkan label angka di setiap batang grafik
                # Menggunakan offset dinamis berdasarkan nilai maksimum untuk menghindari angka keluar grafik
                max_val = top15.values.max()
                for i, (idx, val) in enumerate(zip(top15.index[::-1], top15.values[::-1])):
                    # Jika nilai batang cukup besar (>5% dari max), taruh di dalam batang
                    # Jika tidak, taruh di luar batang dengan offset kecil
                    if val > max_val * 0.05:
                        ax1.text(val - max_val * 0.02, i, str(val), va='center', ha='right',
                                fontsize=9, fontweight='bold', color='white')
                    else:
                        ax1.text(val + max_val * 0.01, i, str(val), va='center', ha='left',
                                fontsize=9, fontweight='bold')

                # Mengatur batas sumbu X agar label tidak terpotong
                # Menambahkan 10% ruang ekstra di sebelah kanan
                ax1.set_xlim(0, max_val * 1.1)

                # Merapikan layout agar tidak terpotong
                plt.tight_layout()

                # Menampilkan grafik ke Streamlit
                # Streamlit akan otomatis re-render grafik karena fig1 dibuat ulang setiap kali
                st.pyplot(fig1, clear_figure=True)

                # Menutup figure untuk menghemat memori
                plt.close(fig1)

            # Menampilkan peta interaktif di kolom sebelah kiri
            with map_col:
                # Memuat file GeoJSON dan membuat peta
                # Menggunakan try-except untuk menangani error jika file tidak ditemukan
                try:
                    # Membaca file GeoJSON yang berisi data geografis kecamatan
                    # gpd (GeoPandas) digunakan untuk menangani data geografis
                    gdf = gpd.read_file(GEOJSON_PATH)

                    # Mengatur Coordinate Reference System (CRS) - sistem koordinat peta
                    # EPSG:4326 adalah sistem koordinat standar (latitude/longitude WGS84)
                    if gdf.crs is None:
                        # Jika belum ada CRS, set ke EPSG:4326
                        gdf.set_crs(epsg=4326, inplace=True)
                    else:
                        # Jika sudah ada CRS, konversi ke EPSG:4326
                        gdf = gdf.to_crs(epsg=4326)

                    # Mencari field/kolom yang berisi nama kecamatan dalam file GeoJSON
                    # Karena nama kolom bisa berbeda-beda, kita coba beberapa kemungkinan
                    geo_name_field = None
                    candidates = ["nm_kecamatan", "KECAMATAN", "NAMA_KEC", "NAME", "NAMA", "NAMOBJ"]

                    # Loop untuk menemukan kolom nama kecamatan yang sesuai
                    for cand in candidates:
                        if cand in gdf.columns:
                            geo_name_field = cand
                            break  # Keluar dari loop jika sudah ketemu

                    # Jika kolom nama kecamatan ditemukan dalam GeoJSON
                    if geo_name_field:
                        # Mengelompokkan data peserta berdasarkan kecamatan
                        # groupby digunakan untuk mengelompokkan data
                        # agg (aggregate) digunakan untuk melakukan operasi pada setiap grup
                        kec_summary = df.groupby('KEC').agg({
                            'NAMA': 'count',  # Menghitung jumlah peserta (menghitung baris nama)
                        }).rename(columns={'NAMA': 'Total Peserta'})  # Mengganti nama kolom

                        # Menambahkan informasi jumlah peserta berdasarkan jenis kelamin
                        if 'JK' in df.columns:  # Jika kolom Jenis Kelamin ada
                            # Mengelompokkan berdasarkan kecamatan dan jenis kelamin
                            # unstack mengubah data dari format panjang ke format lebar
                            gender_counts = df.groupby(['KEC', 'JK']).size().unstack(fill_value=0)

                            # Menambahkan kolom Laki-laki jika ada
                            if 'L' in gender_counts.columns:
                                kec_summary['Laki-laki'] = gender_counts['L']

                            # Menambahkan kolom Perempuan jika ada
                            if 'P' in gender_counts.columns:
                                kec_summary['Perempuan'] = gender_counts['P']

                        # Menambahkan informasi pendidikan yang paling dominan di setiap kecamatan
                        if 'PEND' in df.columns:  # Jika kolom Pendidikan ada
                            # mode() mencari nilai yang paling sering muncul
                            dominant_pend = df.groupby('KEC')['PEND'].agg(
                                lambda x: x.mode()[0] if len(x.mode()) > 0 else 'N/A'
                            )
                            kec_summary['Pendidikan Dominan'] = dominant_pend

                        # Menyiapkan kunci pencocokan untuk menggabungkan data
                        # Mengubah semua nama ke huruf kapital dan menghapus spasi
                        # Ini dilakukan agar pencocokan nama lebih akurat
                        kec_summary['name_key_df'] = kec_summary.index.astype(str).str.upper().str.strip()
                        gdf['name_key_geo'] = gdf[geo_name_field].astype(str).str.upper().str.strip()

                        # Menggabungkan data geografis dengan data peserta
                        # merge menggabungkan dua DataFrame berdasarkan kolom kunci
                        # how='left' berarti semua data dari gdf tetap dipertahankan
                        gdf_merged = gdf.merge(kec_summary, left_on='name_key_geo', right_on='name_key_df', how='left')

                        # Mengisi nilai kosong (NaN) dengan 0 untuk kecamatan tanpa peserta
                        gdf_merged['Total Peserta'] = gdf_merged['Total Peserta'].fillna(0)

                        # Menghitung titik tengah peta menggunakan batas koordinat
                        # total_bounds mengembalikan batas minimum dan maksimum koordinat
                        bounds = gdf.total_bounds  # Format: [minx, miny, maxx, maxy]

                        # Menghitung longitude (bujur) tengah
                        center_lon = (bounds[0] + bounds[2]) / 2

                        # Menghitung latitude (lintang) tengah
                        center_lat = (bounds[1] + bounds[3]) / 2

                        # Membuat peta interaktif menggunakan Folium
                        # location: koordinat pusat peta
                        # zoom_start: level zoom awal (semakin besar semakin dekat)
                        # tiles: tema/gaya tampilan peta
                        m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB Positron")

                        # Menambahkan layer choropleth (peta tematik berwarna)
                        # Choropleth mewarnai area berdasarkan nilai data
                        folium.Choropleth(
                            geo_data=gdf_merged.__geo_interface__,  # Data geografis
                            data=gdf_merged,  # Data yang akan divisualisasikan
                            columns=['name_key_geo', 'Total Peserta'],  # Kolom kunci dan nilai
                            key_on='feature.properties.name_key_geo',  # Kunci pencocokan di GeoJSON
                            fill_color='YlOrRd',  # Skema warna: Kuning-Oranye-Merah
                            fill_opacity=0.7,  # Tingkat ketransparanan isi (0-1)
                            line_opacity=0.5,  # Tingkat ketransparanan garis batas
                            legend_name='Total Peserta per Kecamatan',  # Judul legenda
                            nan_fill_color='lightgray'  # Warna untuk area tanpa data
                        ).add_to(m)  # Menambahkan layer ke peta

                        # Menambahkan marker berbentuk lingkaran untuk setiap kecamatan yang memiliki data
                        # Loop melalui setiap baris data kecamatan yang sudah digabungkan
                        for _, row in gdf_merged.iterrows():
                            # Mengecek apakah baris ini memiliki data geometri
                            if row.geometry is not None:
                                # Mengambil dan memformat data untuk ditampilkan
                                # pd.notna() mengecek apakah nilai bukan NaN (kosong)
                                total = int(row['Total Peserta']) if pd.notna(row['Total Peserta']) else 0
                                nama = row.get(geo_name_field, 'N/A')
                                laki = int(row.get('Laki-laki', 0)) if pd.notna(row.get('Laki-laki')) else 0
                                perempuan = int(row.get('Perempuan', 0)) if pd.notna(row.get('Perempuan')) else 0
                                pend = row.get('Pendidikan Dominan', 'N/A') if pd.notna(row.get('Pendidikan Dominan')) else 'N/A'

                                # Mendapatkan titik representatif untuk menempatkan marker
                                # Ini adalah titik yang mewakili posisi area kecamatan
                                rep_point = row.geometry.representative_point()

                                # Hanya menambahkan marker jika ada peserta
                                if total > 0:
                                    # Membuat konten popup HTML yang akan muncul saat marker diklik
                                    popup_html = f"""
                                    <b>Kecamatan:</b> {nama}<br>
                                    <b>Total Peserta:</b> {total}<br>
                                    <b>Laki-laki:</b> {laki} | <b>Perempuan:</b> {perempuan}<br>
                                    <b>Pendidikan Dominan:</b> {pend}
                                    """

                                    # Membuat marker lingkaran di peta
                                    folium.CircleMarker(
                                        location=[rep_point.y, rep_point.x],  # Posisi marker (lat, lon)
                                        radius=min(4 + (total / 10), 15),  # Ukuran lingkaran sesuai jumlah peserta, maksimal 15
                                        color='#3388ff',  # Warna garis tepi
                                        fill=True,  # Lingkaran diisi dengan warna
                                        fill_opacity=0.7,  # Tingkat ketransparanan isi
                                        popup=folium.Popup(popup_html, max_width=300)  # Popup informasi
                                    ).add_to(m)  # Menambahkan marker ke peta

                        # Menampilkan peta di Streamlit dengan ukuran tertentu
                        st_folium(m, width=700, height=500)
                    else:
                        # Jika tidak menemukan kolom nama kecamatan yang sesuai
                        st.warning("Field nama kecamatan tidak ditemukan dalam GeoJSON.")

                # Menangani error jika file GeoJSON tidak ditemukan
                except FileNotFoundError:
                    st.warning(f"File GeoJSON tidak ditemukan: {GEOJSON_PATH}")

                # Menangani error lain yang mungkin terjadi saat memuat peta
                except Exception as e:
                    st.error(f"Error memuat peta: {str(e)}")
        else:
            # Jika tidak ada data kecamatan dalam DataFrame
            st.info("Data kecamatan tidak tersedia.")
    else:
        # Jika kolom 'KEC' tidak ditemukan dalam DataFrame
        st.info("Kolom 'KEC' tidak ditemukan dalam data.")

    # Membuat garis pemisah horizontal
    st.markdown("---")

    # Membuat dua kolom dengan lebar sama untuk grafik berikutnya
    col1, col2 = st.columns(2)

    # ============ 2. Distribusi Usia Peserta ============
    # Menampilkan histogram distribusi usia peserta di kolom pertama
    with col1:
        st.markdown("#### Distribusi Usia Peserta")

        # Mengecek apakah kolom 'USIA' ada dalam DataFrame
        if 'USIA' in df.columns:
            # Mengkonversi kolom USIA ke tipe numerik
            # errors='coerce' akan mengubah nilai yang tidak valid menjadi NaN
            # dropna() menghapus nilai NaN
            ages = pd.to_numeric(df['USIA'], errors='coerce').dropna()

            # Mengecek apakah ada data usia yang valid
            if not ages.empty:
                # Membuat figure dan axes untuk histogram
                fig2, ax2 = plt.subplots(figsize=(8, 6))

                # Membuat background chart transparan
                fig2.patch.set_alpha(0)
                ax2.patch.set_alpha(0)

                # Membuat histogram dengan 20 bin (kelompok)
                # edgecolor='white' memberikan garis putih antar batang
                # alpha=0.8 membuat batang sedikit transparan
                ax2.hist(ages, bins=20, color='#FF6B6B', edgecolor='white', alpha=0.8)

                # Menambahkan label sumbu
                ax2.set_xlabel('Usia (tahun)')
                ax2.set_ylabel('Jumlah Peserta')
                ax2.set_title('Distribusi Usia Peserta')

                # Menambahkan garis vertikal untuk menunjukkan rata-rata usia
                # linestyle='--' membuat garis putus-putus
                ax2.axvline(ages.mean(), color='#4ECDC4', linestyle='--', linewidth=2, label=f'Rata-rata: {ages.mean():.1f} tahun')

                # Menampilkan legenda
                ax2.legend()

                # Merapikan layout
                plt.tight_layout()

                # Menampilkan grafik
                st.pyplot(fig2, clear_figure=True)

                # Menutup figure untuk menghemat memori
                plt.close(fig2)
            else:
                st.info("Data usia tidak tersedia.")
        else:
            st.info("Kolom 'USIA' tidak ditemukan dalam data.")

    # ============ 3. Total Peserta per Tahun ============
    # Menampilkan grafik batang jumlah peserta per tahun di kolom kedua
    with col2:
        st.markdown("#### Total Peserta per Tahun")

        # Mengecek apakah kolom 'TAHUN' ada dalam DataFrame
        if 'TAHUN' in df.columns:
            # Menghitung jumlah peserta per tahun
            # value_counts() menghitung frekuensi setiap tahun
            # sort_index() mengurutkan berdasarkan tahun (dari kecil ke besar)
            year_counts = df['TAHUN'].value_counts().sort_index()

            # Mengecek apakah ada data tahun
            if not year_counts.empty:
                # Membuat figure dan axes untuk grafik batang
                fig4, ax4 = plt.subplots(figsize=(8, 6))

                # Membuat background chart transparan
                fig4.patch.set_alpha(0)
                ax4.patch.set_alpha(0)

                # Membuat grafik batang vertikal
                # astype(str) mengubah tahun menjadi string agar lebih rapi di sumbu X
                bars = ax4.bar(year_counts.index.astype(str), year_counts.values, color='#9B59B6', edgecolor='white')

                # Menambahkan label sumbu
                ax4.set_xlabel('Tahun')
                ax4.set_ylabel('Jumlah Peserta')
                ax4.set_title('Total Peserta per Tahun')

                # Menambahkan label angka di atas setiap batang
                # Menggunakan offset dinamis berdasarkan nilai maksimum untuk menghindari angka keluar grafik
                max_height = year_counts.values.max()
                for bar, val in zip(bars, year_counts.values):
                    # Menghitung posisi tengah batang untuk menempatkan teks
                    # Offset dinamis = 2% dari nilai maksimum agar proporsional
                    offset = max(max_height * 0.02, 1)  # Minimal offset 1
                    # ha='center' menengahkan teks secara horizontal
                    # va='bottom' menempatkan teks di atas batang
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                            str(val), ha='center', va='bottom', fontweight='bold', fontsize=9)

                # Mengatur batas sumbu Y agar label tidak terpotong
                # Menambahkan 15% ruang ekstra di atas untuk label angka
                ax4.set_ylim(0, max_height * 1.15)

                # Merapikan layout
                plt.tight_layout()

                # Menampilkan grafik
                st.pyplot(fig4, clear_figure=True)

                # Menutup figure untuk menghemat memori
                plt.close(fig4)
            else:
                st.info("Data tahun tidak tersedia.")
        else:
            st.info("Kolom 'TAHUN' tidak ditemukan dalam data.")

    # ============ 4. Distribusi Tingkat Pendidikan (Lebar Penuh) ============
    st.markdown("#### Distribusi Tingkat Pendidikan")

    # Mengecek apakah kolom 'PEND' (Pendidikan) ada dalam DataFrame
    if 'PEND' in df.columns:
        # Menghitung jumlah peserta untuk setiap tingkat pendidikan
        edu_counts = df['PEND'].value_counts()

        # Mengecek apakah ada data pendidikan
        if not edu_counts.empty:
            # Membuat figure dan axes dengan ukuran lebih lebar (12x5)
            fig3, ax3 = plt.subplots(figsize=(12, 5))

            # Membuat background chart transparan
            fig3.patch.set_alpha(0)
            ax3.patch.set_alpha(0)

            # Membuat grafik batang horizontal untuk menghindari label yang menumpuk
            # Menggunakan colormap Set3 untuk memberikan warna berbeda pada setiap batang
            colors = plt.cm.Set3([i / len(edu_counts) for i in range(len(edu_counts))])

            # Membuat grafik batang horizontal
            # [::-1] membalik urutan agar yang terbanyak di atas
            bars = ax3.barh(edu_counts.index[::-1], edu_counts.values[::-1], color=colors[::-1])

            # Menambahkan label sumbu
            ax3.set_xlabel('Jumlah Peserta')
            ax3.set_ylabel('Tingkat Pendidikan')
            ax3.set_title('Distribusi Tingkat Pendidikan')

            # Menambahkan label angka di setiap batang
            # Menggunakan offset dinamis berdasarkan nilai maksimum untuk menghindari angka keluar grafik
            max_width = edu_counts.values.max()
            for bar, val in zip(bars, edu_counts.values[::-1]):
                # Menempatkan teks di sebelah kanan batang dengan offset proporsional
                # bar.get_width() mendapatkan panjang batang
                # bar.get_y() + bar.get_height()/2 menempatkan teks di tengah vertikal batang
                # Jika nilai batang cukup besar (>5% dari max), taruh di dalam batang
                # Jika tidak, taruh di luar batang dengan offset kecil
                if val > max_width * 0.05:
                    ax3.text(bar.get_width() - max_width * 0.02, bar.get_y() + bar.get_height()/2,
                            str(val), ha='right', va='center', fontsize=9, fontweight='bold', color='white')
                else:
                    ax3.text(bar.get_width() + max_width * 0.01, bar.get_y() + bar.get_height()/2,
                            str(val), ha='left', va='center', fontsize=9, fontweight='bold')

            # Mengatur batas sumbu X agar label tidak terpotong
            # Menambahkan 10% ruang ekstra di sebelah kanan
            ax3.set_xlim(0, max_width * 1.1)

            # Merapikan layout
            plt.tight_layout()

            # Menampilkan grafik
            st.pyplot(fig3, clear_figure=True)

            # Menutup figure untuk menghemat memori
            plt.close(fig3)
        else:
            st.info("Data pendidikan tidak tersedia.")
    else:
        st.info("Kolom 'PEND' tidak ditemukan dalam data.")

    # ============ 5. Distribusi Jenis Kelamin ============
    st.markdown("#### Distribusi Jenis Kelamin")

    # Mengecek apakah kolom 'JK' (Jenis Kelamin) ada dalam DataFrame
    if 'JK' in df.columns:
        # Menghitung jumlah peserta untuk setiap jenis kelamin
        gender_counts = df['JK'].value_counts()

        # Mengecek apakah ada data jenis kelamin
        if not gender_counts.empty:
            # Menghitung total peserta untuk persentase
            total_peserta = gender_counts.sum()

            # Membuat mapping label yang lebih deskriptif
            label_mapping = {'L': 'Laki-laki', 'P': 'Perempuan'}

            # Membuat dua kolom untuk visualisasi
            gender_col1, gender_col2 = st.columns([1, 1])

            with gender_col1:
                # Membuat figure dan axes untuk pie chart sederhana
                fig_gender, ax_gender = plt.subplots(figsize=(6, 5))

                # Membuat background chart transparan
                fig_gender.patch.set_alpha(0)
                ax_gender.patch.set_alpha(0)

                # Menyiapkan data untuk pie chart
                labels = [label_mapping.get(k, k) for k in gender_counts.index]
                sizes = gender_counts.values
                colors = ['#3498DB', '#E91E63']  # Biru untuk Laki-laki, Pink untuk Perempuan

                # Membuat pie chart sederhana
                ax_gender.pie(
                    sizes,
                    labels=labels,
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=colors[:len(sizes)]
                )

                ax_gender.set_title('Proporsi Jenis Kelamin')
                ax_gender.axis('equal')
                plt.tight_layout()
                st.pyplot(fig_gender, clear_figure=True)
                plt.close(fig_gender)

            with gender_col2:
                # Menampilkan detail jenis kelamin dalam bentuk info/kesimpulan
                st.markdown("##### Detail Jenis Kelamin")

                # Menghitung jumlah dan persentase untuk setiap jenis kelamin
                laki_count = gender_counts.get('L', 0)
                perempuan_count = gender_counts.get('P', 0)
                laki_pct = (laki_count / total_peserta * 100) if total_peserta > 0 else 0
                perempuan_pct = (perempuan_count / total_peserta * 100) if total_peserta > 0 else 0

                # Menampilkan total peserta
                st.info(f"**Total Peserta:** {total_peserta:,} orang")

                # Menampilkan detail laki-laki
                st.success(f"**Laki-laki:** {laki_count:,} orang ({laki_pct:.1f}%)")

                # Menampilkan detail perempuan
                st.error(f"**Perempuan:** {perempuan_count:,} orang ({perempuan_pct:.1f}%)")

                # Menampilkan kesimpulan    
                if laki_count > perempuan_count:
                    selisih = laki_count - perempuan_count
                    st.warning(f"**Kesimpulan:** Peserta laki-laki lebih banyak dengan selisih {selisih:,} orang")
                elif perempuan_count > laki_count:
                    selisih = perempuan_count - laki_count
                    st.warning(f"**Kesimpulan:** Peserta perempuan lebih banyak dengan selisih {selisih:,} orang")
                else:
                    st.warning(f"**Kesimpulan:** Jumlah peserta laki-laki dan perempuan seimbang")
        else:
            st.info("Data jenis kelamin tidak tersedia.")
    else:
        st.info("Kolom 'JK' tidak ditemukan dalam data.")

    # Membuat garis pemisah horizontal di akhir bagian insights
    st.markdown("---")

# ---------------- WORKSHOP PAGE ----------------
def render_workshop_page():
    """
    Fungsi untuk menampilkan halaman Workshop lengkap dengan:
    - Visualisasi insights (peta, grafik usia, pendidikan, tahun)
    - Filter data (lokasi, tanggal, pencarian teks)
    - Preview data dalam bentuk tabel dengan pagination
    """

    # Menampilkan judul halaman dengan warna kustom menggunakan HTML
    # unsafe_allow_html=True memungkinkan penggunaan kode HTML
    st.markdown('<h1 style="color:#051726">Workshop</h1>', unsafe_allow_html=True)

    # Mengambil data workshop dari session state
    # session_state menyimpan data yang persisten selama sesi aplikasi berjalan
    df = st.session_state.get("df_workshop", pd.DataFrame()).copy()

    # ============ INPUT DATA BARU WORKSHOP ============
    with st.expander("➕ Tambah Data Workshop Baru", expanded=False):
        with st.form("form_tambah_workshop", clear_on_submit=True):
            col_in1, col_in2 = st.columns(2)
            
            with col_in1:
                in_tgl = st.date_input("Tanggal", value=pd.Timestamp.now())
                in_nama = st.text_input("Nama Peserta")
                in_lokasi = st.text_input("Lokasi (Kab/Kota)")
                in_kec = st.text_input("Kecamatan")
            
            with col_in2:
                in_pend = st.selectbox("Pendidikan", ["SD", "SMP", "SMA", "SMK", "D1", "D2", "D3", "S1", "S2", "S3", "Tidak Sekolah"])
                in_usia = st.number_input("Usia", min_value=15, max_value=80, value=20)
                in_jk = st.selectbox("Jenis Kelamin", ["L", "P"])
            
            submitted = st.form_submit_button("Simpan Data")
            
            if submitted:
                if not in_nama:
                    st.error("Nama wajib diisi!")
                else:
                    new_data = {
                        "tanggal": pd.to_datetime(in_tgl),
                        "TAHUN": in_tgl.year,
                        "NAMA": in_nama,
                        "LOKASI": in_lokasi,
                        "KEC": in_kec,
                        "PEND": in_pend,
                        "USIA": in_usia,
                        "JK": in_jk
                    }
                    
                    # Update session state
                    current_df = st.session_state.get("df_workshop", pd.DataFrame())
                    updated_df = pd.concat([current_df, pd.DataFrame([new_data])], ignore_index=True)
                    st.session_state["df_workshop"] = updated_df
                    st.success(f"Data {in_nama} berhasil ditambahkan!")
                    st.rerun()

    # Menampilkan informasi file sumber dan jumlah baris data
    # st.info(f"File sumber: **{WORKSHOP_CSV}** — total baris: {len(df)}")

    # ============ Bagian Filter ============
    st.markdown("### Filter Data")

    # Membuat salinan DataFrame untuk ditampilkan dan difilter
    df_display = df.copy()

    # Mencari dan mengkonversi kolom tanggal
    # Loop melalui berbagai kemungkinan nama kolom tanggal yang mungkin ada dalam data
    for cand in ["TGL_WORKSH", "TGL", "tanggal", "date", "tgl"]:
        # Mengecek apakah nama kolom kandidat ada dalam DataFrame
        if cand in df_display.columns:
            try:
                # Mencoba mengkonversi kolom menjadi format datetime
                # errors='coerce' akan mengubah nilai yang tidak valid menjadi NaT (Not a Time)
                df_display['tanggal'] = pd.to_datetime(df_display[cand], errors='coerce')
                break  # Keluar dari loop jika konversi berhasil
            except Exception:
                # Jika terjadi error, lanjut ke kolom kandidat berikutnya
                continue

    # Membuat tiga kolom untuk komponen filter (tanpa rentang tanggal)
    f1, f2, f3 = st.columns([2,1,2])

    # ===== Filter 1: Lokasi =====
    with f1:
        # Mengecek apakah kolom 'LOKASI' ada dalam data
        if 'LOKASI' in df_display.columns:
            # Membuat daftar pilihan lokasi
            # dropna() menghapus nilai kosong
            # unique() mengambil nilai unik
            # sorted() mengurutkan secara alfabetis
            opts = ["Semua"] + sorted(df_display['LOKASI'].dropna().astype(str).unique().tolist())

            # Membuat dropdown selectbox untuk memilih lokasi
            # key digunakan agar state filter tersimpan
            sel_lokasi = st.selectbox("LOKASI", opts, key="workshop_filter_lokasi")
        else:
            # Jika kolom LOKASI tidak ada, set default "Semua"
            sel_lokasi = "Semua"

    # ===== Filter 2: Tahun =====
    # Filter tahun memungkinkan user memilih tahun tertentu untuk mempersempit data
    # Filter ini akan mempengaruhi insights dan tampilan data
    with f2:
        # Mengecek apakah kolom tanggal ada dan memiliki data valid
        if 'tanggal' in df_display.columns and df_display['tanggal'].notna().any():
            # Mengambil daftar tahun unik dari kolom tanggal
            # .dt.year mengambil komponen tahun dari datetime
            # dropna() menghapus nilai NaT (Not a Time)
            # unique() mengambil nilai unik, sorted() mengurutkan
            tahun_list = sorted(df_display['tanggal'].dropna().dt.year.unique().tolist())

            # Membuat daftar pilihan tahun dengan "Semua" sebagai opsi pertama
            tahun_opts = ["Semua"] + [str(t) for t in tahun_list]

            # Membuat dropdown selectbox untuk memilih tahun
            sel_tahun = st.selectbox("TAHUN", tahun_opts, key="workshop_filter_tahun")
        else:
            # Jika tidak ada data tanggal, set default "Semua"
            sel_tahun = "Semua"

    # ===== Filter 3: Pencarian Teks =====
    with f3:
        # Membuat input teks untuk pencarian bebas
        # User bisa mencari berdasarkan nama, lokasi, atau keterangan
        q = st.text_input("Cari teks (nama / lokasi / keterangan)", key="workshop_filter_q")

    # ============ Menerapkan Filter ke Data ============
    # Membuat mask (filter boolean) yang awalnya semua True
    # Mask ini akan digunakan untuk memfilter baris DataFrame
    mask = pd.Series([True]*len(df_display))

    # ===== Menerapkan Filter Lokasi =====
    if sel_lokasi != "Semua" and 'LOKASI' in df_display.columns:
        # Operator &= berarti AND, hanya baris yang cocok dengan lokasi yang dipilih
        mask &= df_display['LOKASI'].astype(str) == sel_lokasi

    # ===== Menerapkan Filter Tahun =====
    # Filter tahun memfilter data berdasarkan tahun yang dipilih
    # Filter ini juga mempengaruhi insights karena diterapkan sebelum df_for_insights dibuat
    if sel_tahun != "Semua" and 'tanggal' in df_display.columns:
        # Mengkonversi sel_tahun (string) ke integer untuk perbandingan
        # .dt.year mengambil komponen tahun dari kolom datetime
        tahun_int = int(sel_tahun)
        mask &= df_display['tanggal'].dt.year == tahun_int

    # ============ Data untuk Insights (tanpa filter teks) ============
    # Data ini digunakan untuk insights agar tetap konsisten dan proporsional
    # Hanya menggunakan filter lokasi dan tahun
    df_for_insights = df_display[mask].reset_index(drop=True)

    # ===== Menerapkan Filter Pencarian Teks =====
    # Filter teks hanya diterapkan untuk tampilan tabel, tidak untuk insights
    if q:  # Jika ada input pencarian teks
        # Membuat mask boolean baru untuk pencarian teks
        text_mask = pd.Series([False] * len(df_display))
        # select_dtypes(include='object') memilih semua kolom bertipe teks
        for col in df_display.select_dtypes(include='object').columns:
            text_mask |= df_display[col].astype(str).str.contains(q, case=False, na=False)

        # Gabungkan mask pencarian teks dengan mask filter lainnya
        mask &= text_mask

    # ============ Data yang Sudah Difilter (termasuk filter teks) ============
    # Menerapkan mask untuk memfilter DataFrame
    # reset_index(drop=True) mengatur ulang nomor indeks dari 0
    # Data ini digunakan untuk menampilkan tabel
    df_filtered = df_display[mask].reset_index(drop=True)

    # filter tanggal juga, tanggal yang tidak valid (NaT) akan otomatis terfilter keluar
    # Menampilkan informasi jumlah baris setelah filter diterapkan
    # Menyusun informasi filter yang aktif untuk ditampilkan ke user
    filter_info_parts = []
    if sel_lokasi != "Semua":
        filter_info_parts.append(f"lokasi: **{sel_lokasi}**")
    if sel_tahun != "Semua":
        filter_info_parts.append(f"tahun: **{sel_tahun}**")

    # if q:
    #     # Jika ada pencarian teks, tampilkan info hasil pencarian
    #     filter_text = ", ".join(filter_info_parts) if filter_info_parts else "semua filter"
    #     st.info(f"Menampilkan **{len(df_filtered)} peserta** dari hasil pencarian '{q}' (total {len(df_for_insights)} peserta yang sesuai {filter_text})")
    # elif filter_info_parts:
    #     # Jika ada filter lokasi atau tahun yang aktif
    #     filter_text = ", ".join(filter_info_parts)
    #     st.info(f"Menampilkan data untuk {filter_text} — **{len(df_for_insights)}** peserta (dari {len(df)} total)")
    # else:
    #     # Menampilkan jumlah data yang konsisten dengan insight (df_for_insights)
    #     st.info(f"Menampilkan data untuk **semua lokasi dan tahun** — **{len(df_for_insights)}** peserta. Load data sebelum di cleaning")

    st.markdown("---")

    # ============ Menampilkan Insights dengan Data Terfilter (tanpa pencarian teks) ============
    # Insights menggunakan df_for_insights yang tidak terpengaruh oleh filter pencarian teks
    # Ini membuat insights tetap konsisten dan proporsional
    if not df_for_insights.empty:
        # Menampilkan bagian insights (visualisasi data) dengan data yang sudah difilter (lokasi, tahun & tanggal)
        # Parameter sel_tahun ditambahkan agar insights menampilkan informasi tahun yang dipilih
        render_workshop_insights(df_for_insights, sel_lokasi, sel_tahun)

    # ============ Menampilkan Tabel Data yang Sudah Difilter ============
    st.markdown("### Preview Data Workshop")

    # Menampilkan informasi jumlah baris setelah filter diterapkan
    st.write(f"Menampilkan {len(df_filtered)} baris (setelah filter).")

    # ===== Pagination (Pembagian Halaman) =====
    # Membuat dropdown untuk memilih jumlah baris per halaman
    # index=1 berarti pilihan default adalah 25 (index ke-1 dari list)
    per_page = st.selectbox("Baris per halaman", [10,25,50,100], index=1, key="workshop_perpage")

    # Menghitung total halaman yang dibutuhkan
    # (len(df_filtered)-1)//per_page + 1 adalah rumus untuk menghitung jumlah halaman
    # Operator // adalah pembagian integer (hasil dibulatkan ke bawah)
    # max(1, ...) memastikan minimal ada 1 halaman meskipun data kosong
    total_pages = max(1, (len(df_filtered)-1)//per_page + 1)

    # Membuat input angka untuk memilih halaman
    # min_value dan max_value membatasi input sesuai jumlah halaman yang ada
    page = st.number_input("Halaman", min_value=1, max_value=total_pages, value=1, key="workshop_page")

    # Menghitung indeks awal untuk data yang akan ditampilkan
    # Misalnya: halaman 1 mulai dari indeks 0, halaman 2 mulai dari indeks per_page, dst.
    start_idx = (page-1)*per_page

    # Menampilkan DataFrame dengan slice data sesuai halaman yang dipilih
    # iloc[start:end] mengambil baris dari indeks start sampai sebelum end
    st.dataframe(df_filtered.iloc[start_idx:start_idx+per_page])

# ---------------- Data pages (Jobfair) ----------------
# Bagian untuk menampilkan halaman data Jobfair

def data_page(target_csv, session_key, title_label):
    """
    Fungsi untuk menampilkan halaman data Jobfair dengan fitur:
    - Form input untuk menambah data peserta baru
    - Filter data (perusahaan, tahun, tanggal, pencarian teks)
    - Preview data dengan pagination

    Parameter:
    target_csv: Path file CSV sumber data
    session_key: Kunci untuk menyimpan data di session state
    title_label: Judul halaman yang akan ditampilkan
    """

    # Menampilkan judul halaman dengan styling HTML
    st.markdown(f'<h1 style="color:#051726">{title_label}</h1>', unsafe_allow_html=True)
    st.write()

    # Mengambil data dari session state
    # Jika tidak ada, akan mengembalikan DataFrame kosong
    df = st.session_state.get(session_key, pd.DataFrame()).copy()

    # Menampilkan informasi file sumber dan jumlah total baris data #
    st.info(f"File sumber: **{target_csv}** — total baris: {len(df)}")

    # ============ Form Input untuk Tambah Data Baru ============
    # DINONAKTIFKAN SEMENTARA: Bagian input/tambah data di-comment
    # st.markdown("### Tambah Data Peserta Jobfair")

    # DINONAKTIFKAN SEMENTARA: Seluruh bagian form input di-comment
    # Membuat expander (collapsible section) untuk form input
    # expanded=False berarti awalnya form tersembunyi
    # with st.expander("Klik untuk menambah data baru", expanded=False):
    #     # Membuat form input
    #     # clear_on_submit=True akan mengosongkan form setelah data berhasil disimpan
    #     with st.form(key="form_tambah_jobfair", clear_on_submit=True):
    #         # Membuat 3 kolom untuk mengelompokkan input field
    #         col_input1, col_input2, col_input3 = st.columns(3)

    #         # ===== Kolom 1: Data Dasar =====
    #         with col_input1:
    #             # Input tanggal jobfair
    #             input_tgl = st.date_input("Tanggal Jobfair", key="input_tgl_jobfair")

    #             # Input nama peserta
    #             input_nama = st.text_input("Nama Peserta", key="input_nama_jobfair")

    #             # Dropdown untuk memilih tingkat pendidikan
    #             input_pend = st.selectbox("Pendidikan", ["SMA", "SMK", "D3", "S1", "S2", "Lainnya"], key="input_pend_jobfair")

    #         # ===== Kolom 2: Data Akademik & Perusahaan =====
    #         with col_input2:
    #             # Input jurusan pendidikan
    #             input_jurusan = st.text_input("Jurusan", key="input_jurusan_jobfair")

    #             # Input nama perusahaan tujuan
    #             input_perusahaan = st.text_input("Perusahaan", key="input_perusahaan_jobfair")

    #             # Input posisi/jabatan yang dilamar
    #             input_jabatan = st.text_input("Jabatan", key="input_jabatan_jobfair")

    #         # ===== Kolom 3: Status & Tahun =====
    #         with col_input3:
    #             # Dropdown untuk status lamaran saat ini
    #             input_status = st.selectbox("Status", ["MELAMAR", "INTERVIEW", "DITERIMA", "DITOLAK"], key="input_status_jobfair")

    #             # Dropdown untuk status akhir lamaran
    #             input_status_final = st.selectbox("Status Final", ["TIDAK TERDETEKSI", "DITERIMA", "DITOLAK", "PENDING"], key="input_status_final_jobfair")

    #             # Input angka untuk tahun jobfair
    #             # min_value dan max_value membatasi range tahun yang bisa dipilih
    #             input_tahun = st.number_input("Tahun", min_value=2020, max_value=2030, value=2025, key="input_tahun_jobfair")

    #         # Tombol submit form
    #         # width='stretch' membuat tombol memenuhi lebar form
    #         submit_btn = st.form_submit_button("Simpan Data", width='stretch')

    #         # ===== Proses Penyimpanan Data =====
    #         # Mengecek apakah tombol submit diklik
    #         if submit_btn:
    #             # Validasi: nama peserta tidak boleh kosong
    #             # strip() menghapus spasi di awal dan akhir string
    #             if input_nama.strip() == "":
    #                 st.error("Nama peserta tidak boleh kosong!")
    #             else:
    #                 # Membuat dictionary data baru sesuai struktur kolom Jobfair
    #                 new_data = {
    #                     "TGL_JOBFAIR": input_tgl.strftime("%Y-%m-%d"),  # Format tanggal ke string YYYY-MM-DD
    #                     "TAHUN": int(input_tahun),  # Konversi ke integer
    #                     "NAMA PESERTA": input_nama.strip(),  # Nama tanpa spasi berlebih
    #                     "PEND": input_pend,  # Tingkat pendidikan
    #                     "JURUSAN": input_jurusan.strip(),  # Jurusan tanpa spasi berlebih
    #                     "STATUS": input_status,  # Status lamaran
    #                     "PERUSAHAAN": input_perusahaan.strip(),  # Nama perusahaan tanpa spasi
    #                     "JABATAN": input_jabatan.strip(),  # Jabatan tanpa spasi berlebih
    #                     "STATUS_FINAL": input_status_final  # Status akhir lamaran
    #                 }

    #                 # Menambahkan data baru ke DataFrame yang ada di session state
    #                 # Langkah 1: Buat DataFrame baru dari dictionary (1 baris)
    #                 new_row = pd.DataFrame([new_data])

    #                 # Langkah 2: Gabungkan dengan DataFrame yang sudah ada
    #                 # pd.concat() menggabungkan dua DataFrame secara vertikal
    #                 # ignore_index=True akan membuat ulang nomor indeks dari 0
    #                 st.session_state[session_key] = pd.concat([st.session_state[session_key], new_row], ignore_index=True)

    #                 # Menampilkan pesan sukses
    #                 st.success(f"Data peserta **{input_nama}** berhasil ditambahkan!")

    #                 # Memuat ulang halaman agar data baru langsung terlihat
    #                 st.rerun()

    # Membuat garis pemisah horizontal
    # st.markdown("---")

    # ============ Bagian Filter & Preview Data ============
    st.markdown("### Filter & Preview")

    # Mengambil data terbaru dari session state
    # Data bisa berubah jika ada penambahan data baru dari form
    df_display = st.session_state.get(session_key, pd.DataFrame()).copy()

    # Mencari dan mengkonversi kolom tanggal
    # Loop melalui berbagai kemungkinan nama kolom tanggal dalam data Jobfair
    for cand in ["TGL_JOBFAIR", "TGL", "tanggal", "date", "tgl"]:
        # Mengecek apakah nama kolom kandidat ada dalam DataFrame
        if cand in df_display.columns:
            try:
                # Mencoba mengkonversi kolom menjadi format datetime
                # errors='coerce' akan mengubah nilai yang tidak valid menjadi NaT (Not a Time)
                df_display['tanggal'] = pd.to_datetime(df_display[cand], errors='coerce')
                break  # Keluar dari loop jika konversi berhasil
            except Exception:
                # Jika terjadi error, lanjut ke kolom kandidat berikutnya
                continue

    # Membuat 4 kolom dengan lebar sama untuk komponen filter
    f1, f2, f3, f4 = st.columns([2, 2, 2, 2])

    # ===== Filter 1: Perusahaan =====
    with f1:
        # Mengecek apakah kolom 'PERUSAHAAN' ada dalam data
        if 'PERUSAHAAN' in df_display.columns:
            # Membuat daftar pilihan perusahaan
            # dropna() menghapus nilai kosong
            # unique() mengambil nilai unik
            # sorted() mengurutkan secara alfabetis
            opts_perusahaan = ["Semua"] + sorted(df_display['PERUSAHAAN'].dropna().astype(str).unique().tolist())

            # Membuat dropdown selectbox untuk memilih perusahaan
            # key menggunakan session_key agar unik untuk setiap halaman
            sel_perusahaan = st.selectbox("Perusahaan", opts_perusahaan, key=f"{session_key}_filter_perusahaan")
        else:
            # Jika kolom PERUSAHAAN tidak ada, set default "Semua"
            sel_perusahaan = "Semua"

    # ===== Filter 2: Tahun =====
    with f2:
        # Mengecek apakah kolom 'TAHUN' ada dalam data
        if 'TAHUN' in df_display.columns:
            # Membuat daftar pilihan tahun
            opts_tahun = ["Semua"] + sorted(df_display['TAHUN'].dropna().astype(str).unique().tolist())
            sel_tahun = st.selectbox("Tahun", opts_tahun, key=f"{session_key}_filter_tahun")
        else:
            # Jika kolom TAHUN tidak ada, set default "Semua"
            sel_tahun = "Semua"

    # ===== Filter 3: Rentang Tanggal =====
    with f3:
        # Mengecek apakah kolom tanggal ada dan memiliki data valid
        if 'tanggal' in df_display.columns and df_display['tanggal'].notna().any():
            # Mencari tanggal minimum dan maksimum dalam data
            min_d = df_display['tanggal'].min().date()
            max_d = df_display['tanggal'].max().date()

            # Membuat date input dengan rentang tanggal default
            # User bisa memilih rentang tanggal untuk filter
            dr = st.date_input("Rentang tanggal", [min_d, max_d], key=f"{session_key}_filter_date")
        else:
            # Jika tidak ada data tanggal, set None
            dr = None

    # ===== Filter 4: Pencarian Teks =====
    with f4:
        # Membuat input teks untuk pencarian bebas
        # User bisa mencari berdasarkan nama, perusahaan, atau jabatan
        q = st.text_input("Cari teks (nama / perusahaan / jabatan)", key=f"{session_key}_filter_q")

    # ============ Menerapkan Semua Filter ke Data ============
    # Membuat mask (filter boolean) yang awalnya semua True
    # Mask ini akan digunakan untuk memfilter baris DataFrame
    mask = pd.Series([True]*len(df_display))

    # ===== Menerapkan Filter Perusahaan =====
    if sel_perusahaan != "Semua" and 'PERUSAHAAN' in df_display.columns:
        # Operator &= berarti AND, hanya baris yang cocok dengan perusahaan yang dipilih
        mask &= df_display['PERUSAHAAN'].astype(str) == sel_perusahaan

    # ===== Menerapkan Filter Tahun =====
    if sel_tahun != "Semua" and 'TAHUN' in df_display.columns:
        # Filter tahun memfilter data berdasarkan tahun yang dipilih
        # Filter ini juga mempengaruhi insights karena diterapkan sebelum df_for_insights dibuat
        mask &= df_display['TAHUN'].astype(str) == sel_tahun

    # ===== Menerapkan Filter Rentang Tanggal =====
    if dr is not None and 'tanggal' in df_display.columns:
        try:
            # Mengecek apakah dr adalah tuple/list dengan 2 elemen (rentang tanggal)
            if isinstance(dr, (list, tuple)) and len(dr) == 2:
                # Mengkonversi tanggal awal yang dipilih user
                start = pd.to_datetime(dr[0])
                # Mengkonversi tanggal akhir dan menambahkan 1 hari lalu kurangi 1 detik
                # Ini memastikan seluruh hari terakhir termasuk dalam filter (sampai 23:59:59)
                end = pd.to_datetime(dr[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                # Memfilter data yang tanggalnya berada dalam rentang start-end
                # between() mengecek apakah nilai berada di antara dua nilai
                mask &= df_display['tanggal'].between(start, end)
            elif isinstance(dr, (list, tuple)) and len(dr) == 1:
                # Jika user hanya memilih 1 tanggal, filter berdasarkan tanggal tersebut
                selected_date = pd.to_datetime(dr[0])
                mask &= df_display['tanggal'].dt.date == selected_date.date()
            else:
                # Jika dr adalah single date object
                selected_date = pd.to_datetime(dr)
                mask &= df_display['tanggal'].dt.date == selected_date.date()
        except (IndexError, TypeError, AttributeError) as e:
            # Menangkap error jika user iseng mengubah rentang tanggal
            st.warning(f"⚠️ Format tanggal tidak valid. Silakan pilih rentang tanggal yang benar.")
            # Skip filter tanggal jika terjadi error
            pass

    # ============ Data untuk Insights (tanpa filter teks) ============
    # Data ini digunakan untuk insights agar tetap konsisten dan proporsional
    # Hanya menggunakan filter perusahaan, tahun, dan tanggal
    df_for_insights = df_display[mask].reset_index(drop=True)

    # ===== Menerapkan Filter Pencarian Teks =====
    # Filter teks hanya diterapkan untuk tampilan tabel, tidak untuk insights
    if q:  # Jika ada input pencarian teks
        # Membuat mask boolean baru untuk pencarian teks
        text_mask = pd.Series([False] * len(df_display))
        # select_dtypes(include='object') memilih semua kolom bertipe teks
        for col in df_display.select_dtypes(include='object').columns:
            text_mask |= df_display[col].astype(str).str.contains(q, case=False, na=False)

        # Gabungkan mask pencarian teks dengan mask filter lainnya
        mask &= text_mask
    # ============ Data yang Sudah Difilter (termasuk filter teks) ============
    # Menerapkan mask untuk memfilter DataFrame
    # reset_index(drop=True) mengatur ulang nomor indeks dari 0
    # Data ini digunakan untuk menampilkan tabel
    df_filtered = df_display[mask].reset_index(drop=True)

    # filter tanggal juga, tanggal yang tidak valid (NaT) akan otomatis terfilter keluar
    # Menampilkan informasi jumlah baris setelah filter diterapkan
    # Menyusun informasi filter yang aktif untuk ditampilkan ke user
    filter_info_parts = []
    if sel_perusahaan != "Semua":
        filter_info_parts.append(f"perusahaan: **{sel_perusahaan}**")
    if sel_tahun != "Semua":
        filter_info_parts.append(f"tahun: **{sel_tahun}**")

    if q:
        # Jika ada pencarian teks, tampilkan info hasil pencarian
        filter_text = ", ".join(filter_info_parts) if filter_info_parts else "semua filter"
        st.info(f"Menampilkan **{len(df_filtered)} peserta** dari hasil pencarian '{q}' (total {len(df_for_insights)} peserta yang sesuai {filter_text})")
    elif filter_info_parts:
        # Jika ada filter lokasi atau tahun yang aktif
        filter_text = ", ".join(filter_info_parts)
        st.info(f"Menampilkan data untuk {filter_text} — {len(df_filtered)} peserta (dari {len(df)} total)")
    else:
        st.info(f"Menampilkan data untuk **semua lokasi dan tahun** — {len(df_filtered)} peserta")

    st.markdown("---")

    # ============ Menampilkan Insights dengan Data Terfilter (tanpa pencarian teks) ============
    # Insights menggunakan df_for_insights yang tidak terpengaruh oleh filter pencarian teks
    # Ini membuat insights tetap konsisten dan proporsional
    if not df_for_insights.empty:
        # Menampilkan bagian insights (visualisasi data) dengan data yang sudah difilter (lokasi, tahun & tanggal)
        # Parameter sel_tahun ditambahkan agar insights menampilkan informasi tahun yang dipilih
        render_workshop_insights(df_for_insights, sel_perusahaan, sel_tahun)

    # ============ Menampilkan Tabel Data yang Sudah Difilter ============
    st.markdown("### Preview Data Jobfair")

    # Menampilkan informasi jumlah baris setelah filter diterapkan
    st.write(f"Menampilkan {len(df_filtered)} baris (setelah filter).")

    # ===== Pagination (Pembagian Halaman) =====
    # Membuat dropdown untuk memilih jumlah baris per halaman
    # index=1 berarti pilihan default adalah 25 (index ke-1 dari list)
    per_page = st.selectbox("Baris per halaman", [10,25,50,100], index=1, key="jobfair_perpage")

    # Menghitung total halaman yang dibutuhkan
    # (len(df_filtered)-1)//per_page + 1 adalah rumus untuk menghitung jumlah halaman
    # Operator // adalah pembagian integer (hasil dibulatkan ke bawah)
    # max(1, ...) memastikan minimal ada 1 halaman meskipun data kosong
    total_pages = max(1, (len(df_filtered)-1)//per_page + 1)

    # Membuat input angka untuk memilih halaman
    # min_value dan max_value membatasi input sesuai jumlah halaman yang ada
    page = st.number_input("Halaman", min_value=1, max_value=total_pages, value=1, key="jobfair_page")

    # Menghitung indeks awal untuk data yang akan ditampilkan
    # Misalnya: halaman 1 mulai dari indeks 0, halaman 2 mulai dari indeks per_page, dst.
    start_idx = (page-1)*per_page

    # Menampilkan DataFrame dengan slice data sesuai halaman yang dipilih
    # iloc[start:end] mengambil baris dari indeks start sampai sebelum end
    st.dataframe(df_filtered.iloc[start_idx:start_idx+per_page])


# ---------------- JOBFAIR PAGE (Menampilkan 2 Dataset) ----------------
# Halaman Jobfair yang menampilkan 2 dataset:
# 1. Data Jobfair Baru (JOBFAIR 2023-2025 BARU.csv) - dengan kolom tambahan JK, TTL, USIA, KOTA
# 2. Data Jobfair Lama (JOBFAIR HASIL 2023 - 2025.csv) - data hasil sebelumnya

def render_jobfair_page():
    """
    Fungsi untuk menampilkan halaman Jobfair.
    MODIFIED: Hanya menampilkan data Jobfair Baru (JOBFAIR 2023-2025 BARU.csv).
    Bagian People Analytics (tab lama) disembunyikan sesuai permintaan.
    """

    # Menampilkan judul halaman
    st.markdown('<h1 style="color:#051726">Data Jobfair</h1>', unsafe_allow_html=True)

    st.markdown("### Data Jobfair")
    st.caption(f"Sumber: **{JOBFAIR_BARU_CSV}** — Kolom tambahan: JK, TTL, USIA, KOTA, HASIL")

    # Mengambil data dari session state
    df_display_baru = st.session_state.get("df_jobfair_baru", pd.DataFrame()).copy()

    # ============ INPUT DATA BARU JOBFAIR ============
    with st.expander("➕ Tambah Data Jobfair Baru", expanded=False):
        with st.form("form_tambah_jobfair", clear_on_submit=True):
            col_jf1, col_jf2 = st.columns(2)
            
            with col_jf1:
                jf_tgl = st.date_input("Tanggal", value=pd.Timestamp.now())
                jf_nama = st.text_input("Nama Peserta")
                jf_perusahaan = st.text_input("Perusahaan")
                jf_jabatan = st.text_input("Jabatan")
                jf_kota = st.text_input("Kota Domisili")
            
            with col_jf2:
                jf_pend = st.selectbox("Pendidikan", ["SMA", "SMK", "D3", "S1", "S2", "Lainnya"])
                jf_jurusan = st.text_input("Jurusan")
                jf_usia = st.number_input("Usia", min_value=17, max_value=60, value=22)
                jf_jk = st.selectbox("Jenis Kelamin", ["L", "P"])
                jf_hasil = st.selectbox("Hasil", ["DITERIMA", "DITOLAK", "PENDING", "CADANGAN"])
            
            submitted_jf = st.form_submit_button("Simpan Data")
            
            if submitted_jf:
                if not jf_nama:
                    st.error("Nama wajib diisi!")
                else:
                    new_data_jf = {
                        "tanggal": pd.to_datetime(jf_tgl),
                        "TAHUN": jf_tgl.year,
                        "NAMA": jf_nama,
                        "PERUSAHAAN": jf_perusahaan,
                        "JABATAN": jf_jabatan,
                        "PEND": jf_pend,
                        "JURUSAN": jf_jurusan,
                        "KOTA": jf_kota,
                        "USIA": jf_usia,
                        "JK": jf_jk,
                        "HASIL": jf_hasil,
                        # Kolom tambahan agar kompatibel dengan data existing
                        "TGL_JOBFAIR": jf_tgl.strftime("%Y-%m-%d")
                    }
                    
                    # Update session state
                    current_df_jf = st.session_state.get("df_jobfair_baru", pd.DataFrame())
                    updated_df_jf = pd.concat([current_df_jf, pd.DataFrame([new_data_jf])], ignore_index=True)
                    st.session_state["df_jobfair_baru"] = updated_df_jf
                    st.success(f"Data {jf_nama} berhasil ditambahkan!")
                    st.rerun()

    if df_display_baru.empty:
        st.warning("Data Jobfair belum tersedia atau file tidak ditemukan.")
    else:
        # ============ FILTER DATA (BERLAKU UNTUK INSIGHT DAN TABEL DATA) ============
        # Filter ini akan mempengaruhi insight dan data yang ditampilkan di tabel
        st.markdown("### Filter Data")

        # Konversi kolom tanggal terlebih dahulu
        for cand in ["TGL_JOBFAIR", "TGL", "tanggal"]:
            if cand in df_display_baru.columns:
                try:
                    df_display_baru['tanggal'] = pd.to_datetime(df_display_baru[cand], errors='coerce')
                    break
                except:
                    continue

        # Membuat 4 kolom untuk filter
        fb1, fb2, fb3, fb4 = st.columns([2, 2, 2, 2])

        # Filter Tahun
        with fb1:
            if 'TAHUN' in df_display_baru.columns:
                opts_thn_b = ["Semua"] + sorted(df_display_baru['TAHUN'].dropna().astype(str).unique().tolist())
                sel_thn_b = st.selectbox("Tahun", opts_thn_b, key="jf_baru_thn")
            else:
                sel_thn_b = "Semua"

        # Filter Pendidikan
        with fb2:
            if 'PEND' in df_display_baru.columns:
                opts_pend_b = ["Semua"] + sorted(df_display_baru['PEND'].dropna().astype(str).unique().tolist())
                sel_pend_b = st.selectbox("Pendidikan", opts_pend_b, key="jf_baru_pend")
            else:
                sel_pend_b = "Semua"

        # Filter Hasil
        with fb3:
            if 'HASIL' in df_display_baru.columns:
                opts_hasil = ["Semua"] + sorted(df_display_baru['HASIL'].dropna().astype(str).unique().tolist())
                sel_hasil = st.selectbox("Hasil", opts_hasil, key="jf_baru_hasil")
            else:
                sel_hasil = "Semua"

        # Pencarian Teks
        with fb4:
            q_baru = st.text_input("Cari (nama/perusahaan/jabatan)", key="jf_baru_q")

        # Menerapkan filter ke data
        df_filtered = df_display_baru.copy()

        if sel_thn_b != "Semua" and 'TAHUN' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['TAHUN'].astype(str) == sel_thn_b]

        if sel_pend_b != "Semua" and 'PEND' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['PEND'].astype(str) == sel_pend_b]

        if sel_hasil != "Semua" and 'HASIL' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['HASIL'].astype(str) == sel_hasil]

        # Simpan data sebelum filter pencarian untuk insight
        df_for_insight = df_filtered.copy()

        # Filter pencarian teks (hanya untuk tabel)
        if q_baru:
            text_mask = pd.Series([False] * len(df_filtered))
            for col in ['NAMA', 'PERUSAHAAN', 'JABATAN', 'JURUSAN', 'KOTA']:
                if col in df_filtered.columns:
                    text_mask |= df_filtered[col].astype(str).str.contains(q_baru, case=False, na=False)
            df_filtered = df_filtered[text_mask]

        # Menampilkan info filter aktif
        filter_info_jf = []
        if sel_thn_b != "Semua":
            filter_info_jf.append(f"Tahun: **{sel_thn_b}**")
        if sel_pend_b != "Semua":
            filter_info_jf.append(f"Pendidikan: **{sel_pend_b}**")
        if sel_hasil != "Semua":
            filter_info_jf.append(f"Hasil: **{sel_hasil}**")

        if filter_info_jf:
            st.info(f"Filter aktif: {', '.join(filter_info_jf)} — Data: **{len(df_for_insight):,}** dari **{len(df_display_baru):,}**")
        else:
            st.info(f"Menampilkan **semua data**: **{len(df_for_insight):,}** peserta")

        st.markdown("---")

        # ============ INSIGHT INTERAKTIF (BERDASARKAN FILTER) ============
        st.markdown("### Insights Jobfair")
        if filter_info_jf:
            st.caption(f"Berikut adalah beberapa insights berdasarkan filter yang aktif")

        # ----- ROW 1: Jumlah Peserta per Tahun & Rata-rata Usia -----
        col_ins1, col_ins2 = st.columns(2)

        # INSIGHT 1: Jumlah Peserta per Tahun & Rata-rata Usia
        with col_ins1:
            st.markdown("##### Peserta per Tahun & Rata-rata Usia")

            if 'TAHUN' in df_for_insight.columns and len(df_for_insight) > 0:
                # Konversi USIA ke numerik
                df_for_chart = df_for_insight.copy()
                if 'USIA' in df_for_chart.columns:
                    df_for_chart['USIA'] = pd.to_numeric(df_for_chart['USIA'], errors='coerce')

                # Hitung jumlah peserta per tahun
                jumlah_peserta = df_for_chart.groupby('TAHUN').size().reset_index(name='Jumlah')

                # Hitung rata-rata usia per tahun
                if 'USIA' in df_for_chart.columns:
                    usia_rata = df_for_chart.groupby('TAHUN')['USIA'].mean().reset_index()
                    usia_rata.columns = ['TAHUN', 'RATA_USIA']
                    import math
                    usia_rata['RATA_USIA'] = usia_rata['RATA_USIA'].apply(lambda x: math.ceil(x) if pd.notna(x) else 0).astype(int)
                    plot_df = jumlah_peserta.merge(usia_rata, on='TAHUN', how='left')
                else:
                    plot_df = jumlah_peserta.copy()
                    plot_df['RATA_USIA'] = 0

                # Membuat bar chart
                fig1, ax1 = plt.subplots(figsize=(6, 4))
                fig1.patch.set_alpha(0)
                ax1.patch.set_alpha(0)

                bars = ax1.bar(plot_df['TAHUN'].astype(str), plot_df['Jumlah'], color='#4C72B0')
                ax1.set_xlabel('Tahun', fontsize=9)
                ax1.set_ylabel('Jumlah', fontsize=9)

                # Tambahkan angka pada tiap bar
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    usia = plot_df['RATA_USIA'].iloc[i] if plot_df['RATA_USIA'].iloc[i] > 0 else 0
                    ax1.text(bar.get_x() + bar.get_width()/2, height + 5,
                            f"{int(height):,}", ha='center', fontsize=9, fontweight='bold')
                    if usia > 0:
                        ax1.text(bar.get_x() + bar.get_width()/2, height * 0.5,
                                f"Usia:{usia}", ha='center', fontsize=8, color='white', fontweight='bold')

                plt.tight_layout()
                st.pyplot(fig1)
                plt.close(fig1)

                # Ringkasan singkat
                total_peserta = plot_df['Jumlah'].sum()
                rata_usia = df_for_chart['USIA'].mean() if 'USIA' in df_for_chart.columns else 0
                st.caption(f"Total: **{total_peserta:,}** peserta | Rata-rata usia: **{rata_usia:.1f}** tahun")
            else:
                st.info("Data tidak tersedia untuk filter ini.")

        # INSIGHT 2: Top 5 Jabatan
        with col_ins2:
            st.markdown("##### Top 5 Jabatan")

            if 'JABATAN' in df_for_insight.columns and len(df_for_insight) > 0:
                top5_jabatan = df_for_insight['JABATAN'].value_counts().head(5)

                if not top5_jabatan.empty:
                    fig2, ax2 = plt.subplots(figsize=(6, 4))
                    fig2.patch.set_alpha(0)
                    ax2.patch.set_alpha(0)

                    colors_jab = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974']
                    top5_sorted = top5_jabatan.sort_values()

                    # Potong nama jabatan yang panjang
                    labels = [textwrap.shorten(str(j), width=25, placeholder="...") for j in top5_sorted.index]
                    bars = ax2.barh(labels, top5_sorted.values, color=colors_jab[:len(top5_sorted)])

                    xmax = top5_sorted.values.max()
                    ax2.set_xlim(0, xmax * 1.2)

                    for bar in bars:
                        width = int(bar.get_width())
                        ax2.text(width + xmax * 0.02, bar.get_y() + bar.get_height()/2,
                                f'{width:,}', va='center', fontsize=8)

                    ax2.set_xlabel('Jumlah Pelamar', fontsize=9)
                    plt.tight_layout()
                    st.pyplot(fig2)
                    plt.close(fig2)
                else:
                    st.info("Tidak ada data jabatan.")
            else:
                st.info("Data tidak tersedia untuk filter ini.")

        st.markdown("")

        # ----- ROW 2: Top 10 Perusahaan -----
        st.markdown("##### Top 10 Perusahaan")

        if 'PERUSAHAAN' in df_for_insight.columns and len(df_for_insight) > 0:
            top10_peru = df_for_insight['PERUSAHAAN'].value_counts().head(10)

            if not top10_peru.empty:
                fig3, ax3 = plt.subplots(figsize=(10, 4))
                fig3.patch.set_alpha(0)
                ax3.patch.set_alpha(0)

                top10_sorted = top10_peru.sort_values()
                labels = [textwrap.shorten(str(p), width=25, placeholder="...") for p in top10_sorted.index]
                bars = ax3.barh(labels, top10_sorted.values, color='#4C72B0')

                max_val = top10_sorted.values.max()
                ax3.set_xlim(0, max_val * 1.15)

                for bar in bars:
                    width = bar.get_width()
                    ax3.text(width + max_val * 0.01, bar.get_y() + bar.get_height()/2,
                            f'{int(width):,}', va='center', fontsize=8)

                ax3.set_xlabel('Jumlah Pelamar', fontsize=9)
                ax3.grid(axis='x', linestyle='--', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig3)
                plt.close(fig3)
            else:
                st.info("Tidak ada data perusahaan.")
        else:
            st.info("Data tidak tersedia untuk filter ini.")

        st.markdown("---")

        # ============ PREVIEW DATA ============
        st.markdown("### Preview Data")
        if q_baru:
            st.write(f"Hasil pencarian '{q_baru}': **{len(df_filtered)}** data ditemukan")
        else:
            st.write(f"Menampilkan **{len(df_filtered):,}** data")

        # Pagination
        per_page_b = st.selectbox("Baris per halaman", [10, 25, 50, 100], index=1, key="jf_baru_perpage")
        total_pages_b = max(1, (len(df_filtered) - 1) // per_page_b + 1)
        page_b = st.number_input("Halaman", min_value=1, max_value=total_pages_b, value=1, key="jf_baru_page")
        start_b = (page_b - 1) * per_page_b

        # Menampilkan tabel data
        st.dataframe(df_filtered.iloc[start_b:start_b + per_page_b], width='stretch')


# ---------------- GABUNGAN PAGE ----------------
# Bagian untuk menampilkan halaman gabungan Workshop dan Jobfair

def render_gabungan_page():
    """
    Fungsi untuk menampilkan halaman Data Gabungan
    Menampilkan data peserta yang mengikuti KEDUA kegiatan (Workshop DAN Jobfair)

    Data diambil dari file: GABUNGAN_JOBFAIR_WORKSHOP.csv

    Kolom yang tersedia:
    - TAHUN: Tahun kegiatan
    - TANGGAL_WORKSHOP: Tanggal mengikuti workshop
    - TANGGAL_JOBFAIR: Tanggal melamar di jobfair
    - NAMA: Nama peserta
    - JENIS_KELAMIN: L/P
    - USIA: Usia peserta
    - JURUSAN: Jurusan pendidikan
    - PENDIDIKAN: Tingkat pendidikan
    - KECAMATAN: Kecamatan tempat tinggal
    - STATUS_WORKSHOP: Status peserta di workshop
    - STATUS_JOBFAIR: Status lamaran (DITERIMA/DITOLAK/TIDAK TERDETEKSI)
    - PERUSAHAAN: Perusahaan yang dilamar
    - JABATAN: Jabatan yang dilamar
    """

    # Menampilkan judul halaman dengan styling HTML
    st.markdown('<h1 style="color:#051726">Data Gabungan Workshop & Jobfair</h1>', unsafe_allow_html=True)

    # ============ Load Data Gabungan ============
    GABUNGAN_FILE = "GABUNGAN_JOBFAIR_WORKSHOP.csv"

    try:
        df_gabungan = pd.read_csv(GABUNGAN_FILE)
    except FileNotFoundError:
        st.error(f"File {GABUNGAN_FILE} tidak ditemukan! Silakan jalankan script merge_gabungan_keduanya.py terlebih dahulu.")
        return
    except Exception as e:
        st.error(f"Error membaca file: {e}")
        return

    # ============ Preprocessing Data ============
    df_display = df_gabungan.copy()

    # Konversi TANGGAL_WORKSHOP ke datetime dan ekstrak tahun workshop
    if 'TANGGAL_WORKSHOP' in df_display.columns:
        df_display['tgl_workshop_dt'] = pd.to_datetime(df_display['TANGGAL_WORKSHOP'], errors='coerce')
        df_display['TAHUN_WORKSHOP'] = df_display['tgl_workshop_dt'].dt.year

    if 'TANGGAL_JOBFAIR' in df_display.columns:
        df_display['tgl_jobfair_dt'] = pd.to_datetime(df_display['TANGGAL_JOBFAIR'], errors='coerce')

    # Normalisasi STATUS_WORKSHOP sesuai ipynb
    if 'STATUS_WORKSHOP' in df_display.columns:
        df_display['STATUS_WORKSHOP'] = df_display['STATUS_WORKSHOP'].replace('HADIR', 'TIDAK ADA STATUS')
        df_display['STATUS_WORKSHOP'] = df_display['STATUS_WORKSHOP'].replace({
            'NON KELUARGA MISKIN': 'KELUARGA NON MISKIN'
        })

    # Konversi USIA ke numeric
    if 'USIA' in df_display.columns:
        df_display['USIA'] = pd.to_numeric(df_display['USIA'], errors='coerce')

    # ============ FILTER DATA ============
    st.markdown("### Filter Data")

    # Membuat 4 kolom untuk filter
    f1, f2, f3, f4 = st.columns(4)

    # Filter Tahun
    with f1:
        if 'TAHUN' in df_display.columns:
            opts_tahun = ["Semua"] + sorted(df_display['TAHUN'].dropna().astype(str).unique().tolist())
            sel_tahun = st.selectbox("Tahun", opts_tahun, key="gab_filter_tahun")
        else:
            sel_tahun = "Semua"

    # Filter Pendidikan
    with f2:
        if 'PENDIDIKAN' in df_display.columns:
            opts_pend = ["Semua"] + sorted(df_display['PENDIDIKAN'].dropna().astype(str).unique().tolist())
            sel_pend = st.selectbox("Pendidikan", opts_pend, key="gab_filter_pend")
        else:
            sel_pend = "Semua"

    # Filter Jenis Kelamin
    with f3:
        if 'JENIS_KELAMIN' in df_display.columns:
            opts_jk = ["Semua"] + sorted(df_display['JENIS_KELAMIN'].dropna().astype(str).unique().tolist())
            sel_jk = st.selectbox("Jenis Kelamin", opts_jk, key="gab_filter_jk")
        else:
            sel_jk = "Semua"

    # Pencarian Teks
    with f4:
        q_search = st.text_input("Cari (nama/jurusan)", key="gab_filter_search")

    # ============ Menerapkan Filter ============
    df_filtered = df_display.copy()

    if sel_tahun != "Semua" and 'TAHUN' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['TAHUN'].astype(str) == sel_tahun]

    if sel_pend != "Semua" and 'PENDIDIKAN' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['PENDIDIKAN'].astype(str) == sel_pend]

    if sel_jk != "Semua" and 'JENIS_KELAMIN' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['JENIS_KELAMIN'].astype(str) == sel_jk]

    # Simpan data untuk insight sebelum filter pencarian
    df_for_insight = df_filtered.copy()

    # Filter pencarian teks (hanya untuk tabel)
    if q_search:
        text_mask = pd.Series([False] * len(df_filtered))
        for col in ['NAMA', 'JURUSAN', 'JABATAN', 'PERUSAHAAN', 'KECAMATAN']:
            if col in df_filtered.columns:
                text_mask |= df_filtered[col].astype(str).str.contains(q_search, case=False, na=False)
        df_filtered = df_filtered[text_mask]

    # Menampilkan info filter aktif
    filter_info = []
    if sel_tahun != "Semua":
        filter_info.append(f"Tahun: **{sel_tahun}**")
    if sel_pend != "Semua":
        filter_info.append(f"Pendidikan: **{sel_pend}**")
    if sel_jk != "Semua":
        filter_info.append(f"JK: **{sel_jk}**")

    if filter_info:
        st.info(f"Filter aktif: {', '.join(filter_info)} — Data: **{len(df_for_insight):,}** dari **{len(df_display):,}**")
    else:
        st.info(f"Menampilkan **semua data**: **{len(df_for_insight):,}** data gabungan")

    st.markdown("---")

    # ============ INSIGHT INTERAKTIF ============
    st.markdown("### Insights")
    if filter_info:
        st.caption(f"Insight berdasarkan filter: {', '.join(filter_info)}")

    # ----- ROW 1: Distribusi Usia & Status Keluarga -----
    col_ins1, col_ins2 = st.columns(2)

    # INSIGHT 1: Distribusi Usia Berdasarkan Jenis Kelamin
    with col_ins1:
        st.markdown("##### Distribusi Usia per Jenis Kelamin")

        if 'USIA' in df_for_insight.columns and 'JENIS_KELAMIN' in df_for_insight.columns and len(df_for_insight) > 0:
            # Hitung jumlah usia berdasarkan jenis kelamin
            usia_gender = df_for_insight.groupby(['USIA', 'JENIS_KELAMIN']).size().unstack(fill_value=0)
            usia_gender = usia_gender.sort_index()

            if not usia_gender.empty:
                fig1, ax1 = plt.subplots(figsize=(8, 4))
                fig1.patch.set_alpha(0)
                ax1.patch.set_alpha(0)

                usia = usia_gender.index
                x = np.arange(len(usia))
                width = 0.4

                # Bar Laki-laki dan Perempuan
                l_vals = usia_gender.get('L', pd.Series([0]*len(usia), index=usia))
                p_vals = usia_gender.get('P', pd.Series([0]*len(usia), index=usia))

                ax1.bar(x - width/2, l_vals, width, label='Laki-laki', color='#4C72B0')
                ax1.bar(x + width/2, p_vals, width, label='Perempuan', color='#C44E52')

                # Tambahkan angka di atas bar
                for i, v in enumerate(l_vals):
                    if v > 0:
                        ax1.text(x[i] - width/2, v + 0.1, str(v), ha='center', fontsize=7, fontweight='bold')
                for i, v in enumerate(p_vals):
                    if v > 0:
                        ax1.text(x[i] + width/2, v + 0.1, str(v), ha='center', fontsize=7, fontweight='bold')

                ax1.set_xticks(x)
                ax1.set_xticklabels([str(int(u)) for u in usia], rotation=45, fontsize=8)
                ax1.set_xlabel("Usia", fontsize=9)
                ax1.set_ylabel("Jumlah", fontsize=9)
                ax1.legend(fontsize=8)
                ax1.grid(axis='y', linestyle='--', alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig1)
                plt.close(fig1)

                # Ringkasan
                total_l = l_vals.sum()
                total_p = p_vals.sum()
                st.caption(f"Laki-laki: **{int(total_l):,}** | Perempuan: **{int(total_p):,}**")
            else:
                st.info("Data tidak tersedia.")
        else:
            st.info("Data tidak tersedia untuk filter ini.")

    # INSIGHT 2: Status Keluarga (Pie Chart)
    with col_ins2:
        st.markdown("##### Status Keluarga Peserta")

        if 'STATUS_WORKSHOP' in df_for_insight.columns and len(df_for_insight) > 0:
            status_counts = df_for_insight['STATUS_WORKSHOP'].value_counts()

            if not status_counts.empty:
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                fig2.patch.set_alpha(0)
                ax2.patch.set_alpha(0)

                colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974']
                ax2.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%',
                       colors=colors[:len(status_counts)], startangle=140,
                       wedgeprops={'edgecolor': 'white'}, textprops={'fontsize': 8})

                plt.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)

                # Detail
                for status, count in status_counts.items():
                    pct = (count / status_counts.sum()) * 100
                    st.caption(f"{status}: **{count:,}** ({pct:.1f}%)")
            else:
                st.info("Data tidak tersedia.")
        else:
            st.info("Data tidak tersedia untuk filter ini.")

    st.markdown("")

    # ----- ROW 2: Distribusi Pendidikan -----
    st.markdown("##### Distribusi Pendidikan Peserta")

    if 'PENDIDIKAN' in df_for_insight.columns and len(df_for_insight) > 0:
        pend_counts = df_for_insight['PENDIDIKAN'].value_counts().sort_values(ascending=False)

        if not pend_counts.empty:
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            fig3.patch.set_alpha(0)
            ax3.patch.set_alpha(0)

            bars = ax3.bar(pend_counts.index, pend_counts.values, color='#4C72B0')

            # Tambahkan angka di atas bar
            for i, v in enumerate(pend_counts.values):
                ax3.text(i, v + 0.3, str(int(v)), ha='center', fontsize=9, fontweight='bold')

            ax3.set_xlabel("Jenjang Pendidikan", fontsize=9)
            ax3.set_ylabel("Jumlah Peserta", fontsize=9)
            ax3.grid(axis='y', linestyle='--', alpha=0.3)
            plt.xticks(rotation=45, fontsize=8)

            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)
        else:
            st.info("Data tidak tersedia.")
    else:
        st.info("Data tidak tersedia untuk filter ini.")

    st.markdown("---")

    # ============ PREVIEW DATA ============
    st.markdown("### Preview Data")
    if q_search:
        st.write(f"Hasil pencarian '{q_search}': **{len(df_filtered)}** data ditemukan")

    # Pilih kolom yang akan ditampilkan (tanpa kolom helper)
    display_cols = [col for col in df_filtered.columns if not col.endswith('_dt') and col != 'TAHUN_WORKSHOP']
    df_show = df_filtered[display_cols]

    # Pagination
    per_page = st.selectbox("Baris per halaman", [10, 25, 50, 100], index=1, key="gab_perpage")
    total_pages = max(1, (len(df_show) - 1) // per_page + 1)
    page = st.number_input("Halaman", min_value=1, max_value=total_pages, value=1, key="gab_page")
    start_idx = (page - 1) * per_page

    # Menampilkan data
    st.dataframe(df_show.iloc[start_idx:start_idx + per_page], width='stretch')


# ============ MAIN ROUTER ============
# Menentukan halaman mana yang akan ditampilkan berdasarkan session state
# Mengecek status login terlebih dahulu sebelum menampilkan dashboard

# Mengecek apakah user sudah login
# Jika belum login (logged_in tidak ada atau False), tampilkan halaman login
if not st.session_state.get('logged_in', False):
    # User belum login, tampilkan halaman login
    render_login_page()
else:
    # User sudah login, tampilkan dashboard sesuai halaman yang dipilih
    page = st.session_state.get('page', 'workshop')

    if page == 'workshop':
        render_workshop_page()
    elif page == 'jobfair':
        # Menggunakan fungsi baru yang menampilkan 2 dataset (Baru dan Lama)
        render_jobfair_page()
    elif page == 'gabungan':
        render_gabungan_page()
    else:
        # Default ke workshop jika page tidak dikenali
        render_workshop_page()

