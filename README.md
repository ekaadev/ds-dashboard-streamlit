# Dashboard Application Workshop dan Jobfair

## Langkah-Langkah Menjalankan Aplikasi

1. Generate venvironment virtual (opsional tapi disarankan):
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
2. Install library yang dibutuhkan:
   ```bash
   pip install -r requirements.txt
   ```
3. Jalankan aplikasi dengan perintah:
   ```bash
   streamlit run streamlit_app.py
   ```
4. Buka browser dan akses aplikasi di:
   ```
    http://localhost:8501
    ```
5. Aplikasi sudah berjalan dan siap digunakan!

## Langkah-Langkah Menajalankan File .sql

### phpMyAdmin

1. Nyalakan XAMPP 
   - Start Apache dan MySQL dari Control Panel XAMPP.
2. Buka phpMyAdmin
   - Buka browser → akses http://localhost/phpmyadmin
3. Buat database baru 
   - Klik Databases 
   - Masukkan nama database contoh: mydb 
   - Klik Create
4. Impor file .sql
   - Klik database yang baru dibuat (misal: mydb)
   - Klik tab Import 
   - Klik Choose File → pilih file export.sql 
   - Scroll ke bawah → klik Go

### MySQL Workbench
1. Buka MySQL Workbench
2. Klik koneksi ke server lokal → biasanya bernama:
```
Local instance MySQL80 (root)
```

3. Buat database baru
   - Buka tab Query
   - Jalankan:
   ```sql
    CREATE DATABASE mydb;
    USE mydb;
    ```
   
4. Impor file .sql
   - Menu bar → Server 
   - Pilih Data Import 
   - Pilih Import from Self-Contained File 
   - Browse → pilih file export.sql

5. Pada bagian Default Target Schema, pilih (atau buat):
    - Pilih mydb (atau database yang sudah dibuat)

6. Klik Start Import untuk memulai proses impor data
7. Setelah selesai, data sudah berhasil diimpor ke database mydb