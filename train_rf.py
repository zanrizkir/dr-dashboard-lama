import os
import cv2
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm # Digunakan untuk memunculkan progress bar

# ==========================================
# 1. PENGATURAN DIREKTORI DAN HYPERPARAMETER
# ==========================================
# Ganti dengan jalur absolut jika terjadi error FileNotFoundError
CSV_PATH = 'data/train.csv' 
IMAGE_DIR = 'data/aptos2019-blindness-detection/train_images/'
MODEL_SAVE_PATH = 'models/random_forest_model.pkl'

# Target ukuran gambar. 
# Semakin besar, semakin bagus kualitasnya tapi semakin berat komputasinya.
# 64x64 adalah kompromi yang baik untuk Random Forest di laptop standar.
IMG_SIZE = (64, 64) 

# ==========================================
# 2. FUNGSI PEMUATAN DAN PREPROCESSING DATA
# ==========================================
def load_data(csv_path, image_dir, img_size):
    """
    Fungsi ini membaca file CSV, mencari gambar yang sesuai,
    mengubah ukurannya, dan meratakannya (flatten) menjadi array 1D.
    """
    print(f"-> Membaca label dari: {csv_path}")
    df = pd.read_csv(csv_path)
    
    images = []
    labels = []
    
    print(f"-> Mulai memuat dan meratakan (flatten) gambar dari: {image_dir}")
    # Menggunakan tqdm untuk progress bar visual
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Memproses Gambar"):
        # APTOS 2019 menggunakan ekstensi .png
        img_name = f"{row['id_code']}.png" 
        img_path = os.path.join(image_dir, img_name)
        
        if os.path.exists(img_path):
            try:
                # Baca gambar (secara default OpenCV membaca dalam format BGR)
                img = cv2.imread(img_path)
                # Ubah ke format RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Ubah ukuran (resize) sesuai target
                img = cv2.resize(img, img_size)
                
                # [PENTING UNTUK RANDOM FOREST] 
                # Random Forest tidak bisa memproses array 3D (gambar berwarna).
                # Kita harus meratakannya menjadi array 1D menggunakan fungsi flatten().
                # Contoh: Gambar 64x64x3 RGB akan diratakan menjadi deretan 12.288 piksel
                img_flattened = img.flatten() 
                
                # Simpan ke dalam list
                images.append(img_flattened)
                labels.append(row['diagnosis'])
                
            except Exception as e:
                print(f"\n[Peringatan] Gagal memproses gambar {img_name}: {e}")
        else:
            print(f"\n[Peringatan] Gambar tidak ditemukan: {img_path}")

    # Ubah list menjadi NumPy array agar bisa dibaca oleh scikit-learn
    X = np.array(images)
    y = np.array(labels)
    
    print(f"\n-> Proses pemuatan selesai.")
    print(f"   Total gambar termuat : {len(X)}")
    print(f"   Dimensi matriks X    : {X.shape} (Jumlah Data x Jumlah Piksel)")
    
    return X, y

# ==========================================
# 3. FUNGSI UTAMA (MAIN)
# ==========================================
def main():
    # Buat folder 'models/' jika belum ada
    os.makedirs('models', exist_ok=True)
    
    # 3a. Muat data
    X, y = load_data(CSV_PATH, IMAGE_DIR, IMG_SIZE)
    
    if len(X) == 0:
        print("\n[ERROR] Tidak ada data gambar yang berhasil dimuat. Proses dihentikan.")
        print("Pastikan folder 'data/train_images/' sudah berisi file berekstensi .png")
        return

    # 3b. Bagi data menjadi Data Latih (80%) dan Data Uji (20%)
    print("\n-> Membagi data menjadi 80% Latih dan 20% Uji...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3c. Inisialisasi dan Latih Model Random Forest
    # n_estimators=100 berarti kita akan membuat 100 pohon keputusan (decision trees)
    print("\n-> Memulai pelatihan model Random Forest (ini akan memakan waktu cukup lama)...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    # Proses pelatihan dimulai (fungsi .fit)
    rf_model.fit(X_train, y_train)
    print("-> Pelatihan selesai!")

    # 3d. Evaluasi Model
    print("\n-> Melakukan pengujian (Prediksi pada data uji)...")
    y_pred = rf_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== HASIL EVALUASI ===")
    print(f"Akurasi: {acc:.4f} ({acc * 100:.2f}%)")
    print("\nLaporan Klasifikasi (Classification Report):")
    print(classification_report(y_test, y_pred))

    # 3e. Simpan Model
    print(f"\n-> Menyimpan model ke dalam file: {MODEL_SAVE_PATH}")
    joblib.dump(rf_model, MODEL_SAVE_PATH)
    print("-> Model berhasil disimpan!")
    print("Silakan jalankan Streamlit Anda setelah ini.")

if __name__ == '__main__':
    main()