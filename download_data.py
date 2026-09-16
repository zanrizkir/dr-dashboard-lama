import os

# Suntikkan token Kaggle langsung ke dalam sistem sebelum library dipanggil
os.environ['KAGGLE_API_TOKEN'] = 'KGAT_e4221f4a4e2603cd426e879c605b6453'

import kagglehub

print("Sistem berhasil terautentikasi!")
print("Memulai unduhan dataset APTOS 2019 (Sekitar 8 GB, mohon bersabar)...")

# Memulai proses unduhan
path = kagglehub.competition_download('aptos2019-blindness-detection')

print("\n--- UNDUHAN SELESAI ---")
print("Silakan buka File Explorer dan copy folder 'train_images' dari lokasi berikut:")
print(path)