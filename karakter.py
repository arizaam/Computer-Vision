import cv2
import numpy as np
import os

# --- Persiapan ---
# Buat folder output jika belum ada
output_folder = "output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Ukuran kanvas
canvas_size = 400
kanvas = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8) # Kanvas putih

# Warna
HITAM = (0, 0, 0)
BIRU_GELAP = (100, 50, 0)
MERAH = (0, 0, 255)
HIJAU = (0, 255, 0)

# --- 1. Membuat Karakter (Robot Sederhana) ---

# Kepala (Kotak)
cv2.rectangle(kanvas, (100, 80), (300, 280), BIRU_GELAP, -1)
cv2.rectangle(kanvas, (100, 80), (300, 280), HITAM, 2)

# Antena (Garis dan Lingkaran)
cv2.line(kanvas, (200, 80), (200, 50), HITAM, 3)
cv2.circle(kanvas, (200, 50), 10, MERAH, -1)

# Mata (Lingkaran)
cv2.circle(kanvas, (150, 150), 20, HITAM, -1)
cv2.circle(kanvas, (250, 150), 20, HITAM, -1)
# Pupil (Lingkaran lebih kecil)
cv2.circle(kanvas, (155, 155), 8, (255, 255, 255), -1)
cv2.circle(kanvas, (255, 155), 8, (255, 255, 255), -1)

# Mulut (Garis)
cv2.line(kanvas, (140, 230), (260, 230), HITAM, 5)

# Teks "ROBOT"
cv2.putText(kanvas, "ROBOT", (120, 330), cv2.FONT_HERSHEY_SIMPLEX, 1, HITAM, 2)


# Simpan karakter asli
cv2.imwrite(os.path.join(output_folder, "karakter.png"), kanvas)
karakter_asli = kanvas.copy()

print(f"Karakter asli disimpan di: {os.path.join(output_folder, 'karakter.png')}")

# Tampilkan Karakter Asli (opsional)
cv2.imshow("1. Karakter Asli", karakter_asli)
cv2.waitKey(0)


# --- 2. Transformasi pada Karakter ---

karakter_transform = karakter_asli.copy()
tinggi, lebar = karakter_transform.shape[:2]

# 2.1. Translasi (Geser 50 piksel ke kanan dan 30 piksel ke bawah)
M_translasi = np.float32([[1, 0, 50], [0, 1, 30]])
karakter_translasi = cv2.warpAffine(karakter_transform, M_translasi, (lebar, tinggi), borderValue=(255, 255, 255))
cv2.imwrite(os.path.join(output_folder, "karakter_translasi.png"), karakter_translasi)
print(f"Hasil Translasi disimpan di: {os.path.join(output_folder, 'karakter_translasi.png')}")
cv2.imshow("2.1. Translasi (Geser)", karakter_translasi)
cv2.waitKey(0)

# 2.2. Rotasi (Putar 45 derajat di sekitar pusat)
pusat = (lebar // 2, tinggi // 2)
M_rotasi = cv2.getRotationMatrix2D(pusat, 45, 1.0)
karakter_rotasi = cv2.warpAffine(karakter_transform, M_rotasi, (lebar, tinggi), borderValue=(255, 255, 255))
cv2.imwrite(os.path.join(output_folder, "karakter_rotasi.png"), karakter_rotasi)
print(f"Hasil Rotasi disimpan di: {os.path.join(output_folder, 'karakter_rotasi.png')}")
cv2.imshow("2.2. Rotasi 45 Derajat", karakter_rotasi)
cv2.waitKey(0)


# 2.3. Resize (Ubah Ukuran menjadi setengah)
skala = 0.5
dimensi_baru = (int(lebar * skala), int(tinggi * skala))
karakter_resize = cv2.resize(karakter_transform, dimensi_baru, interpolation=cv2.INTER_AREA)

# Simpan dalam kanvas baru (agar ukurannya sama untuk tampilan)
kanvas_resize = np.full((tinggi, lebar, 3), 255, dtype=np.uint8)
kanvas_resize[0:dimensi_baru[1], 0:dimensi_baru[0]] = karakter_resize
cv2.imwrite(os.path.join(output_folder, "karakter_resize.png"), kanvas_resize)
print(f"Hasil Resize disimpan di: {os.path.join(output_folder, 'karakter_resize.png')}")
cv2.imshow("2.3. Resize (Perkecil)", kanvas_resize)
cv2.waitKey(0)

# 2.4. Crop (Potong bagian kepala)
x, y, w, h = 100, 50, 200, 250  # Koordinat dan ukuran potongan
karakter_crop = karakter_transform[y:y+h, x:x+w]

# Simpan hasil crop (ukuran berbeda)
cv2.imwrite(os.path.join(output_folder, "karakter_crop.png"), karakter_crop)
print(f"Hasil Crop disimpan di: {os.path.join(output_folder, 'karakter_crop.png')}")
cv2.imshow("2.4. Crop (Potongan Kepala)", karakter_crop)
cv2.waitKey(0)


# --- 3. Operasi Aritmatika atau Bitwise ---

# Karakter kedua (Bentuk sederhana)
latar_kedua = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8) # Kanvas hitam
# Lingkaran besar hijau
cv2.circle(latar_kedua, (250, 250), 180, HIJAU, -1)
# Kotak kecil merah
cv2.rectangle(latar_kedua, (300, 50), (380, 150), MERAH, -1)

# Karakter asli diubah ukurannya agar memiliki fokus untuk operasi
karakter_fokus = karakter_asli.copy()
karakter_fokus[np.all(karakter_fokus == 255, axis=2)] = [0, 0, 0] # Ubah putih menjadi hitam

# 3.1. Operasi Aritmatika: cv2.addWeighted() (Campuran berbobot)
# Mencampur latar kedua dengan karakter asli yang sudah diubah warna putihnya menjadi hitam
karakter_campuran = cv2.addWeighted(karakter_fokus, 0.7, latar_kedua, 0.3, 0)
cv2.imwrite(os.path.join(output_folder, "karakter_campuran_aritmatika.png"), karakter_campuran)
print(f"Hasil AddWeighted disimpan di: {os.path.join(output_folder, 'karakter_campuran_aritmatika.png')}")
cv2.imshow("3.1. AddWeighted (Campuran Aritmatika)", karakter_campuran)
cv2.waitKey(0)

# 3.2. Operasi Bitwise: cv2.bitwise_and() untuk Masking
# Kita akan menempelkan karakter asli (foreground) ke latar kedua (background)

# 1. Buat mask dari karakter asli (hanya bagian non-putih)
# Ubah ke grayscale
karakter_gray = cv2.cvtColor(karakter_asli, cv2.COLOR_BGR2GRAY)
# Ambil bagian yang bukan putih (threshold)
# ret, mask = cv2.threshold(karakter_gray, 254, 255, cv2.THRESH_BINARY_INV)
# Untuk karakter dengan warna, lebih baik menggunakan batas warna
mask = cv2.inRange(karakter_asli, (0, 0, 0), (254, 254, 254))

# 2. Buat mask invers
mask_inv = cv2.bitwise_not(mask)

# 3. Hitamkan area di latar kedua yang akan ditempati karakter (background)
latar_bg = cv2.bitwise_and(latar_kedua, latar_kedua, mask=mask_inv)

# 4. Ambil hanya bagian karakter dari karakter asli (foreground)
karakter_fg = cv2.bitwise_and(karakter_asli, karakter_asli, mask=mask)

# 5. Gabungkan latar belakang dan karakter
karakter_bitwise = cv2.add(latar_bg, karakter_fg)
cv2.imwrite(os.path.join(output_folder, "karakter_bitwise_and_masking.png"), karakter_bitwise)
print(f"Hasil Bitwise AND (Masking) disimpan di: {os.path.join(output_folder, 'karakter_bitwise_and_masking.png')}")
cv2.imshow("3.2. Bitwise AND (Masking)", karakter_bitwise)
cv2.waitKey(0)


# Tutup semua jendela
cv2.destroyAllWindows()
cv2.imwrite(os.path.join('D:\Kuliah\Computer Vision\output' ,"karakter.png"), kanvas)
karakter = kanvas.copy
cv2.waitKey(0)