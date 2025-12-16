import cv2
import json
import numpy as np
import pandas as pd
from datetime import datetime

# --- Konfigurasi ---
cascade_path = 'haarcascade_frontalface_default.xml'
trainer_path = 'face_model.xml'
labels_file = 'labels.json'
attendance_file = 'Attendance_16-Des.csv'

# Muat Model dan Classifier
face_cascade = cv2.CascadeClassifier(cascade_path)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(trainer_path)

# Baca Mapping ID ke Nama
try:
    with open(labels_file, 'r') as f:
        # PENTING: Key dalam JSON akan dibaca sebagai STRING. 
        # Kita perlu mengkonversinya kembali ke INTEGER.
        raw_map = json.load(f)
        id_to_name = {int(k): v for k, v in raw_map.items()}
        print(f"ID-Name Mapping berhasil dimuat dari {labels_file}")
except FileNotFoundError:
    print("Error: label.csv tidak ditemukan. Pastikan Anda sudah melatih model dan menyimpan label.")
    id_to_name = {}

# Fungsi untuk mencatat Absensi
def mark_attendance(Id):
    # Dapatkan nama dari ID
    name = id_to_name.get(Id, "Unknown")
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')

    # Buat entri baru
    new_entry = pd.DataFrame([{'Id': Id, 'Name': name, 'Date': date_str, 'Time': time_str, 'Status': 'Present'}])

    # Cek apakah file absensi sudah ada
    try:
        df_absensi = pd.read_csv(attendance_file)
        
        # Cek apakah user sudah absen hari ini
        if not ((df_absensi['Id'] == Id) & (df_absensi['Date'] == date_str)).any():
            # Catat hanya jika belum absen hari ini
            df_absensi = pd.concat([df_absensi, new_entry], ignore_index=True)
            df_absensi.to_csv(attendance_file, index=False)
            print(f"Absensi Dicatat: {name} pada {time_str}")
        else:
            print(f"{name} sudah absen hari ini.")

    except FileNotFoundError:
        # Jika file belum ada, buat file baru
        new_entry.to_csv(attendance_file, index=False)
        print(f"File Absensi dibuat. Absensi Dicatat: {name} pada {time_str}")


# Mulai Video Capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set Lebar frame
cam.set(4, 480) # set Tinggi frame

minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True:
    ret, img = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Prediksi ID
        Id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        # Cek Confidence (semakin rendah semakin yakin)
        if confidence < 100: 
            name = id_to_name.get(Id, "Unknown")
            confidence_level = f"  {round(100 - confidence)}%"
            mark_attendance(Id) # Panggil fungsi absensi

        else:
            Id = 0 
            name = "Unknown"
            confidence_level = f"  {round(100 - confidence)}%"

        # Tampilkan hasil di layar
        cv2.putText(img, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, confidence_level, (x + 5, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)

    cv2.imshow('Attendance System', img)

    k = cv2.waitKey(10) & 0xff
    if k == 27: # Tekan 'ESC' untuk keluar
        break

print("\n[INFO] Program Selesai.")
cam.release()
cv2.destroyAllWindows()