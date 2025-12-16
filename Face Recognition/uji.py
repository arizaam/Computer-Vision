import cv2
import os
import json

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_model.xml")

with open("labels.json", "r") as f:
    label_ids = json.load(f)

# Load Haar Cascade
xmlPath = os.path.join(".", "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(xmlPath)


# Buka webcam (0 = default camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Konversi ke grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # Gambar kotak pada wajah
    for x, y, w, h in faces:
        face_roi = gray[y : y + h, x : x + w]
        label, confidence = recognizer.predict(face_roi)

        name = label_ids[str(label)]
        text = f"{name} ({confidence:.1f})"

        cv2.putText(
            frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Tampilkan output
    cv2.imshow("Face recognition", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
