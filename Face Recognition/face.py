import cv2
import os

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)

person_name = ("Farid")
save_dir = f"dataset/{person_name}"
os.makedirs(save_dir, exist_ok=True)

count = 0

while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        face = gray[y : y + h, x : x + w]
        face = cv2.resize(face, (200, 200))
        cv2.imwrite(f"{save_dir}/{count}.jpg", face)
        count += 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Capture", frame)

    if cv2.waitKey(5) == ord("q") or count >= 10:
        break

cam.release()
cv2.destroyAllWindows()