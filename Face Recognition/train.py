import cv2
import os
import numpy as np


def load_dataset(dataset_path="dataset"):
    faces = []
    labels = []
    label_ids = {}  # untuk mapping label → nama orang
    current_label = 0

    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)

        # skip file
        if not os.path.isdir(person_path):
            continue

        label_ids[current_label] = person_name

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            faces.append(img)
            labels.append(current_label)

        current_label += 1

    return faces, labels, label_ids


faces, labels, label_ids = load_dataset("dataset")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))

recognizer.save("face_model.xml")

# simpan mapping label → nama
import json

with open("labels.json", "w") as f:
    json.dump(label_ids, f)

print("selesai")