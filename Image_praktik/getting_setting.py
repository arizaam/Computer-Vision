from __future__ import print_function
import os
import cv2

# Membuat parser argumen
# ap = argparse.ArgumentParser()
# ap.add_argument(
#     "-i", 
#     "--image", 
#     required=True, 
#     help="Path ke file gambar"
# )
# args = vars(ap.parse_args())

imagePath = os.path.join(".", "gng.png")

# Membaca gambar dari path
image = cv2.imread(imagePath)
cv2.imshow("Original", image)

# Mengambil nilai RGB di pixel (0, 0)
(b, g, r) = image[0, 0]
print("Pixel di (0, 0) - Red: {}, Green: {}, Blue: {}".format(r, g, b))

# Mengubah pixel (0, 0) menjadi merah
image[0, 0] = (0, 0, 255)
(b, g, r) = image[0, 0]
print("Pixel di (0, 0) setelah diubah - Red: {}, Green: {}, Blue: {}".format(r, g, b))

# Menampilkan area pojok kiri atas
corner = image[0:100, 0:100]
cv2.imshow("Corner", corner)

# Mengubah area pojok kiri atas menjadi hijau
image[0:100, 0:100] = (0, 255, 0)
cv2.imshow("Updated", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
