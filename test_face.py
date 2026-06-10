import cv2
import joblib
import numpy as np
import os

# ==========================



print("Current Directory:", os.getcwd())

if not os.path.exists("face_recognition_model.pkl"):
    print("ERROR: face_recognition_model.pkl not found")
    exit()

if not os.path.exists("pca_model.pkl"):
    print("ERROR: pca_model.pkl not found")
    exit()

if not os.path.exists("label_encoder.pkl"):
    print("ERROR: label_encoder.pkl not found")
    exit()

if not os.path.exists("test.jpg"):
    print("ERROR: test.jpg not found")
    print("Place a test image in project folder and name it test.jpg")
    exit()

# ==========================
# Load Models
# ==========================

model = joblib.load("face_recognition_model.pkl")
pca = joblib.load("pca_model.pkl")
encoder = joblib.load("label_encoder.pkl")

print("Models Loaded Successfully")

# ==========================
# Read Test Image
# ==========================

img = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Could not read image")
    exit()

img = cv2.resize(img, (100, 100))

# Show image
cv2.imshow("Test Image", img)

# ==========================
# Preprocess
# ==========================

img = img.flatten().reshape(1, -1)

img = img / 255.0

# ==========================
# PCA Transformation
# ==========================

img_pca = pca.transform(img)

# ==========================
# Prediction
# ==========================

prediction = model.predict(img_pca)

person_name = encoder.inverse_transform(prediction)

print("\n=========================")
print("Predicted Person:", person_name[0])
print("=========================\n")

cv2.waitKey(0)
cv2.destroyAllWindows()