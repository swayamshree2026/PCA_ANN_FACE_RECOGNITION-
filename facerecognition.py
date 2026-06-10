import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# =========================
# Load Dataset
# =========================

data = []
labels = []

dataset_path = os.path.join("dataset", "dataset", "faces")

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)

    if not os.path.isdir(person_path):
        continue

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        if not os.path.isfile(img_path):
            continue

        if not img_name.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        ):
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img = cv2.resize(img, (100, 100))

        data.append(img.flatten())
        labels.append(person)

X = np.array(data)
y = np.array(labels)

print("Dataset Shape:", X.shape)

# =========================
# Label Encoding
# =========================

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print("Classes:", encoder.classes_)

# =========================
# Train-Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

X_train = X_train / 255.0
X_test = X_test / 255.0

# =========================
# PCA
# =========================

pca = PCA(n_components=100)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print("PCA Shape:", X_train_pca.shape)

# =========================
# Show First Eigenface
# =========================

eigenface = pca.components_[0].reshape(100, 100)

plt.imshow(eigenface, cmap="gray")
plt.title("First Eigenface")
plt.axis("off")
plt.show()

# =========================
# Neural Network
# =========================

model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42
)

print("Training Model...")

model.fit(X_train_pca, y_train)

# =========================
# Prediction & Accuracy
# =========================

pred = model.predict(X_test_pca)

accuracy = accuracy_score(y_test, pred)

print("Accuracy:", accuracy * 100)

# =========================
# Confusion Matrix
# =========================

cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(8, 6))
plt.imshow(cm)
plt.colorbar()

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

plt.show()

# =========================
# Save Models
# =========================

joblib.dump(model, "face_recognition_model.pkl")
joblib.dump(pca, "pca_model.pkl")
joblib.dump(encoder, "label_encoder.pkl")

print("Models saved successfully!")
from sklearn.metrics import classification_report

y_pred = model.predict(X_test_pca)

print(classification_report(y_test, y_pred))
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.plot(model.loss_curve_)

plt.title("Training Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")

plt.grid(True)

plt.show()
