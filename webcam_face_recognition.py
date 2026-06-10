import cv2
import joblib
import numpy as np

# ==========================
# Load Saved Models
# ==========================

print("Loading models...")

model = joblib.load("face_recognition_model.pkl")
pca = joblib.load("pca_model.pkl")
encoder = joblib.load("label_encoder.pkl")

print("Models loaded successfully!")

# ==========================
# Load Face Detector
# ==========================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# ==========================
# Open Webcam
# ==========================

cap = cv2.VideoCapture(0)

# If webcam doesn't open try camera index 1
if not cap.isOpened():

    print("Camera 0 failed. Trying Camera 1...")

    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("ERROR: Cannot access webcam")
        exit()

print("Webcam started successfully!")
print("Press Q to quit")

# ==========================
# Webcam Loop
# ==========================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(50, 50)
    )

    for (x, y, w, h) in faces:

        try:

            # Extract Face
            face = gray[y:y+h, x:x+w]

            # Resize same as training image size
            face = cv2.resize(face, (100, 100))

            # Flatten
            face = face.flatten().reshape(1, -1)

            # Normalize
            face = face / 255.0

            # PCA
            face_pca = pca.transform(face)

            # Prediction
            prediction = model.predict(face_pca)

            person_name = encoder.inverse_transform(prediction)[0]

            # Draw Rectangle
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Show Name
            cv2.putText(
                frame,
                person_name,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        except Exception as e:
            print("Prediction Error:", e)

    cv2.imshow("PCA + ANN Face Recognition", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

# ==========================
# Cleanup
# ==========================

cap.release()
cv2.destroyAllWindows()
