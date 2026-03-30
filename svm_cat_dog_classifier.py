import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Dataset path
DATADIR = "dataset/train"

CATEGORIES = ["cats", "dogs"]

IMG_SIZE = 64  # resize images

data = []

print("Loading and processing images...")

# Load and preprocess images
for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    class_num = CATEGORIES.index(category)

    count = 0  # LIMIT images for SVM

    for img in os.listdir(path):
        try:
            img_path = os.path.join(path, img)
            img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img_array is None:
                continue

            resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            data.append([resized, class_num])

            count += 1
            if count >= 1000:  # limit per category
                break

        except Exception:
            pass

print(f"Total images loaded: {len(data)}")

# Shuffle dataset
random.shuffle(data)

# Split features and labels
X = []
y = []

for features, label in data:
    X.append(features)
    y.append(label)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Flatten images
X = X.reshape(-1, IMG_SIZE * IMG_SIZE)

# Normalize
X = X / 255.0

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training SVM model...")

# Train model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Performance")
print("------------------")
print("Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Cat", "Dog"]))

# ==============================
# 📊 1. Confusion Matrix
# ==============================

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Cat", "Dog"])
disp.plot()

plt.title("Confusion Matrix")
plt.show()

# ==============================
# 🖼️ 2. Sample Predictions
# ==============================

plt.figure(figsize=(10, 6))

for i in range(6):
    index = random.randint(0, len(X_test) - 1)

    img = X_test[index].reshape(IMG_SIZE, IMG_SIZE)

    plt.subplot(2, 3, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(f"Pred: {'Dog' if y_pred[index] == 1 else 'Cat'}")
    plt.axis('off')

plt.suptitle("Sample Predictions")
plt.show()

# ==============================
# 📈 3. Accuracy Bar Chart
# ==============================

plt.figure()
plt.bar(["Accuracy"], [accuracy])
plt.title("Model Accuracy")
plt.ylabel("Score")
plt.show()