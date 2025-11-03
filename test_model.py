from src.evaluate import evaluate_model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# === Cấu hình đường dẫn ===
MODEL_PATH = "models/cnn_rubbish_classifier.h5"
TEST_DIR = "DATASET/TEST"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# === Load model ===
print("[INFO] Loading trained model...")
model = load_model(MODEL_PATH)
print("[INFO] Model loaded successfully!")

# === Chuẩn bị dữ liệu test ===
print("[INFO] Loading test data...")
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# === Đánh giá mô hình ===
model_evaluation = evaluate_model(model, test_data)

import numpy as np
import matplotlib.pyplot as plt

# Lấy ngẫu nhiên 5 ảnh
x_batch, y_batch = next(test_data)
preds = model.predict(x_batch)

for i in range(5):
    plt.imshow(x_batch[i])
    pred_label = "Non-Organic" if preds[i][0] > 0.5 else "Organic"
    true_label = "Non-Organic" if y_batch[i] == 1 else "Organic"
    plt.title(f"Pred: {pred_label} | True: {true_label}")
    plt.axis("off")
    plt.show()
