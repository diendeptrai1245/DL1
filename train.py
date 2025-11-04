from src.data_preprocessing import load_data
from src.model import build_model
from src.augmentation import get_augmented_data

train_dir = "DATASET/TRAIN"
test_dir = "DATASET/TEST"

# Load data và augment
train_data = get_augmented_data(train_dir)
_, test_data = load_data(train_dir, test_dir)

# Tạo model
model = build_model()

# Huấn luyện
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=10
)

# Lưu model
model.save("models/cnn_rubbish_classifier.h5")
print("Model saved successfully!")
