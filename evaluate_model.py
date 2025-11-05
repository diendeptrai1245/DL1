import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# PATH AND PARAMETERS
MODEL_PATH = "models/cnn_rubbish_classifier.h5"
TEST_DIR = "DATASET/TEST"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# LOAD MODEL
print("[INFO] Loading trained model...")
model = load_model(MODEL_PATH)

# PREPARE TEST DATA
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# PREDICTIONS ON TEST DATA
print("[INFO] Predicting on test data...")
pred_probs = model.predict(test_data)
pred_classes = (pred_probs > 0.5).astype("int32").flatten()
true_classes = test_data.classes
class_labels = list(test_data.class_indices.keys())

#  EVALUATION ON TEST DATA
loss, acc = model.evaluate(test_data)
print(f"\n✅ Test Accuracy: {acc*100:.2f}% | Test Loss: {loss:.4f}")

def compute_miou(cm):
    num_classes = cm.shape[0]
    ious = []
    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        denom = (tp + fp + fn)
        iou = tp / denom if denom != 0 else 0
        ious.append(iou)
    return np.mean(ious)


# Classification Report
print("\n📊 Classification Report:")
print(classification_report(true_classes, pred_classes, target_names=class_labels))

# Confusion Matrix
cm = confusion_matrix(true_classes, pred_classes)

miou = compute_miou(cm)
print(f"\n📏 Mean IoU (mIoU): {miou * 100:.2f}%")

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# LOAD TRAINING HISTORY
with open('models/train_history.pkl', 'rb') as f:
    history = pickle.load(f)

# PLOT TRAINING HISTORY
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('Training & Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history['accuracy'], label='Train Acc')
plt.plot(history['val_accuracy'], label='Val Acc')
plt.title('Training & Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()



