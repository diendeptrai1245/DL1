# train.py
from src.data_preprocessing import load_data
from src.model import build_model
from src.augmentation import get_augmented_data

# Import config module
import config
import json
import pickle

# Main training script  
print("Loading augmented training data...")
train_data = get_augmented_data(config.TRAIN_DIR)
print("Loading test data...")

# (Following your original structure)
_, test_data = load_data(config.TRAIN_DIR, config.TEST_DIR)

# Save class names to a JSON file
print("Saving class names...")
try:
    class_indices = train_data.class_indices
    class_names = list(class_indices.keys()) # Get class names from the generator
    
    with open(config.CLASS_NAMES_PATH, 'w') as f:
        json.dump(class_names, f)
    print(f"Saved class list: {class_names} to {config.CLASS_NAMES_PATH}")
except AttributeError:
    print("Warning: Could not find 'class_indices' on train_data. Class names not saved.")

# Create model 
print("Building model...")
model = build_model()

# Train
print("Starting model training...")
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=config.EPOCHS # CHANGED: Use config for epochs
)
# Save training history plots  
# print("Saving training plots...")
# save_training_plots(history)

# Save model
print(f"Saving model to {config.MODEL_SAVE_PATH}...")

# Save training history using pickle
with open("models/train_history.pkl", "wb") as f:
    pickle.dump(history.history, f)
print("✅ Training history saved to models/train_history.pkl")

# Use config for save path
model.save(config.MODEL_SAVE_PATH) 
print("Model saved successfully.")