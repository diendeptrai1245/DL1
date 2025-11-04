import os
from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import config 
import json

# Initialize Flask app
app = Flask(__name__)

# Load Model and Class Names from config
print(f"Loading model from {config.MODEL_SAVE_PATH}...")
model = tf.keras.models.load_model(config.MODEL_SAVE_PATH)

print(f"Loading class names from {config.CLASS_NAMES_PATH}...")
with open(config.CLASS_NAMES_PATH, 'r', encoding='utf-8') as f:
    classes = json.load(f)

# Image processing helper function
def process_image(img_path):
    img = image.load_img(
        img_path, 
        target_size=(config.IMAGE_WIDTH, config.IMAGE_HEIGHT) 
    )
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Home page route 
@app.route('/')
def home():
    return render_template('index.html')

# Prediction API route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    
    upload_folder = 'uploads'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    img_path = os.path.join(upload_folder, file.filename)
    file.save(img_path)

    processed_img = process_image(img_path)
    prediction = model.predict(processed_img)
  
    # ------------------ Prediction Logic ------------------
    prediction_value = prediction[0][0]
    
    # If value > 0.5, it's class 1 (e.g., 'recyclable')
    # Otherwise, it's class 0 (e.g., 'organic')
    predicted_class_index = 1 if prediction_value > 0.5 else 0 
    predicted_class_name = classes[predicted_class_index] 
    # ---------------------------------------------------

    os.remove(img_path)

    return jsonify({'prediction': predicted_class_name})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)