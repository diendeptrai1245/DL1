from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# Load model
model = load_model("models/cnn_rubbish_classifier.h5")

# Prediction function
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    if prediction[0][0] > 0.5:
        print(f"{img_path}: Non-Organic Waste (Không hữu cơ)")
    else:
        print(f"{img_path}: Organic Waste (Hữu cơ)")

# Test
predict_image("D:\\DeepLearning\\DL1\\DATASET\\TEST\\R\\R_10001.jpg")
