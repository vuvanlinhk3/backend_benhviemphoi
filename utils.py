import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import datetime

model = load_model("densenet121_pneumonia_model.h5")

def analyze_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    preds = model.predict(x)[0]
    pneumonia_prob = float(preds[0])
    normal_prob = float(1 - preds[0])

    result = "Pneumonia" if pneumonia_prob > 0.5 else "Normal"
    timestamp = datetime.datetime.now().isoformat()

    return result, pneumonia_prob, normal_prob, timestamp
