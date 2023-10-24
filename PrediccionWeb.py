import os
import tempfile
from flask import Flask, request, redirect, render_template, url_for
from skimage import io
import base64
from skimage.transform import resize
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

model = load_model('modelo_entrenado.h5')
app = Flask(__name__, template_folder="templates/")

@app.route("/")
def main():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No se ha proporcionado ninguna imagen'})

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'Nombre de archivo de imagen no válido'})

    image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    imagen_gris = cv2.resize(image, (28, 28), interpolation=cv2.INTER_CUBIC)
    imagen_gris = cv2.cvtColor(imagen_gris, cv2.COLOR_BGR2GRAY)
    imagen_tensor = tf.cast(imagen_gris, dtype=tf.float32)
    imagen_tensor = tf.reshape(imagen_tensor, (1, 28, 28, 1))

    prediction = model.predict(imagen_tensor)
    classes = ["Katakana A", "Katakana E", "Katakana I", "Katakana O", "Katakana U"]

    predicted_class = classes[np.argmax(prediction)]

    return jsonify({'prediction': predicted_class})


@app.route('/predicciones')
def show_predictions():
    nums = request.args.get('nums')
    img_data = request.args.get('img_data')
    componentes = nums.split(', ')
    nums = [float(componente) for componente in componentes]
    letras = ["Katakana A", "Katakana E", "Katakana I","Katakana O","Katakana U"]
    if img_data is not None:
        return render_template('Prediccion.html', nums=nums, letras=letras, img_data=img_data)
    else:
        return redirect("/", code=302)


if __name__ == "__main__":
    app.run()