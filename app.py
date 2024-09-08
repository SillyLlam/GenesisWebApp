from flask import Flask, render_template, request, redirect, url_for
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('models/best_model.keras')

# Define class labels
class_labels = ['Acne', 'Dark Circles', 'Dry', 'Normal', 'Oily', 'Pimple']

# Ensure the uploads directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filepath = os.path.join('uploads', file.filename)
            file.save(filepath)

            # Process the image and make predictions
            img = image.load_img(filepath, target_size=(128, 128))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions)
            predicted_class_name = class_labels[predicted_class]

            # Remove the uploaded file after processing
            os.remove(filepath)

            # Redirect to results page with the prediction
            return redirect(url_for('results', prediction=predicted_class_name))
    return render_template('index.html')

@app.route('/results')
def results():
    prediction = request.args.get('prediction')
    return render_template('results.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
