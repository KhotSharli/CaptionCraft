from flask import Flask, render_template, request
from PIL import Image
import os
import numpy as np
from tqdm.notebook import tqdm
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from CaptionCraft import generate_captions_for_image

app = Flask(__name__)

# Load your image caption generation code and necessary functions here

# Define a route for the home page
@app.route('/')
def index():
    return render_template('upload.html')

# Define a route to handle caption generation from an uploaded image
@app.route('/generate_caption', methods=['POST'])
def generate_caption():
    if 'image' not in request.files:
        return "No image found"
    
    file = request.files['image']
    if file.filename == '':
        return "No selected image"
    
    # Save the uploaded image
    image_path = os.path.join('static', file.filename)
    file.save(image_path)
    
    # Generate caption for the uploaded image
    generated_caption = generate_captions_for_image(image_path)  # Call your caption generation function
    
    return render_template('result.html', image=image_path, caption=generated_caption)

if __name__ == '__main__':
    app.run(debug=True)
