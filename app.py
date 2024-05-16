import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from os import environ
from flask import Flask, flash, request, redirect, url_for, render_template,Response
from werkzeug.utils import secure_filename
import cv2,base64
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from keras.models import load_model

app = Flask(__name__)
 
UPLOAD_FOLDER = 'static/uploads/'
app.secret_key = "secret-key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


model = load_model("fake_image_model.h5")
class_names = ["Fake","Real"]

print("Model Loaded...!!!")

def convert_to_ela_image(path, quality):
    temp_filename = 'temp_file_name.jpg'
    
    
    image = Image.open(path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality = quality)
    temp_image = Image.open(temp_filename)
    
    ela_image = ImageChops.difference(image, temp_image)
    
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    return ela_image

def prepare_image(image_path):
    return np.array(convert_to_ela_image(image_path, 90).resize((128,128))).flatten() / 255.0

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
 
@app.route('/')
def home():
	return render_template('upload.html')


def predict_class(filename):
    image = str(UPLOAD_FOLDER+filename)
    image = prepare_image(image)
    image = image.reshape(-1, 128, 128, 3)
    y_pred = model.predict(image)
    y_pred_class = np.argmax(y_pred, axis = 1)[0]
    class_name = class_names[y_pred_class]
    Prediction = round(np.amax(y_pred)*100,2)
    ret_arr = [class_name,Prediction]
    return ret_arr

def ela_image(filename):
    img = str(UPLOAD_FOLDER+filename)
    img =convert_to_ela_image(img,90)
    img = np.array(img)
    image_content = cv2.imencode('.jpg', img)[1].tostring()
    encoded_image = base64.encodebytes(image_content)
    ela = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')
    return ela
    
         
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('Image successfully uploaded and displayed below')
        ret_arr=predict_class(filename)
        return render_template('upload.html', filename=filename,ret_arr=ret_arr,ela=ela_image(filename),init=True)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):         
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run(debug=True)
