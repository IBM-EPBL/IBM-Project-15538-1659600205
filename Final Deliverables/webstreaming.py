import cv2
import os
import numpy as np
from keras.models import load_model
from skimage.transform import resize
from flask import Flask, render_template, request, send_from_directory


model = load_model('aslpng.h5')
UPLOAD_FOLDER = './uploads'

app = Flask(__name__)


def detect(frame):
    img = resize(frame, (64, 64, 1))
    img = np.expand_dims(img, axis = 0)
    if(np.max(img) > 1):
        img = img/255.0
    prediction = model.predict(img)
    prediction = (model.predict(img) > 0.5).astype('int32')
    if prediction[0][0] == 1:
        label = 'A'
    elif prediction[0][1] == 1:
        label = 'B'
    elif prediction[0][2] == 1:
        label = 'C'
    elif prediction[0][3] == 1:
        label = 'D'
    elif prediction[0][4] == 1:
        label = 'E'
    elif prediction[0][5] == 1:
        label = 'F'
    elif prediction[0][6] == 1:
        label = 'G'
    elif prediction[0][7] == 1:
        label = 'H'
    elif prediction[0][8] == 1:
        label = 'I'
    else:
        label = 'No Match Found'
    
    return label


@app.route('/predict', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        render_template('index.html')
    else:
        file = request.files['image']
        upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(upload_image_path)
        frame = cv2.imread(os.path.join(UPLOAD_FOLDER, file.filename))
        label = detect(frame)
        return render_template(
            'predict.html', image_file_name = file.filename, label = label
        )


@app.route('/predict/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/')
def index():
    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug = True)
