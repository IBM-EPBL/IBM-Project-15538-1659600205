import numpy as np
import cv2
from keras.models import load_model
from flask import Flask, render_template, Response
import tensorflow as tf
from skimage.transform import resize
from camera import Camera

global graph
global Writer

graph = tf.compat.v1.get_default_graph
Writer = None


model = load_model('aslpng.h5')

vals = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']

app = Flask(__name__)

vs = cv2.VideoCapture(0)

pred = ''

def detect(frame):
    img = resize(frame, (64, 64, 1))
    img = np.expand_dims(img, axis = 0)
    if(np.max(img) > 1):
        img = img/255.0
    with graph.as_default():
        prediction =  (model.predict(img) > 0.5).astype('int32')
    print(prediction)
    pred = vals[prediction[0]]
    print(pred)
    return pred

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.video()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(
        gen(Camera()),
        mimetype = 'multipart/x-mixed-replace; boundary = frame'
    )



if __name__ == '__main__':
    app.run(host = '0.0.0.0', debug = True)