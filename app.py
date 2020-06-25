from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

from tensorflow.keras.models import model_from_json
from tensorflow.python.framework import ops
ops.reset_default_graph()
from keras.preprocessing import image

import numpy as np 
import h5py
import PIL
import os
from PIL import Image 

app = Flask(__name__)

MODEL_ARCHITECTURE = './model_adam_1.json'
MODEL_WEIGHTS = './model_100_epochs.h5'

json_file = open(MODEL_ARCHITECTURE)
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights(MODEL_WEIGHTS)
print("Model loaded. Check http://127.0.0.1:5000/")

def model_predict(img_path, model):
	"""
	Args:
		-- img_path :a URL path where a given image is stored.
		-- model : a given Keras CNN model.
	"""
	IMG = image.load_img(img_path).convert('L')
	print(type(IMG))

	# Pre-processing the image
	IMG_ = IMG.resize((64,64))
	print(type(IMG_))
	IMG_ = np.asarray(IMG_)
	print(IMG_.shape)
	IMG_ = np.true_divide(IMG_, 255)
	IMG_ = IMG_.reshape(1, 64,64 ,1)
	print(type(IMG_), IMG_.shape)

	print(model)

	model.compile(loss = "categorical_crossentropy", metrics=['accuracy'], optimizer='Adam')
	prediction = model.predict_classes(IMG_)

	return prediction

# ::: FLASK Routes
@app.route('/', methods=['GET'])
def index():
	# Main Page
	return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():

	# Constants

	classes = {'TRAIN':['NORMAL', 'PNEUMONIA'],
			   'VALIDATION':['NORMAL', 'PNEUMONIA']
			  }

	if request.method == "POST":

		# Get the file from post request
		f = request.files['file']

		# Save the file to ./uploads
		basepath = os.path.dirname(__file__)
		file_path = os.path.join(
			basepath, "uploads", secure_filename(f.filename))
		f.save(file_path)

		# Make Prediction

		prediction = model_predict(file_path, model)

		predicted_class = classes['TRAIN'][prediction[0]]
		print('We think that is {}.'.format(predicted_class.lower()))

		return str(predicted_class).lower()

if __name__  == '__main__' :
	app.run(debug = True)

























