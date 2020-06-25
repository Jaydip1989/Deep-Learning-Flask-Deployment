#################################################################################################################################
# Data can be obtained from : https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
#################################################################################################################################
import numpy as np 
import keras
import tensorflow
import math
from keras.models import Sequential, load_model, model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, BatchNormalization, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from tensorflow.python.framework import ops
ops.reset_default_graph()
import PIL
from PIL import Image

train_data = "/Users/dipit/chest_xray/train/"
val_data = "/Users/dipit/chest_xray/test"
test_data = "/Users/dipit/chest_xray/val"


img_rows, img_cols = 64, 64
num_channels = 1
num_classes  = 2
batch_size = 16

train_datagen = ImageDataGenerator(rescale = 1./255,
								   height_shift_range = 0.3,
								   rotation_range = 45,
								   width_shift_range = 0.3,
								   fill_mode = "nearest",
								   zoom_range = 0.3,
								   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale= 1./255)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(directory = train_data,
													target_size = (img_rows, img_cols),
													shuffle = True,
													batch_size = batch_size,
													color_mode = 'grayscale',
													class_mode = "categorical")

val_generator = val_datagen.flow_from_directory(directory = val_data,
												target_size=(img_rows, img_cols),
												shuffle = False,
												batch_size = batch_size,
												color_mode = 'grayscale',
												class_mode = "categorical")

test_generator = test_datagen.flow_from_directory(directory=test_data,
												  target_size=(img_rows, img_cols),
												  shuffle = False,
												  batch_size = 2,
												  color_mode = 'grayscale',
												  class_mode = None)


model = Sequential()
model.add(Conv2D(32, (5,5), input_shape=(img_rows, img_cols, num_channels), activation="relu", padding="same"))
model.add(Conv2D(32, (3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides = 2))
model.add(BatchNormalization())

model.add(Conv2D(64, (3,3), padding="same", activation="relu"))
model.add(Conv2D(64, (3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides = 2))
model.add(BatchNormalization())

model.add(Conv2D(64, (3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size = (2,2), strides = 2))
model.add(BatchNormalization())

model.add(Conv2D(64, (3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size = (2,2),strides=2))
model.add(BatchNormalization())

model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(128, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(128, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(num_classes, activation="softmax"))

print(model.summary())

model.compile(loss="categorical_crossentropy",
				optimizer=Adam(lr=0.0001),
				metrics=['accuracy'])

train_steps = math.ceil(train_generator.n/train_generator.batch_size)
val_steps = math.ceil(val_generator.n/val_generator.batch_size)
epochs = 100

history = model.fit(train_generator,
					epochs = epochs,
					steps_per_epoch = train_steps,
					validation_data = val_generator,
					validation_steps = val_steps)

scores = model.evaluate_generator(val_generator, steps = val_steps, verbose=1)
print('\n Validation Accuracy:%.3f Validation Loss:%.3f'%(scores[1]*100, scores[0]))


model.save_weights('model_100_epochs.h5')

model_json = model.to_json()
with open('model_adam_1.json','w') as json_file:
	json_file.write(model_json)

print('Model saved to the disk')

val_predict = model.predict(val_generator, steps=val_steps, verbose=1)
val_labels = np.argmax(val_predict)
print(val_labels)


test_steps = math.ceil(test_generator.n/test_generator.batch_size)

test_predict = model.predict(test_generator, steps=test_steps, verbose=1)
test_labels = np.argmax(test_predict)
print(test_labels)






















