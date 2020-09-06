import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os

import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

import pydicom

app = Flask(__name__)

def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def create_model():
    input_img = Input((224, 224, 1), name='img')
    n_filters=16 
    dropout=0.05 
    batchnorm=True
    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Dropout(dropout*0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2)) (c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
    
    # expansive path
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
    
model = create_model()
model.load_weights("model.h5")
uploads_dir = os.path.join(app.instance_path, 'uploads')
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    f = request.files['file']  
    f.save(os.path.join(uploads_dir,f.filename))
    print("Stored in: ",os.path.join(uploads_dir,f.filename))
    img = pydicom.dcmread(os.path.join(uploads_dir,f.filename)).pixel_array
    print("Pixel Array Received")
    img = resize(img, (256, 256), mode='reflect')
    img = np.expand_dims(img, axis=-1)
    prediction=model.predict(img)
    print("Model Array Received")
    pred = resize(pred, (1024, 1024), mode='reflect')
    comp = pred[:, :, 0] > 0.5
    comp = measure.label(comp)
    conf=0
    for region in measure.regionprops(comp):
        y, x, y2, x2 = region.bbox
        height = y2 - y
        width = x2 - x
        conf = np.mean(pred[y:y+height, x:x+width])
    
    if conf>0.50:
        output = 'Pneumonia Positive'
    else:
        output = 'Pneumonia Negative'
        
    data = [{'name': output}]
    return jsonify(data)

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    f = request.files['file']  
    f.save(os.path.join(uploads_dir,f.filename))
    print("Stored in: ",os.path.join(uploads_dir,f.filename))
    img = pydicom.dcmread(os.path.join(uploads_dir,f.filename)).pixel_array
    print("Pixel Array Received")
    img = resize(img, (256, 256), mode='reflect')
    img = np.expand_dims(img, axis=-1)
    prediction=model.predict(img)
    print("Model Array Received")
    pred = resize(pred, (1024, 1024), mode='reflect')
    comp = pred[:, :, 0] > 0.5
    comp = measure.label(comp)
    conf=0
    for region in measure.regionprops(comp):
        y, x, y2, x2 = region.bbox
        height = y2 - y
        width = x2 - x
        conf = np.mean(pred[y:y+height, x:x+width])
    
    if conf>0.50:
        output = 'Pneumonia Positive'
    else:
        output = 'Pneumonia Negative'
        
    data = [{'name': output}]
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)