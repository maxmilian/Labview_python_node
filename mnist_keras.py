import numpy as np
import keras
from tensorflow.keras.models import load_model
import logging

logging.basicConfig(filename='mnist.log', level=logging.DEBUG, format='%(asctime)s %(message)s')
model_file = 'mniss_keras.hdf5'

def model_training(_optimizer, _loss, _metrics, _epochs):
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    class_names = ['0', '1', '2', '3', '4',
                   '5', '6', '7', '8', '9']
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=_optimizer,loss=_loss,metrics=[_metrics])
    model.fit(train_images, train_labels, epochs=_epochs)
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    results = [test_loss, test_acc]
    model.save(model_file)
    return results

def predict(_image):
    logging.debug(_image)
    _image = np.array(_image)
    _image =_image.reshape(1, 28, 28)
    model = load_model(model_file)
    predictions = model.predict(_image)
    return predictions[0]
