# this code will save model.hdf5 and model.json files in the folder in which this code runs

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import keras
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from kerasbestfit import kbf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# load the built-in Keras MNIST dataset and split into train and validation datasets
(mnist_train_images, mnist_train_labels), (mnist_test_images, mnist_test_labels) = mnist.load_data()
train_images = mnist_train_images.reshape(60000, 784)
test_images = mnist_test_images.reshape(10000, 784)
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images /= 255
test_images /= 255
train_labels = keras.utils.to_categorical(mnist_train_labels, 10)
test_labels = keras.utils.to_categorical(mnist_test_labels, 10)

# create model
model = Sequential()
model.add(Dense(784, activation='elu', input_shape=(784,)))
model.add(Dense(1000, activation='elu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# define log function that is passed to kbf.find_best_fit
def log_msg(msg=''):
    print(msg)
    return


# train the model and return the best fit
results, log = kbf.find_best_fit(model=model,
                                 metric='val_acc',
                                 xtrain=train_images,
                                 ytrain=train_labels,
                                 xval=test_images,
                                 yval=test_labels,
                                 validation_split=0,
                                 batch_size=500,
                                 epochs=5,
                                 patience=5,
                                 snifftest_max_epoch=0,
                                 snifftest_metric_val=0,
                                 show_progress=True,
                                 format_metric_val='{:1.10f}',
                                 save_best=True,
                                 save_path='',
                                 best_metric_val_so_far=0,
                                 logmsg_callback=log_msg,
                                 finish_by=0)
