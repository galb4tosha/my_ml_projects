import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Flatten
from keras import Model, Input, Sequential
import numpy as np 
import os
import click

from clearml import Task

# @click.command()
# @click.argument("x_train_path", type=click.Path(exists=True))
# @click.argument("y_train_path", type=click.Path(exists=True))
# @click.argument("save_path", type=click.Path())
def read_data(x_train_path, y_train_path, save_path):
    x_data = np.load(x_train_path).reshape((42000, 28, 28))
    y_data = np.load(y_train_path)
    save = save_path
    return x_data, y_data, save

def blocks(in_layer, filters, n_blocks): 
    
    for _ in range(n_blocks):
        in_layer = Conv2D(filters, (3,3), activation='relu', padding='same', strides=(1,1))(in_layer)   
    in_layer = MaxPooling2D((2,2))(in_layer)

    return in_layer


def classifier(in_layer):
    flat = Flatten()(in_layer)
    dense = Dense(128, activation='relu')(flat)
    dense2 = Dense(64, activation='relu')(dense)
    last_layer =  Dense(10, activation='softmax')(dense2)
    return last_layer

def get_model():
    in_ = Input(shape=(28,28,1))
    conv1 = blocks(in_, 32, 2)
    conv2 = blocks(conv1, 64, 2)
    out_ = classifier(conv2)

    model = Model(inputs=in_, outputs=out_)
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics='accuracy')

    print(model.summary())
    return model

def train_and_save_model(x_train, y_train, model, save_path):
    model.fit(x_train, y_train, batch_size=64, epochs=50)
    model.save(save_path)

if __name__ == "__main__":
    task = Task.init(project_name="My Project", task_name="My Experiment")
    x_train, y_train, save_path = read_data("data/processed_data/x_train.npy", "data/processed_data/y_train.npy", "models/mnist.h5")
    model = get_model()
    train_and_save_model(x_train, y_train, model, save_path)
