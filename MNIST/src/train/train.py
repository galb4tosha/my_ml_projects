import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Flatten
from keras import Model, Input, Sequential
import numpy as np 
import os
import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback

def read_data(x_train_path, y_train_path):
    x_data = np.load(x_train_path).reshape((42000, 28, 28))
    y_data = np.load(y_train_path)
    return x_data, y_data

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

def train_and_save_model(x_train, y_train, model):
    save_path = "models/mnist.h5"
    run = neptune.init(
        project="galb4tosha/test-project",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkN2UwNjk5ZS1jZjBiLTQ3OTUtODhkOC0yYjJmNjY4ZDM5NTYifQ==",
    )  # your credentials
    neptune_cbk = NeptuneCallback(run=run, base_namespace="training")
    model.fit(x_train, y_train, batch_size=128, epochs=50, callbacks=[neptune_cbk],)
    model.save(save_path)

if __name__ == "__main__":
    x_train, y_train = read_data("data/processed_data/x_train.npy", "data/processed_data/y_train.npy")
    model = get_model()
    train_and_save_model(x_train, y_train, model)
