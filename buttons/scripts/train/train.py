import numpy as np
from os.path import join
from keras import Model, Input
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback


def read_dataset(dataset_path="data/processed_data"):
    x_train = np.load(join(dataset_path, "x_train.npy"))
    y_train = np.load(join(dataset_path, "y_train.npy"))
    y_train = to_categorical(y_train)
    return x_train, y_train


def get_model():
    input = Input(shape=(32, 32, 3))
    x = Conv2D(32, (3, 3), activation="relu")(input)
    x = MaxPooling2D()(x)
    x = Conv2D(16, (3, 3), activation="relu")(x)
    x = MaxPooling2D()(x)
    x = Conv2D(8, (3, 3), activation="relu")(x)
    x = Flatten()(x)
    output = Dense(6, activation="softmax")(x)
    model = Model(inputs=input, outputs=output)
    print(model.summary())
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics="accuracy")
    return model


def train_model(model, dataset):
    x_train, y_train = dataset
    run = neptune.init(
        project="galb4tosha/but-trans-dis",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkN2UwNjk5ZS1jZjBiLTQ3OTUtODhkOC0yYjJmNjY4ZDM5NTYifQ==",
    )
    neptune_cbk = NeptuneCallback(run=run, base_namespace="training")
    vat_acc_checkpoint = ModelCheckpoint(
        "models/best_model.h5", monitor="val_accuracy", mode="max", save_best_only=True
    )
    model.fit(
        x_train,
        y_train,
        batch_size=64,
        epochs=250,
        validation_split=0.2,
        callbacks=[vat_acc_checkpoint, neptune_cbk],
    )


if __name__ == "__main__":
    dataset = read_dataset()
    model = get_model()
    train_model(model, dataset)
