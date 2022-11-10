from tensorflow.keras import datasets, layers, models, optimizers
import tensorflow as tf
from keras.callbacks import ModelCheckpoint

def get_custom_vgg_11():
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(512, (3, 3), activation='relu'))
    # model.add(layers.Conv2D(512, (3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(512, (3, 3), activation='relu'))
    # model.add(layers.Conv2D(512, (3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    # model.add(layers.Dense(4096, activation='relu'))
    # model.add(layers.Dense(4096, activation='relu'))
    # model.add(layers.Dense(1000, activation='relu'))

    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(100, activation='relu'))
    
    model.add(layers.Dense(10, activation='softmax'))
    
    print(model.summary())
    return model

def train_model():
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    
    model = get_custom_vgg_11()
    lr_schedule = optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-2,
                                                            decay_steps=100000,
                                                            decay_rate=0.9)
    optimizer = optimizers.SGD(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer,
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy']
                    )
    val_acc_checkpoint = ModelCheckpoint(
        "vgg_models/models/best_model.h5", monitor="val_accuracy", mode="max", save_best_only=True
    )
    _ = model.fit(train_images, train_labels, epochs=100, 
                    validation_data=(test_images, test_labels),
                    callbacks=[val_acc_checkpoint])
    _, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print(test_acc)

train_model()
