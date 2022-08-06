import tensorflow.keras as keras


class DataGenerator(keras.utils.Sequence):
    def __init__(self):
        super().__init__()
