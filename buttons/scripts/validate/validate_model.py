from keras.models import load_model
from keras.utils import to_categorical
import numpy as np
from os.path import join


def read_dataset(dataset_path="data/processed_data", ):
    x_test = np.load(join(dataset_path, "x_test.npy"))
    y_test = np.load(join(dataset_path, "y_test.npy"))
    y_test = to_categorical(y_test)
    return x_test, y_test

def validate_model(dataset, model_path="models/best_model.h5"):
    model = load_model(model_path)
    x_test, y_test = dataset
    model.evaluate(x_test, y_test)
    
    

if __name__ == "__main__":
    dataset = read_dataset()
    validate_model(dataset)
