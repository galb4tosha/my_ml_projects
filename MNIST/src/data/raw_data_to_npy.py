from genericpath import isdir
import numpy as np
import click
import os

@click.command()
@click.argument("train_data_path", type=click.Path(exists=True))
@click.argument("test_data_path", type=click.Path(exists=True))
@click.argument("save_dir", type=click.Path())
def preprocess_and_save_data(train_data_path: str, test_data_path: str, save_dir: str):
    x_train = np.genfromtxt(train_data_path, delimiter=",", skip_header=True)
    y_train = x_train[:, 0]
    x_train = x_train[:, 1:]/255.0
    x_test = np.genfromtxt(test_data_path, delimiter=",", skip_header=True)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    np.save(os.path.join(save_dir, "x_train.npy"), x_train)
    np.save(os.path.join(save_dir, "y_train.npy"), y_train)
    np.save(os.path.join(save_dir, "x_test.npy"), x_test)

if __name__ == "__main__":
    preprocess_and_save_data()
