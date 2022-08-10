import cv2 
import numpy as np
import os
import random


if __name__ == "__main__":
    row_data_path = "data/raw/crop_data_labeled"
    y_dict = {
        "button_off": 0,
        "button_on": 1,
        "display_off": 2,
        "display_on": 3,
        "switcher_off": 4,
        "switcher_on": 5,
    }
    x_train = list()
    y_train = list()
    x_test = list()
    y_test = list()
    for dir in os.listdir(row_data_path):
        if dir != ".DS_Store":
            for file_path in os.listdir(os.path.join(row_data_path, dir)):
                if file_path != ".DS_Store":
                    img_path = os.path.join(row_data_path, dir, file_path)
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, (32, 32))
                    img = img / 255.0
                    dataset = random.choices(["train", "test"], weights=[9, 1])
                    if dataset == ["train"]:
                        x_train.append(img)
                        y_train.append(y_dict[dir])
                    else:
                        x_test.append(img)
                        y_test.append(y_dict[dir])
    print(len(x_train), len(x_test))
    np.save("data/processed_data/x_train", x_train)
    np.save("data/processed_data/y_train", y_train)
    np.save("data/processed_data/x_test", x_test)
    np.save("data/processed_data/y_test", y_test)