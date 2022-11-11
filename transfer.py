""" transfer numpy array to png images """

import numpy as np
from PIL import Image
import os

def transfer_h5_to_png(x_train, y_train, x_test, y_test, target_root_dir):
    # if y is like: [0., 0., ... 1., 0., ...] one-hot, need to transfer it into label index(0~42)
    # here, y_train/y_test's shape: (num, 43)
    y_train = np.where(y_train == 1.)[1]
    y_test = np.where(y_test == 1.)[1]
    
    for index, img_array in enumerate(x_train):
        target_train_root = os.path.join(target_root_dir, 'train')
        if not os.path.exists(target_train_root):
            os.mkdir(target_train_root)
        label = y_train[index]
        target_label_dir = os.path.join(target_train_root, str(label))
        if not os.path.exists(target_label_dir):
            os.mkdir(target_label_dir)
        img = Image.fromarray(np.uint8(img_array))
        final_path = os.path.join(target_label_dir, str(index) + '.png')
        img.save(final_path)

    for index, img_array in enumerate(x_test):
        target_test_dir = os.path.join(target_root_dir, 'test')
        if not os.path.exists(target_test_dir):
            os.mkdir(target_test_dir)
        label = y_test[index]
        target_label_dir = os.path.join(target_test_dir, str(label))
        if not os.path.exists(target_label_dir):
            os.mkdir(target_label_dir)
        img = Image.fromarray(np.uint8(img_array))
        final_path = os.path.join(target_label_dir, str(index) + '.png')
        img.save(final_path)

transfer_h5_to_png(X_train, Y_train, X_test, Y_test, 'GTSRB')
