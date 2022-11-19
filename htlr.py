# USAGE
# python htlr.py --save-model 1 --weights output/htlr.hdf5
# python htlr.py --load-model 1 --weights output/htlr.hdf5


from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
import argparse
import cv2

import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os as os
from csv import reader

from networks.lenet import LeNet


def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = list()
    [unique.append(x) for x in class_values if x not in unique]
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


def main(args):
    curr_dir = os.getcwd()
    dataset = np.array(load_csv(curr_dir + '/htlr.csv'))

    class_values = str_column_to_int(dataset, len(dataset[0]) - 1)

    train_x, train_y = dataset[:, :-1], dataset[:, [-1]]

    train_x, train_y = shuffle(train_x, train_y, random_state=0)
    train_data, test_data, train_labels, test_labels = train_test_split(train_x, train_y, test_size=0.10)

    train_data = train_data.reshape(len(train_data), 50, 50).astype(np.float)

    if K.image_data_format() == "channels_first":
        train_data = train_data.reshape((train_data.shape[0], 1, 50, 50))
        test_data = test_data.reshape((test_data.shape[0], 1, 50, 50))

    else:
        train_data = train_data.reshape((train_data.shape[0], 50, 50, 1))
        test_data = test_data.reshape((test_data.shape[0], 50, 50, 1))

    train_data = train_data.astype("float32") / 255.0
    test_data = test_data.astype("float32") / 255.0

    train_labels = np_utils.to_categorical(train_labels, 29)
    test_labels = np_utils.to_categorical(test_labels, 29)

    print("[INFO] compiling model...")
    opt = SGD(lr=0.01)
    model = LeNet.build(num_channels=1, img_rows=50, img_cols=50,
                        num_classes=29,
                        weights_path=args["weights"] if args["load_model"] > 0 else None)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    if args["load_model"] < 0:
        print("[INFO] training...")
        model.fit(train_data, train_labels, batch_size=128, epochs=40,
                  verbose=1)

        print("[INFO] evaluating...")
        (loss, accuracy) = model.evaluate(test_data, test_labels,
                                          batch_size=128, verbose=1)
        print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

    if args["save_model"] > 0:
        print("[INFO] dumping weights to file...")
        model.save_weights(args["weights"], overwrite=True)

    for i in np.random.choice(np.arange(0, len(test_labels)), size=(100,)):
        probs = model.predict(test_data[np.newaxis, i])
        prediction = probs.argmax(axis=1)
        if K.image_data_format() == "channels_first":
            image = (test_data[i][0] * 255).astype("uint8")
        else:
            image = (test_data[i] * 255).astype("uint8")
        image = cv2.merge([image] * 3)
        image = cv2.resize(image, (196, 196), interpolation=cv2.INTER_LINEAR)
        actual = ''
        predicted = ''
        cv2.putText(image, str(actual), (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        for value, index in class_values.items():
            if index == prediction[0]:
                predicted = value
            if index == np.argmax(test_labels[i]):
                actual = value

        print("[INFO] Predicted: {}, Actual: {}".format(predicted,
                                                        actual))
        cv2.imshow("Digit", image)
        cv2.waitKey(0)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--save-model", type=int, default=-1,
                    help="(optional) whether or not model should be saved to disk")
    ap.add_argument("-l", "--load-model", type=int, default=-1,
                    help="(optional) whether or not pre-trained model should be loaded")
    ap.add_argument("-w", "--weights", type=str,
                    help="(optional) path to weights file")
    args = vars(ap.parse_args())

    main(args)
