import csv
import hashlib
import os
from _csv import reader

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np


def remove_duplicates(dataset_path):
    files_list = list()
    for root, dirs, filename in os.walk(dataset_path):
        for img in filename:
            files_list.append(os.path.join(root, img))

    duplicates = []
    hash_keys = dict()
    for index, filename in enumerate(files_list):
        if os.path.isfile(filename):
            with open(filename, 'rb') as f:
                filehash = hashlib.md5(f.read()).hexdigest()
            if filehash not in hash_keys:
                hash_keys[filehash] = index
            else:
                duplicates.append((index, hash_keys[filehash]))

    for index in duplicates:
        os.remove(files_list[index[0]])


def plot_random_duplicates(files_list, duplicates):
    for file_indexes in duplicates[:len(duplicates)]:
        try:

            img = mpimg.imread(files_list[file_indexes[1]])
            plt.subplot(121), plt.imshow(img)
            plt.title(file_indexes[1]), plt.xticks([]), plt.yticks([])

            img = mpimg.imread(files_list[file_indexes[0]])
            plt.subplot(122), plt.imshow(img)
            plt.title(str(file_indexes[0]) + ' duplicate'), plt.xticks([]), plt.yticks([])
            plt.show()

        except OSError as e:
            print(f"File corrupted {e.args[0]}")


def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


def resize_images(image_dir, format='.png'):
    for root, dirs, files in os.walk(image_dir, topdown=False):
        for name in files:
            if name.endswith(format):
                try:
                    file_path = os.path.join(root, name)
                    im = Image.open(file_path)
                    im_resize = im.resize((50, 50), Image.ANTIALIAS)
                    im_resize.save(file_path, 'PNG', quality=90)
                except IOError:
                    print("Unable to find %s image" % file_path)


def images_to_csv(dataset_dir, format='.png'):
    file_list = []
    for root, dirs, files in os.walk(dataset_dir, topdown=False):
        for filename in files:
            if filename.endswith('.png'):
                try:
                    full_name = os.path.join(root, filename)
                    img = Image.open(full_name)  # open the image file
                    img.verify()  # verify that it is, in fact an image
                    file_list.append(full_name)
                except (IOError, SyntaxError) as e:
                    print('Bad file:', filename)

    for index, file in enumerate(file_list):
        img_file = Image.open(file)
        width, height = img_file.size
        format = img_file.format
        mode = img_file.mode

        # Make image Greyscale
        img_grey = img_file.convert('L')
        # img_grey.save('result.png')
        # img_grey.show()

        # Save Greyscale values
        value = np.asarray(img_grey.getdata(), dtype=np.float)  # .reshape((-1,1))
        value = value.flatten()
        file = file.split('/')  # Getting Alphabet name
        index = np.array([file[-2]])  # Adding at the end of row
        value = np.append(value, index)

        with open("htlr.csv", 'a') as f:
            writer = csv.writer(f)
            writer.writerow(value)
