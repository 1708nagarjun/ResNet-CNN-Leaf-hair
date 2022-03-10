#!/usr/bin/env python
# coding: utf-8
import os
import image_slicer
import shutil
from tqdm import tqdm
from datetime import datetime
import pandas as pd
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import sys
import cv2

# Pipeline Parameters
img_size = (112, 112)
model_back_vs_leaf = keras.models.load_model('back-leaf-v2-e30-acc98.h5')
model_hair_vs_nohair = keras.models.load_model(
    'hair-nohair-v4d5e30-acc95.41.h5')

# User Inputs
source_dir = "./input/"
dest_dir = "./output/"
temp_dir = "./tmp/"

# Global Vars
sample = 0
i = 0
leaf_disc_imgs = []
a = ''
num_img = 0
results = []


def getColor(val):
    if (val == 1):
        return (0, 0, 0)
    elif (val == 2):
        return (0, 0, 0)
    else:
        return (0, 0, 0)


def input():
    if len(sys.argv) == 3:
        if os.path.isdir(sys.argv[1]) == True:
            global source_dir
            source_dir = sys.argv[1]
        else:
            print("Leaf disc image folder does not exist")
            exit()

        if sys.argv[2] == '':
            print("No experiment name given.")
            exit()
        else:
            global dest_dir
            if os.path.isdir(os.path.join(dest_dir, sys.argv[2])) == True:
                print("Leaf disc image folder already exists")
                exit()
            else:
                dest_dir = os.path.join(dest_dir, sys.argv[2])
                os.makedirs(dest_dir)
    else:
        print(sys.argv)
        exit("Usage: python classifier.py /path/to/leaf/disc experiment_name")


# Index Images
def index_images():
    global num_img
    for image in sorted(os.listdir(source_dir)):
        if image.endswith('.jpg'):
            a = os.path.join(source_dir, image)
            num_img += 1
            leaf_disc_imgs.append(a)
        else:
            continue


# Slice Images
def slice_image(image):
    leaf_disc_name = os.path.basename(image)
    tiles = image_slicer.slice(image, 500, save=False)
    slice_dest = os.path.join(temp_dir, leaf_disc_name[:-4])
    os.makedirs(os.path.abspath(slice_dest))
    image_slicer.save_tiles(tiles, directory=slice_dest,
                            prefix=leaf_disc_name, format='png')

#  Predict Back vs Leaf


def pred_back_leaf(img_slice):
    img = load_img(img_slice, target_size=img_size)
    x = img_to_array(img)
    x = tf.expand_dims(x, 0)
    return (model_back_vs_leaf.predict(x) > 0.5).astype("int32")

#  Predict Hair vs NoHair


def pred_hair_nohair(img_slice):
    img = load_img(img_slice, target_size=img_size)
    x = img_to_array(img)
    x = tf.expand_dims(x, 0)
    return (model_hair_vs_nohair.predict(x) > 0.5).astype("int32")


def index_slices(img):
    slice_dir = os.path.join(temp_dir, os.path.basename(img)[:-4])
    slices = []
    for image in sorted(os.listdir(slice_dir)):
        if image.endswith('.png'):
            a = os.path.join(slice_dir, image)
            slices.append(a)
        else:
            continue
    return slices


def append_results(data):
    global results
    results.append(data)


def output_csv():
    global results
    global dest_dir
    results_df = pd.DataFrame.from_records(results, columns=[
                                           "img_name", "back", "hair", "nohair", "perc_hair", "perc_nohair"])
    dt_obj = datetime.now()
    filename = "results" + dt_obj.strftime("%d-%m-%Y_%H-%M-%S") + ".csv"
    results_df.to_csv(os.path.join(dest_dir, filename), index=None)


def main():
    #     Clear Global Vars
    global num_img
    global sample
    global i
    global leaf_disc_imgs
    global a
    global results
    results = []
    sample = 0
    i = 0
    leaf_disc_imgs = []
    a = ''
    num_img = 0

#     Check Environment
    if not os.path.exists(source_dir):
        print("Error: Invalid Input Directory")
        exit(-1)

    if not os.path.exists(dest_dir):
        print("Error: Invalid Output Directory")
        exit(-1)

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

#   Empty Temp Dir
    shutil.rmtree(temp_dir)

#   Index Images
    index_images()
    print(str(num_img) + " images found.")

#   Slice Images
    print("Slicing Images...")
    for img in tqdm(leaf_disc_imgs):
        slice_image(img)

#   Classify Images
    for img in leaf_disc_imgs:
        print("Classifying " + str(os.path.basename(img)))
#         Init Vars
        leaf = []
        back = []
        hair = []
        nohair = []

#       Index Slices
        slices = index_slices(img)

        slicemap = []

        pbar = tqdm(total=506)
#         Classify Back vs Leaf
        for slice in slices:
            result = pred_back_leaf(slice)
            if result:
                leaf.append(slice)
            else:
                back.append(slice)
                slicemap.append((str(os.path.basename(slice)[:-4]), 0))
                pbar.update(1)

#         Classify Hair vs Nohair
        for slice in leaf:
            result = pred_hair_nohair(slice)
            if result:
                nohair.append(slice)
                slicemap.append((str(os.path.basename(slice)[:-4]), 1))
                pbar.update(1)
            else:
                hair.append(slice)
                slicemap.append((str(os.path.basename(slice)[:-4]), 2))
                pbar.update(1)

        pbar.close()
#       Write mapfile to csv
        # slicemap_df = pd.DataFrame.from_records(slicemap, columns=[
        #     "slice_name", "type"])
        # filename = str(os.path.basename(img)) + ".csv"
        # slicemap_df.to_csv(os.path.join(dest_dir, filename), index=None)

        # Write Visualization to disk
        src_img = cv2.imread(img)
        height, width, channels = src_img.shape
        for square in slicemap:
            x = square[0][-2:]
            y = square[0][-5:-3]
            pt1 = (int((int(width)/23)*(int(x)-1)),
                   int((int(height)/22)*(int(y)-1)))
            pt2 = (int((int(width)/23)*int(x)),
                   int((int(height)/22)*int(y)))
            clr = getColor(square[1])
            cv2.rectangle(src_img, pt1, pt2, clr, 5)

        filename = str(os.path.basename(img))
        cv2.imwrite(os.path.join(dest_dir, filename), src_img)

#         Calculate results
        percentage_hair = (len(hair)/len(leaf))*100
        append_results((os.path.basename(img), len(back), len(
            hair), len(nohair), percentage_hair, 100-percentage_hair))
    output_csv()


if __name__ == '__main__':
    input()
    main()
