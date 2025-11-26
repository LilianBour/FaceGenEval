#Source : https://github.com/LynnHo/HD-CelebA-Cropper
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import csv
import dlib
import functools
import re
import tqdm
import numpy as np
from functools import partial
from multiprocessing import Pool

import cropper
from extract_68_landmarks import extract_landmarks
from pathlib import Path


# Path to current file
current_dir = Path(__file__).resolve().parent
# Path to project root
project_root = current_dir.parent

_DEFAULT_JPG_QUALITY = 95
imwrite = partial(cv2.imwrite, params=[int(cv2.IMWRITE_JPEG_QUALITY), _DEFAULT_JPG_QUALITY])
align_crop = cropper.align_crop_opencv

######## Functions to Align images of faces
def work(n_landmark,save_dir,save_format,mode,order,align_type,face_factor,crop_size_h,crop_size_w,standard_landmark,landmarks,img_dir,img_names,i):  # a single work
    img = cv2.imread(os.path.join(img_dir, img_names[i]))
    img_crop, tformed_landmarks = align_crop(img, landmarks[i], standard_landmark,crop_size=(crop_size_h, crop_size_w), face_factor=face_factor,align_type=align_type, order=order, mode=mode)
    name = os.path.splitext(img_names[i])[0] + '.' + save_format
    path = os.path.join(save_dir, name)

    if not os.path.isdir(os.path.split(path)[0]):
        os.makedirs(os.path.split(path)[0])
    imwrite(path, img_crop)

    tformed_landmarks.shape = -1
    name_landmark_str = ('%s' + ' %.1f' * n_landmark * 2) % ((name,) + tuple(tformed_landmarks))
    return name_landmark_str

def align_folder(path_images,size=1024):
    """
    Align images using dlib landmarks
    Args:
        path_images: path to folder containing images to align, a new folder called aligned with aligned images will be created at the same level
        size: outpute size of images

    """
    #GLOBAL PARAMS
    root_folder = path_images

    #------------------PART 1 : LANDMARKS EXTRACTION -------------------------------
    # Load Model
    Model_PATH = project_root + "/pretrained_models/shape_predictor_68_face_landmarks.dat"
    frontalFaceDetector = dlib.get_frontal_face_detector()
    faceLandmarkDetector = dlib.shape_predictor(Model_PATH)

    list_images = os.listdir(root_folder + "img/")
    list_landmarks = []
    bad_images = []
    for img in list_images:
        landmarks_for_img = extract_landmarks(root_folder + "img/" + img, frontalFaceDetector, faceLandmarkDetector)
        if landmarks_for_img == []:
            bad_images.append(img)
        else:
            list_landmarks.append(landmarks_for_img)

    #Save images where landmarks could not be extracted aside
    if not os.path.isdir(root_folder + "bad_imgs/"):
        os.makedirs(root_folder + "bad_imgs/")
    for bad_img in bad_images:
        img = cv2.imread(root_folder + "img/" + bad_img)
        cv2.imwrite(root_folder + "bad_imgs/" + bad_img, img)

    #Save landmarks in txt
    with open(root_folder + "landmarks.txt", "w") as f:
        wr = csv.writer(f, delimiter=" ")
        wr.writerows(list_landmarks)

    # --------------------------PART 2 :  Align images-------------------------------
    # Params
    img_dir = root_folder + "img"
    save_dir = root_folder + 'aligned'
    landmark_file = root_folder + 'landmarks.txt'
    standard_landmark_file = 'standard_landmarks.txt'
    crop_size_h = size
    crop_size_w = size
    move_h = 0.25
    move_w = 0.0
    save_format = 'png'  # jpg or png
    n_worker = 8

    # others
    face_factor = 0.55  # The factor of face area relative to the output image.
    align_type = 'affine'  # 'affine', 'similarity'
    order = 5  # 0, 1, 2, 3, 4, 5 #The order of interpolation.
    mode = 'edge'  # ['constant', 'edge', 'symmetric', 'reflect', 'wrap']

    # count landmarks
    with open(landmark_file) as f:
        line = f.readline()
    n_landmark = len(re.split('[ ]+', line)[1:]) // 2

    # read data
    img_names = np.genfromtxt(landmark_file, dtype=str, usecols=0)
    landmarks = np.genfromtxt(landmark_file, dtype=float, usecols=range(1, n_landmark * 2 + 1)).reshape(-1,n_landmark,2)
    standard_landmark = np.genfromtxt(standard_landmark_file, dtype=float).reshape(n_landmark, 2)
    standard_landmark[:, 0] += move_w
    standard_landmark[:, 1] += move_h

    # data dir
    save_dir = os.path.join(save_dir)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    pool = Pool(n_worker)
    name_landmark_strs = list(tqdm.tqdm(pool.imap(
        functools.partial(
            work, n_landmark,save_dir,save_format,mode,order,align_type,face_factor,crop_size_h,crop_size_w,standard_landmark,landmarks,img_dir,img_names
        ),range(img_names.size)
    ), total=img_names.size))
    "name_landmark_strs = list(tqdm.tqdm(pool.imap(work, range(img_names.size)), total=img_names.size))"
    pool.close()
    pool.join()