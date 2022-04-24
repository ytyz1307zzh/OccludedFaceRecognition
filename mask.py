"""
Mask (blackout) part of the face (upper face or lower face) to generate synthetic data.
Synthetic data are later used for data augmentation.
"""

import os
import cv2
import json
import argparse
from tqdm import tqdm
from copy import deepcopy
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

EYE_WIDTH_RATIO = 4 / 15
EYE_HEIGHT_RATIO = 1 / 4
MOUTH_WIDTH_RATIO = 10 / 37
MOUTH_HEIGHT_RATIO = 1 / 12

parser = argparse.ArgumentParser()
parser.add_argument('-data_dir', default='ROF_data', type=str, help="Data directory.")
args = parser.parse_args()

neutral_dir = os.path.join(args.data_dir, "neutral")
subject_name_list = os.listdir(neutral_dir)

for subject_name in tqdm(subject_name_list):
    subject_data_dir = os.path.join(neutral_dir, subject_name)
    image_name_list = [path for path in os.listdir(subject_data_dir) if path.endswith('.jpg')]
    # File containing box data (boudaries for eyes, noses, mouths, etc)
    facebox_data = json.load(open(os.path.join(subject_data_dir, 'face_box-new.json'), 'r', encoding='utf8'))

    for image_name in image_name_list:
        if image_name.endswith("-lower.jpg") or image_name.endswith("-upper.jpg"):
            continue
        image_path = os.path.join(subject_data_dir, image_name)
        image_id = image_name[:-4]  # Get rid of ".jpg"

        image = cv2.imread(image_path)
        box_info = facebox_data[image_id]
        size = box_info["box"]
        width, height = size[2], size[3]

        # Parse the box information
        if "keypoints" in box_info.keys():
            left_eye = box_info["keypoints"]["left_eye"]
            right_eye = box_info["keypoints"]["right_eye"]
            nose = box_info["keypoints"]["nose"]
            mouth_left = box_info["keypoints"]["mouth_left"]
            mouth_right = box_info["keypoints"]["mouth_right"]
        else:
            left_eye = box_info["left_eye"]
            right_eye = box_info["right_eye"]
            nose = box_info["nose"]
            mouth_left = box_info["mouth_left"]
            mouth_right = box_info["mouth_right"]

        # Upper-occluded image: Mask out the eye area
        upper_occluded_image = deepcopy(image)
        eye_upleft = [max(left_eye[0] - EYE_WIDTH_RATIO * width / 2, 0), max(left_eye[1] - EYE_HEIGHT_RATIO * height / 2, 0)]  # (x, y)
        eye_lowright = [right_eye[0] + EYE_WIDTH_RATIO * width / 2, right_eye[1] + EYE_HEIGHT_RATIO * height / 2]
        eye_width = eye_lowright[0] - eye_upleft[0]  # x1 - x2
        eye_height = eye_lowright[1] - eye_upleft[1]  # y1 - y2
        contours = np.array([eye_upleft, [eye_upleft[0], eye_upleft[1] + eye_height],
                             eye_lowright, [eye_upleft[0] + eye_width, eye_upleft[1]]]).astype(np.int32)
        cv2.fillPoly(upper_occluded_image, pts=[contours], color=(0, 0, 0))
        pil_upper_occluded_image = Image.fromarray(upper_occluded_image)
        with open(os.path.join(subject_data_dir, f"{image_id}-upper.jpg"), "wb") as f:
            pil_upper_occluded_image.save(f, "JPEG", quality=85)

        # Lower-occluded image: Mask out the nose and mouth area
        lower_occluded_image = deepcopy(image)
        mouth_upleft = [max(mouth_left[0] - MOUTH_WIDTH_RATIO * width / 2, 0), max(nose[1] - MOUTH_HEIGHT_RATIO * height, 0)]
        mouth_lowright = [mouth_right[0] + MOUTH_WIDTH_RATIO * width / 2, height]
        mouth_width = mouth_lowright[0] - mouth_upleft[0]
        mouth_height = mouth_lowright[1] - mouth_upleft[1]
        contours = np.array([mouth_upleft, [mouth_upleft[0], mouth_upleft[1] + mouth_height],
                             mouth_lowright, [mouth_upleft[0] + mouth_width, mouth_upleft[1]]]).astype(np.int32)
        cv2.fillPoly(lower_occluded_image, pts=[contours], color=(0, 0, 0))
        pil_lower_occluded_image = Image.fromarray(lower_occluded_image)
        with open(os.path.join(subject_data_dir, f"{image_id}-lower.jpg"), "wb") as f:
            pil_lower_occluded_image.save(f, "JPEG", quality=85)


