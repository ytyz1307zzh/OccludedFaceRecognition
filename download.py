"""
Download image files given the original data files from RealWorldOccludedFaces
Github repo for RealWorldOccludedFaces: https://github.com/ekremerakin/RealWorldOccludedFaces
"""

import os
import io
import json
import pickle
from tqdm import tqdm
import requests
import argparse
from PIL import Image
import numpy as np


def url2image(url):
    try:
        r = requests.get(url, timeout=4)
        if r.status_code != 200:
            return None

        image_file = io.BytesIO(r.content)
        img = Image.open(image_file).convert('RGB')
    except Exception as e:
        return None

    return img


parser = argparse.ArgumentParser()
parser.add_argument('-raw_data_dir', default='../RealWorldOccludedFaces/ROF/neutral', type=str,
                    help='Path to the raw data directory obtained from RealWorldOccludedFaces github repo.')
parser.add_argument('-save_dir', default="ROF_data", type=str, help="Directory to save processed data.")
args = parser.parse_args()

# Directory to store the downloaded images
main_path = args.save_dir
os.makedirs(main_path, exist_ok=True)
# Command line argument: the directory containing pickle files (which include image URLs and metadata)
image_src_dir = args.raw_data_dir
# image_src_type: masked, neutral or sunglasses
image_src_type = image_src_dir.split('/')[-1]

for filename in tqdm(os.listdir(image_src_dir)):
    assert filename.endswith(".pkl")
    file_prefix = filename[:-4]

    if image_src_type == "neutral":
        subject_name = file_prefix
    elif image_src_type == "masked":
        subject_name = file_prefix[:-13]  # Ends with "_wearing_mask"
    elif image_src_type == "sunglasses":
        subject_name = file_prefix[:-20]  # Ends with "_wearning_sunglasses"
    else:
        raise ValueError(f"Invalid image_src_type: {image_src_type}!")

    # Create directory to store the data
    os.makedirs(os.path.join(main_path, image_src_type, subject_name), exist_ok=True)

    # Load the data (dictionary)
    path = os.path.join(image_src_dir, filename)
    data = pickle.load(open(path, 'rb'))

    image_cnt = 0
    face_box_data = {}
    for image_id, image_info in data.items():
        url = image_info["url"]
        img = url2image(url)  # Download the image from the given URL
        if img is None:
            continue

        face_info_list = image_info["faces"]
        if len(face_info_list) != 1:
            continue
        face_info = face_info_list[0]

        # Crop the image using facial boundaries
        try:
            cv2_image = np.array(img)
            face_box = face_info["box"]
            assert len(face_box) == 4
            x, y, w, h = face_box
            # Avoid getting out of the image
            face_img = cv2_image[y: np.min([y + h, cv2_image.shape[0] - 1]), x: np.min([x + w, cv2_image.shape[1] - 1])]
            image = Image.fromarray(face_img)

            # Save as JPEG files
            with open(os.path.join(main_path, image_src_type, subject_name, f"{image_id.zfill(6)}.jpg"), "wb") as f:
                image.save(f, "JPEG", quality=85)

            # Re-compute the positions (eyes, noses, mouths)
            for key, value in face_info.items():
                if key == "box":
                    face_info["box"] = [value[0] - x, value[1] - y, value[2], value[3]]
                elif key == "confidence":
                    face_info[key] = value
                elif key == "keypoints":
                    for subkey, subvalue in value.items():
                        value[subkey] = [subvalue[0] - x, subvalue[1] - y]
                else:
                    face_info[key] = [value[0] - x, value[1] - y]

            # Save the face info
            face_box_data[f'{image_id.zfill(6)}'] = face_info

            image_cnt += 1
            # At most keep 20 images per subject
            if image_cnt == 20:
                break
        except ValueError as e:
            pass

    a = 1
    json.dump(face_box_data, open(os.path.join(main_path, image_src_type, subject_name, "face_box.json"), 'w', encoding='utf8'),
              indent=4, ensure_ascii=False)
