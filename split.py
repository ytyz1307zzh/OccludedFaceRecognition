import argparse
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument('-data_dir', default="./ROF_data")
parser.add_argument('-output_dir', default="./split")
args = parser.parse_args()

neutral_dir = os.path.join(args.data_dir, "neutral")
sunglass_dir = os.path.join(args.data_dir, "sunglasses")
mask_dir = os.path.join(args.data_dir, "masked")

subject_list = os.listdir(neutral_dir)
subject2numtrain = {}  # Record the number of training examples for each subject
train_list = []
valid_list = []
train_aug_list = []
test_mask_list = []
test_sunglass_list = []

for subject_name in subject_list:
    subject_dir = os.path.join(neutral_dir, subject_name)
    image_list = os.listdir(subject_dir)

    # Get the original images (not occluded)
    original_images = []
    for image_name in image_list:
        # Skip the face box datafile
        if image_name.endswith(".json"):
            continue
        assert image_name.endswith(".jpg")
        if not image_name.endswith("-lower.jpg") and not image_name.endswith("-upper.jpg"):
            image_path = os.path.join(subject_dir, image_name)
            original_images.append(image_path)

    num_train = round(len(original_images) * 0.8)
    subject2numtrain[subject_name] = num_train
    train_list.extend(original_images[:num_train])
    valid_list.extend(original_images[num_train:])

# Augmented train list
for image_name in train_list:
    image_prefix = image_name[:-4]  # Get rid of ".jpg"
    upper_occluded_image = image_prefix + '-upper.jpg'
    lower_occluded_image = image_prefix + '-lower.jpg'
    train_aug_list.extend([image_name, upper_occluded_image, lower_occluded_image])

# Masked data as test set
subject_list = os.listdir(mask_dir)
for subject_name in subject_list:
    subject_dir = os.path.join(mask_dir, subject_name)
    # In this project, we don't consider unseen subjects
    if subject_name not in subject2numtrain.keys():
        continue
    num_train = subject2numtrain[subject_name]

    mask_images = [os.path.join(subject_dir, filename) for filename in os.listdir(subject_dir)
                   if not filename.endswith(".pkl") and not filename.endswith(".json")]
    # Use 1/4 size of the training data of the same subject
    num_test = min(round(num_train * 0.25), len(mask_images))

    test_mask_list.extend(mask_images[:num_test])

# Sunglasses data as test set
subject_list = os.listdir(sunglass_dir)
for subject_name in subject_list:
    subject_dir = os.path.join(sunglass_dir, subject_name)
    # In this project, we don't consider unseen subjects
    if subject_name not in subject2numtrain.keys():
        continue
    num_train = subject2numtrain[subject_name]

    sunglass_images = [os.path.join(subject_dir, filename) for filename in os.listdir(subject_dir)
                       if not filename.endswith(".pkl") and not filename.endswith(".json")]
    # Use 1/4 size of the training data of the same subject
    num_test = min(round(num_train * 0.25), len(sunglass_images))

    test_sunglass_list.extend(sunglass_images[:num_test])


def save_file(file_name, x):
    with open(file_name, 'w') as f:
        for item in x:
            # If run on Windows, we need to transform the backslashes
            item = re.sub(r'\\', '/', item)
            f.write("%s\n" % item)


if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
save_file(os.path.join(args.output_dir, "train.txt"), train_list)
save_file(os.path.join(args.output_dir, "train_aug.txt"), train_aug_list)
save_file(os.path.join(args.output_dir, "validate.txt"), valid_list)
save_file(os.path.join(args.output_dir, "test_mask.txt"), test_mask_list)
save_file(os.path.join(args.output_dir, "test_sunglass.txt"), test_sunglass_list)
print(f'Training data: {len(train_list)}')
print(f'Augmented training data: {len(train_aug_list)}')
print(f'Validation data: {len(valid_list)}')
print(f'Masked test data: {len(test_mask_list)}')
print(f'Sunglasses test data: {len(test_sunglass_list)}')
