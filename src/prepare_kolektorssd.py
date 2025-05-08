import os
import shutil
import random
import numpy as np
from PIL import Image

random.seed(123)    # for reproducability

root_dir = './data/KolektorSSD'
output_dir = './data/processed_kolektor'
os.makedirs(output_dir, exist_ok=True)

train_dir = os.path.join(output_dir, 'train')
test_dir = os.path.join(output_dir, 'test')
for d in [train_dir, test_dir]:
    os.makedirs(os.path.join(d, 'images'), exist_ok=True)
    os.makedirs(os.path.join(d, 'masks'), exist_ok=True)

# check if image is defective and return True/False
def check_defective(path):
    mask = Image.open(path).convert('L')
    mask_np = np.array(mask)
    return mask_np.max() > 0

# Go through each KOSxx folder
for folder in sorted(os.listdir(root_dir)):
    folder_path = os.path.join(root_dir, folder)
    if not os.path.isdir(folder_path): continue

    for fname in os.listdir(folder_path):
        if fname.endswith('.jpg'):
            base = fname[:-4]
            img_path = os.path.join(folder_path, base + '.jpg')
            path = os.path.join(folder_path, base + '_label.bmp')

            if not os.path.exists(path):
                continue

            # Rename the image to avoid overwriting (using the folder name)
            new_img_name = f'{folder}_{base}.jpg'
            new_mask_name = f'{folder}_{base}_label.bmp'

            
            # Decide if it's train or test
            if check_defective(path):
                target = test_dir
            else:
                # 20% of normal images go to test set
                target = test_dir if random.random() < 0.1 else train_dir

            # Copy with new names to avoid overwriting
            shutil.copy(img_path, os.path.join(target, 'images', new_img_name))
            shutil.copy(path, os.path.join(target, 'masks', new_mask_name))
