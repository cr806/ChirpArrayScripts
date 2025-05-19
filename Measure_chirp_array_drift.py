from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
from Locator_functions_OLD import find_all_labels

root = Path('/Volumes/krauss/Lisa/GMR/Array/250225/loc1_1')
image_dir = Path('Split/part1')

starting_img_num = 0
total_num_imgs = 443

user_feature_list_path = Path('FeatureLocation.json')
image_feature_path = Path('ImageFeatures.csv')
chip_map_path = Path('Label_templates/Chip_map.json')
template_path = Path('Label_templates/IMECII/IMECII_2')
user_scale_factor = (
    0.75  # Scale factor for template to image (i.e. template larger than image feature)
)

image_feature_list = []
for idx in range(0, total_num_imgs + 1):
    file_path = Path(f'img_{idx:09d}_Default_000.tif')
    img_path = Path(root, image_dir, file_path).as_posix()
    im = cv.imread(img_path, cv.IMREAD_UNCHANGED)
    im = cv.normalize(
        im, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U # type: ignore[arg-type]
    )  # Normalize to 8-bit range (0-255)

    im_shape = im.shape
    image_data = {'File Path': file_path, 'Time Stamp': idx}
    scale_factor, image_angle, features = find_all_labels(
        im,
        image_data,
        user_feature_list_path,
        image_feature_path,
        chip_map_path,
        template_path,
        user_scale_factor,
    )

    image_feature_list.append(features)
    print(f'Image {idx + 1} of {total_num_imgs} processed', end='\r')

image_feature_details = pd.concat(image_feature_list, ignore_index=True)
image_feature_details.to_csv(image_feature_path, index=False)

# image_feature_details = pd.read_csv(Path(image_feature_path))
image_feature_details = image_feature_details.set_index('Timestamp')
labels = image_feature_details['Label'].unique()
fig, ax = plt.subplots(len(labels), 2, layout='tight', figsize=(10, 10))
for idx, label in enumerate(labels):
    df = image_feature_details[image_feature_details['Label'] == label]
    ax[idx][0].plot(df['x'], df['y'], marker='o', linewidth=1)
    ax[idx][0].set_title(df['Label'].iloc[0])
    ax[idx][0].set_xlabel('x movement')
    ax[idx][0].set_ylabel('y movement')
    ax[idx][1].plot(df.index, df['x'], color='blue', label='x movement')
    ax[idx][1].set_title(df['Label'].iloc[0])
    ax[idx][1].set_xlabel('Time')
    ax[idx][1].set_ylabel('x movement')
    ax_ = ax[idx][1].twinx()
    ax_.plot(df.index, df['y'], color='orange', label='y movement')
    ax_.set_ylabel('y movement')
    ax[idx][1].legend()
    ax_.legend()
plt.show()
# plt.savefig(Path(root, 'Results', 'Chip-Drift.png'), dpi=300, bbox_inches='tight')
