import json
from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from src.Locator_functions import (
    find_all_labels,
    process_config_files,
    scale_vector,
    vector_relative_to_origin,
)

# Template size: 57 x 80
PIXELS_PER_MICRON = 0.68 # Quarter of IMECII-2
PIXELS_PER_MICRON = 0.70 # Full IMECII-2 (Size: 77 x 117)


def generate_ROI_data_and_save(
    image_feature_path: Union[str, Path],
    user_feature_list_path: Union[str, Path],
    chip_map_path: Union[str, Path],
    ROI_path: Union[str, Path],
    target_shape: Tuple[int, int],
    scale_factor: float,
    image_angle: float,
) -> Dict[str, Union[Dict[str, Union[str, bool, List[int]]], float]]:
    """
    Generate a Region of Interest (ROI) file from image features and chip map data.

    Args:
        image_feature_path: Path to the image feature details file
        user_feature_list_path: Path to the user-defined feature list configuration file
        chip_map_path: Path to the chip map configuration file
        ROI_path: Path where the ROI JSON file will be saved
        target_shape: Tuple of (height, width) representing the target image dimensions
        scale_factor: Scaling factor to apply to chip map coordinates and sizes
        image_angle: Angle of the image relative to the chip map in degrees

    Returns:
        Dict containing ROI definitions where each ROI has:
            - label <str>: ROI identifier
            - flip <bool>: whether the ROI should be flipped
            - coords <List[int, int]>: [y, x] coordinates of ROI origin
            - size <List[int, int]>: [height, width] of ROI
        image_angle <float>: key with the provided float value
    """

    image_feature_details, _, chip_map = process_config_files(
        image_feature_path, user_feature_list_path, chip_map_path
    )

    # Filter chip map to only the gratings and their locations, sizes
    chip_gratings = chip_map.get('gratings', None)

    feature_labels = image_feature_details['Label'].unique()
    feature_labels = [label.lower() for label in feature_labels]
    chip_labels = [label for label in chip_map['labels']
                   if label['label'].lower() in feature_labels]

    locs = []
    for chip_label in chip_labels:
        mask = image_feature_details['Label'] == chip_label['label']
        last_row = image_feature_details[mask].iloc[-1]
        locs.append((last_row['x'], last_row['y']))

    # Use 'vector_relative_to_origin' to calculate translation vector to move
    # the chip labels to the feature labels
    # Note: Chip labels location must be scaled before calculating the translation vector
    coord_offset = vector_relative_to_origin(
        locs[0], scale_vector(chip_labels[0]['label_origin'], scale_factor)
    )

    # Add translated grating location to chip grating dictionary
    for idx, g in enumerate(chip_gratings):
        chip_gratings[idx]['grating_origin'] = vector_relative_to_origin(
            coord_offset, scale_vector(g['grating_origin'], scale_factor)
        )

    # Create ROIs for each grating (North and South) using scaled grating sizes
    ROIs = {}
    for g in chip_gratings:
        x, y = g['grating_origin']
        x, y = int(x), int(y)
        if (
            x < 0
            or y < 0
            or x + int(g['x-size'] * scale_factor) > target_shape[1]
            or y + int(g['y-size'] * scale_factor) > target_shape[0]
        ):
            continue
        grating_x_size = int(g['x-size'] * scale_factor)
        grating_y_size = int(g['y-size'] * scale_factor)
        ROIs[f'ROI_{g["label"]}_N'] = {
            'label': f'{g["label"]}_N',
            'flip': True,
            'coords': [y + grating_y_size // 2, x],
            'size': [grating_y_size // 2, grating_x_size],
        }
        ROIs[f'ROI_{g["label"]}_S'] = {
            'label': f'{g["label"]}_S',
            'flip': False,
            'coords': [y, x],
            'size': [grating_y_size // 2, grating_x_size],
        }
    ROIs['image_angle'] = image_angle
    with open(ROI_path, 'w') as file:
        json.dump(ROIs, file, indent=4)

    return ROIs


def generate_ROI_JSON(path_to_images, img_metadata_json, ROI_path, user_scale_factor):
    user_feature_list_path = Path('config', 'FeatureLocation.json')
    image_feature_path = Path('Generated_files', 'ImageFeatures.csv')
    chip_map_path = Path('Label_templates', 'Chip_map.json')
    template_path = Path('Label_templates', 'IMECII', 'IMEC-II_2')

    img_data = pd.read_json(Path(img_metadata_json))
    img_data = img_data.T.reset_index(drop=True)

    img_path = Path(img_data['Root Path'][0], img_data['File Path'][0]).as_posix()
    im = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    im = cv2.normalize(
        im, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U # type: ignore
    )  # Normalize to 8-bit range (0-255)

    scale_factor, image_angle, _ = find_all_labels(
        im,
        img_data.iloc[0],
        user_feature_list_path,
        image_feature_path,
        chip_map_path,
        template_path,
        user_scale_factor,
    )

    # Rotate image to get 'correct' location of features
    if np.abs(image_angle) > 0:
        h, w = im.shape
        rot_centre = (im.shape[1] // 2, im.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(rot_centre, -image_angle, 1.0)
        im = cv2.warpAffine(im, rotation_matrix, (w, h))

        scale_factor, _, _ = find_all_labels(
            im,
            img_data.iloc[0],
            user_feature_list_path,
            image_feature_path,
            chip_map_path,
            template_path,
            user_scale_factor,
        )

    if scale_factor < (0.95 * user_scale_factor) or scale_factor > (1.05 * user_scale_factor):
        print(f'Unexpected scale-factor: {scale_factor}')
        print('Setting to user specified scale factor and continuing...')
        scale_factor = user_scale_factor

    generate_ROI_data_and_save(
        image_feature_path,
        user_feature_list_path,
        chip_map_path,
        ROI_path,
        im.shape,
        scale_factor,
        image_angle,
    )


if __name__ == '__main__':
    path_to_images = Path(
        '/Volumes/krauss/Lisa/GMR/Array/250318/sensor_3_Ecoli/loc1_sensor2_1/Pos0'
    )
    img_metadata_json = 'image_metadata_SU000001.json'
    ROI_path = Path('ROI_ChirpArray.json')
    user_scale_factor = PIXELS_PER_MICRON  # Template to image (i.e. template larger than image feature)

    generate_ROI_JSON(path_to_images, img_metadata_json, ROI_path, user_scale_factor)
