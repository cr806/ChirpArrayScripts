import itertools
import json
import math
from pathlib import Path

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

## Load user feature locations from JSON file and populate 'user_chip_mapping' dictionary


def load_json(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"[ERROR] File not found at '{file_path}'")
        return None
    except json.JSONDecodeError:
        print(f"[ERROR] Could not decode JSON from '{file_path}'. Check the file format.")
        return None
    except Exception as e:
        print(f'[ERROR] An unexpected error occurred: {e}')
        return None


def load_user_feature_locations(file_path):
    user_raw_data = load_json(file_path)
    user_chip_mapping = {}
    if user_raw_data:
        user_chip_mapping['chip_type'] = user_raw_data.get('chip_type', None)
        features = []
        for feature in user_raw_data.get('features', []):
            label_name = feature.get('label')
            feature_location = feature.get('feature_location')
            if label_name:
                features.append({'label': label_name, 'user_location': feature_location})
        user_chip_mapping['features'] = features
    else:
        print("[ERROR] No valid user 'FeatureLocation.json' to load.")
    return user_chip_mapping


## Load chip map locations from JSON and update 'user_chip_map' dictionary


def get_type_of_chip(chip_type, all_chip_mappings):
    for chip_mapping in all_chip_mappings:
        chip_name_label = chip_mapping.get('chip_type', None)
        if chip_name_label == chip_type:
            return chip_mapping
    return None


def get_location_from_label(label, chip_mapping):
    for chip_label in chip_mapping.get('labels'):
        if chip_label.get('label') == label:
            return chip_label.get('label_origin', None)
    return None


def get_user_label_locations_from_chip_map(chip_mapping, user_chip_mapping):
    user_features = user_chip_mapping.get('features', None)
    if user_features:
        for idx, user_feature in enumerate(user_features):
            label = user_feature.get('label')
            feature_location = get_location_from_label(label, chip_mapping)
            user_chip_mapping['features'][idx]['chip_location'] = feature_location
        return user_chip_mapping
    return None


def load_chip_feature_locations(file_path, user_chip_mapping):
    chip_raw_data = load_json(file_path)
    chip_type = user_chip_mapping.get('chip_type')
    chip_mapping = get_type_of_chip(chip_type, chip_raw_data)
    if chip_mapping:
        return get_user_label_locations_from_chip_map(chip_mapping, user_chip_mapping)
    return None


## Calculate the rotation angle


def angle_between_points(v1, v2):
    """Calculates the signed angle in degrees between the line connecting two vectors
    and the positive x-axis.

    Args:
        v1: (x1, y1)
        v2: (x2, y2)

    Returns:
        Angle in degrees, positive for counter-clockwise rotation from the
        positive x-axis to the line segment from point1 to point2.
    """
    x1, y1 = v1
    x2, y2 = v2

    dx = x2 - x1
    dy = y2 - y1

    return math.degrees(math.atan2(dy, dx))  # Angle relative to positive x-axis


def chip_rotation_angle(user_chip_mapping, key='user_location'):
    locations = [a[key] for a in user_chip_mapping['features']]
    chip_locations = [a['chip_location'] for a in user_chip_mapping['features']]

    combination_idxs = list(itertools.combinations(range(len(locations)), 2))

    location_combinations = [(locations[a], locations[b]) for (a, b) in combination_idxs]
    chip_location_combinations = [
        (chip_locations[a], chip_locations[b]) for (a, b) in combination_idxs
    ]

    angles = [angle_between_points(a, b) for (a, b) in location_combinations]
    chip_angles = [angle_between_points(a, b) for (a, b) in chip_location_combinations]

    rotation_angles = [a - b for a, b in zip(chip_angles, angles)]

    rotation_angle = np.quantile(rotation_angles, 0.5)

    user_chip_mapping[f'{key}_all_rotation_angles'] = angles
    user_chip_mapping['chip_location_all_rotation_angles'] = chip_angles

    if 'refined' in key:
        rotation_angle = user_chip_mapping['rotation_angle'] + rotation_angle

    user_chip_mapping[f'{key}_rotation_angle'] = rotation_angle
    user_chip_mapping['rotation_angle'] = rotation_angle

    return user_chip_mapping, rotation_angle


## Calculate the scale factor


def calculate_distance(v1, v2):
    """Calculates the Euclidean distance between two points in 2D space.

    Args:
      v1: A tuple or list representing the first point (x1, y1).
      v1: A tuple or list representing the second point (x2, y2).

    Returns:
      The Euclidean distance between the two points.
    """
    x1, y1 = v1
    x2, y2 = v2

    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def user_chip_scale_factor(user_chip_mapping, key='user_location'):
    locations = [a[key] for a in user_chip_mapping['features']]
    chip_locations = [a['chip_location'] for a in user_chip_mapping['features']]

    combination_idxs = list(itertools.combinations(range(len(locations)), 2))

    location_combinations = [(locations[a], locations[b]) for (a, b) in combination_idxs]
    chip_location_combinations = [
        (chip_locations[a], chip_locations[b]) for (a, b) in combination_idxs
    ]

    distances = [calculate_distance(a, b) for (a, b) in location_combinations]
    chip_distances = [calculate_distance(a, b) for (a, b) in chip_location_combinations]

    scale_factors = [a / b for a, b in zip(distances, chip_distances)]

    user_chip_mapping[f'{key}_all_scale_factors'] = scale_factors

    scale_factor = np.quantile(scale_factors, 0.5)

    user_chip_mapping[f'{key}_scale_factor'] = scale_factor
    user_chip_mapping['scale_factor'] = scale_factor

    # print(f'Distance between user features: {user_distance:.2f} pixels')
    # print(f'Distance between map features: {chip_distance:.2f} microns')
    # print(f'Scale factor: {scale_factor:.2f} pixels/micron')
    return user_chip_mapping, scale_factor


## Rotate the image and user feature locations using calculated angle
def rotate_image(image, rotation_angle):
    # Calculate the center of rotation
    h, w = image.shape
    center = (w / 2, h / 2)

    # Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

    # Apply the rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))

    return rotated_image


def rotate_user_feature_locations(user_location, image_center, rotation_angle):
    x, y = user_location
    cx, cy = image_center
    angle_rad = math.radians(rotation_angle)

    rotated_x = cx + (x - cx) * math.cos(angle_rad) - (y - cy) * math.sin(angle_rad)
    rotated_y = cy + (x - cx) * math.sin(angle_rad) + (y - cy) * math.cos(angle_rad)
    return [int(rotated_x), int(rotated_y)]


## Load template image and scale according to the scale factor
def load_template(template_path):
    try:
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise ValueError(f'Could not load image at {template_path}')
        return template
    except Exception as e:
        print(f'[ERROR] Error loading template: {e}')
        return None


def scale_template(template, scale_factor):
    new_size = (int(template.shape[1] * scale_factor), int(template.shape[0] * scale_factor))
    scaled_template = cv2.resize(template, new_size, interpolation=cv2.INTER_LINEAR)
    return scaled_template


## Visualise results with matplotlib
def visualize_search_window_preprocessing(
    original_search_window, sharpened_search_window, binarized_search_window
):
    _, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(
        cv2.cvtColor(original_search_window, cv2.COLOR_BGR2RGB)
    )
    axes[0].set_title('Original Search Window')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(sharpened_search_window, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Sharpened Search Window')
    axes[1].axis('off')

    axes[2].imshow(
        cv2.cvtColor(binarized_search_window, cv2.COLOR_BGR2RGB)
    )
    axes[2].set_title('Binarized Search Window')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_template_matching_result(
    search_window, result, max_loc, max_val, mean_val, quality_metric
):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.imshow(cv2.cvtColor(search_window, cv2.COLOR_BGR2RGB))
    ax1.set_title('Search Window')
    circle1 = patches.Circle(max_loc, 5, color='red', fill=False)
    ax1.add_patch(circle1)

    ax2.imshow(result, cmap='gray')
    ax2.set_title(
        f'Template Matching Result\n{max_val =:.2f}\n{mean_val =:.2f}\n{quality_metric =:.2f}'
    )
    circle2 = patches.Circle(max_loc, 5, color='red', fill=False)
    ax2.add_patch(circle2)

    plt.tight_layout()
    plt.show()


def visualize_features_with_matplotlib(rotated_image, chip_mapping, feature_shape, key='features'):
    rotated_image_rgb = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB)

    _, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(rotated_image_rgb)

    if 'features' in key:
        features = chip_mapping.get(key, None)
    if 'gratings' in key:
        features = chip_mapping

    if features:
        for f in features:
            label = f.get('label')
            if 'features' in key:
                location = f.get('refined_location')
            if 'gratings' in key:
                location = f.get('grating_origin')

            if location:
                x, y = location
                if feature_shape:
                    height, width = feature_shape
                else:
                    width = f.get('x-size')
                    height = f.get('y-size')

                rect = patches.Rectangle(
                    (x, y), width, height, linewidth=1, edgecolor='white', facecolor='none'
                )
                ax.add_patch(rect)
                ax.annotate(label, location, color='white', fontsize=8, ha='center', va='bottom')

    plt.title('Rotated Image with features highlighted')
    plt.show()


## Find the location of the template in the image using cv2.matchTemplate
def get_template_image_from_label(chip_type, label):
    # TODO! Factor this filepath out
    template_path = Path(f'../Label_templates/IMECII/{chip_type}/{label}.png')
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        print(f'[ERROR] Could not load template image {template_path}.')
        return None
    return template


def refine_feature_locations(image, user_chip_mapping):
    """Refines the location of a feature using template matching,
    accounting for image rotation.

    Args:
        image: The image to search in (already rotated).
        initial_location: The initial (approximate) location of the feature (x, y) in the *original* image coordinates.
        template: The template image of the feature.
        rotation_angle: The angle by which the image was rotated (in degrees, clockwise).
        image_center: The center of the rotation of the image.
        scale_factor: Scaling factor for matching template size to image size

    Returns:
        A tuple containing:
            - The refined location (x, y) in the rotated image coordinates.
            - The rotated initial location
            - The original initial location
        Returns (None, None, None) if template matching fails.
    """

    chip_type = user_chip_mapping.get('chip_type')
    rotation_angle = user_chip_mapping.get('rotation_angle', 0)
    scale_factor = user_chip_mapping.get('scale_factor', 1.0)
    image_center = (image.shape[1] / 2, image.shape[0] / 2)
    user_features = user_chip_mapping.get('features', None)

    if not user_features:
        return None

    for idx, f in enumerate(user_features):
        label = f.get('label')
        user_location = f.get('user_location')

        # Rotate the user location to match the rotated image
        rotated_user_location = rotate_user_feature_locations(
            user_location, image_center, rotation_angle
        )

        # Load and scale template to match image size
        template = get_template_image_from_label(chip_type, label)
        template = scale_template(template, scale_factor)

        # Define a search window around the rotated initial location
        search_window_size = (template.shape[1] * 1.5, template.shape[0] * 1.5)
        x_start = max(0, int(rotated_user_location[0] - search_window_size[0] / 2))
        y_start = max(0, int(rotated_user_location[1] - search_window_size[1] / 2))
        x_end = min(image.shape[1], int(rotated_user_location[0] + search_window_size[0] / 2))
        y_end = min(image.shape[0], int(rotated_user_location[1] + search_window_size[1] / 2))

        search_window = image[y_start:y_end, x_start:x_end]

        ## Pre-process the search window to match the template type
        # Sharpen the search window
        blurred = cv2.GaussianBlur(search_window, (25, 25), 0)
        sharpened_search_window = cv2.addWeighted(search_window, 1.5, blurred, -0.5, 0)

        # Binarise the search window to match the template
        _, binarized_search_window = cv2.threshold(
            sharpened_search_window, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )

        visualize_search_window_preprocessing(
            search_window, sharpened_search_window, binarized_search_window
        )

        # Perform template matching
        result = cv2.matchTemplate(binarized_search_window, template, cv2.TM_CCORR_NORMED)

        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        # Calculate quality metric (e.g., peak-to-mean ratio)
        mean_val = np.mean(result)
        quality_metric = max_val / mean_val if mean_val > 0 else 0  # Avoid division by zero
        visualize_template_matching_result(
            search_window, result, max_loc, max_val, mean_val, quality_metric
        )

        if quality_metric > 1.5 and max_val > 0.5:
            refined_x = x_start + max_loc[0]
            refined_y = y_start + max_loc[1]
            user_chip_mapping['features'][idx]['refined_location'] = [refined_x, refined_y]
            user_chip_mapping['features'][idx]['match_quality'] = quality_metric
            is_good_match = True
            user_chip_mapping['features'][idx]['label_locating_success'] = is_good_match
        else:
            refined_x = None
            refined_y = None
            user_chip_mapping['features'][idx]['refined_location'] = None
            user_chip_mapping['features'][idx]['match_quality'] = quality_metric
            is_good_match = False
            user_chip_mapping['features'][idx]['label_locating_success'] = is_good_match
    return user_chip_mapping, [refined_x, refined_y], template.shape


## Calculate image offset from chip map
def calculate_chip_offset(user_chip_mapping):
    user_features = user_chip_mapping.get('features', None)
    scale_factor = user_chip_mapping.get('scale_factor', 1.0)

    if not user_features:
        return None

    x_offsets = []
    y_offsets = []
    for idx, f in enumerate(user_features):
        offset = [
            a - (b * scale_factor) for a, b in zip(f['refined_location'], f['chip_location'])
        ]
        x_offsets.append(offset[0])
        y_offsets.append(offset[1])
        user_chip_mapping['features'][idx]['feature_offset'] = offset

    user_chip_mapping['offset'] = [np.quantile(x_offsets, 0.5), np.quantile(y_offsets, 0.5)]
    return user_chip_mapping


## Load and offset grating locations from chip map
def offset_and_scale_grating_data(grating_mapping, user_chip_mapping):
    offset = user_chip_mapping.get('offset', [None, None])
    scale_factor = user_chip_mapping['scale_factor']
    if offset:
        for idx, grating in enumerate(grating_mapping):
            offset_grating_origin = [
                int((a * scale_factor) + b) for a, b in zip(grating['grating_origin'], offset)
            ]
            grating_mapping[idx]['grating_origin'] = offset_grating_origin
            grating_mapping[idx]['x-size'] = int(grating['x-size'] * scale_factor)
            grating_mapping[idx]['y-size'] = int(grating['y-size'] * scale_factor)
        return grating_mapping
    return None


def load_and_offset_grating_data(file_path, user_chip_mapping):
    chip_raw_data = load_json(file_path)
    chip_type = user_chip_mapping.get('chip_type')
    chip_mapping = get_type_of_chip(chip_type, chip_raw_data)
    if chip_mapping:
        raw_grating_mapping = chip_mapping.get('gratings', None)
    if raw_grating_mapping:
        return offset_and_scale_grating_data(raw_grating_mapping, user_chip_mapping)
    return None


## Create ROI JSON file
def create_ROI_JSON(chip_type, grating_data, target_shape, rotation_angle, ROI_path):
    ROIs = {}
    if 'IMECII_2' in chip_type:
        suffix = ['N', 'S']
        for g in grating_data:
            x, y = g['grating_origin']
            if (
                x < 0
                or y < 0
                or x + g['x-size'] > target_shape[1]
                or y + g['y-size'] > target_shape[0]
            ):
                continue
            ROIs[f'ROI_{g["label"]}_{suffix[0]}'] = {
                'label': f'{g["label"]}_{suffix[0]}',
                'flip': True,
                'coords': [y + g['y-size'] // 2, x],
                'size': [g['y-size'] // 2, g['x-size']],
            }
            ROIs[f'ROI_{g["label"]}_{suffix[1]}'] = {
                'label': f'{g["label"]}_{suffix[1]}',
                'flip': False,
                'coords': [y, x],
                'size': [g['y-size'] // 2, g['x-size']],
            }
    else:
        suffix = ['A', 'B']
        for g in grating_data:
            x, y = g['grating_origin']
            if (
                x < 0
                or y < 0
                or x + g['x-size'] > target_shape[1]
                or y + g['y-size'] > target_shape[0]
            ):
                continue
            ROIs[f'ROI_{g["label"]}_{suffix[0]}'] = {
                'label': f'{g["label"]}_{suffix[0]}',
                'flip': True,
                'coords': [y, x],
                'size': [g['y-size'], g['x-size'] // 2],
            }
            ROIs[f'ROI_{g["label"]}_{suffix[1]}'] = {
                'label': f'{g["label"]}_{suffix[1]}',
                'flip': False,
                'coords': [y, x + g['x-szie'] // 2],
                'size': [g['y-size'], g['x-size'] // 2],
            }
    ROIs['image_angle'] = rotation_angle
    with open(ROI_path, 'w') as file:
        json.dump(ROIs, file, indent=4)
