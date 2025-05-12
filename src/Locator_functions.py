import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
import json
import itertools

from pathlib import Path
from typing import List, Dict, Tuple, Union, Any, Optional


def clamp_slice(s: slice, max_size: int) -> slice:
    """Adjust a slice to fit within a maximum size, ensuring valid bounds."""
    start = max(0, s.start if s.start is not None else 0)
    stop = min(max_size, s.stop if s.stop is not None else max_size)
    if start >= stop:  # Ensure slice is non-empty
        stop = start + 1 if start < max_size else max_size
    return slice(start, stop)


def find_label(
    target: np.ndarray, scale_factor: Tuple[float, float], params: Dict[str, Any]
) -> Tuple[int, int]:
    """
    Locates a template image within a target image using template matching.

    Args:
        target: The input image (grayscale) in which to search for the template.
        scale_factor: A tuple of (y_scale, x_scale) factors to resize the template image.
        params: A dictionary containing configuration parameters:
                - template_path: Path to the template image file.
                - template_area: Dictionary with 'x_slice' and 'y_slice' slice objects
                  defining the region of interest in the target image.

    Returns:
        origin <Tuple>: A tuple containing the (x, y) coordinates of the top-left corner of the
                        matched region.

    Note:
        The function assumes the template image should be loaded in grayscale mode.
        The coordinates returned are adjusted to account for the sliced region.
    """
    template_path = params['template_path']

    template = cv.imread(template_path, 0)
    template = cv.resize(template, (0, 0), fx=scale_factor, fy=scale_factor)

    method = cv.TM_CCOEFF_NORMED

    h, w = target.shape[:2]

    x_slice = clamp_slice(params['template_area']['x_slice'], w)
    y_slice = clamp_slice(params['template_area']['y_slice'], h)

    res = cv.matchTemplate(target[y_slice, x_slice], template, method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    # Determine the bottom-left corner of the matched region
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        origin = min_loc
    else:
        origin = max_loc

    #  Correct for the slice
    origin = (origin[0] + x_slice.start, origin[1] + y_slice.start)
    return origin


def image_angle_from_horizontal(
    vector: Union[Tuple[float, float], List[float], np.ndarray],
) -> float:
    """
    Calculates the angle of a 2D vector from the horizontal (x-axis) in degrees.

    - calculates counter-clockwise angle and the positive x-axis
    - uses arctan2 function to handles all quadrants correctly
    - returns angles in the range [-180, 180].

    Args:
        vector: A 2D vector represented as a tuple, list, or numpy array with
                exactly 2 elements (x, y).

    Returns:
        angle_degrees <float>: The angle in degrees from the positive x-axis.
                               - Positive values indicate counterclockwise angles.
                               - Negative values indicate clockwise angles.

    Raises:
        ValueError: If the input vector doesn't have exactly 2 dimensions.

    Examples:
        >>> image_angle_from_horizontal([1, 0])  # Pointing right
        0.0
        >>> image_angle_from_horizontal([0, 1])  # Pointing up
        90.0
        >>> image_angle_from_horizontal([-1, 0])  # Pointing left
        180.0
        >>> image_angle_from_horizontal([0, -1])  # Pointing down
        -90.0
    """
    vector = np.array(vector)

    if len(vector) != 2:
        raise ValueError('Vector must be 2D.')

    x, y = vector[0], vector[1]
    angle_radians = np.arctan2(y, x)
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees


def angle_between_vectors(
    vector1: Union[List[float], Tuple[float, ...], np.ndarray],
    vector2: Union[List[float], Tuple[float, ...], np.ndarray],
) -> float:
    """
    Calculates the smallest angle between two vectors in degrees.

    - uses the dot product: angle = arccos(dot(v1, v2) / (|v1| * |v2|))
    - result is always the smallest angle between the vectors
    - handles edge cases by:
      1. Returning NaN if either vector has zero magnitude
      2. Clipping the cosine value to [-1, 1]

    Args:
        vector1: First vector as a list, tuple, or numpy array
        vector2: Second vector as a list, tuple, or numpy array

    Returns:
        angle_degrees <float>: The angle between the vectors in degrees in range [0, 180]
                               Returns NaN if either vector has zero magnitude

    Examples:
        >>> angle_between_vectors([1, 0], [0, 1])  # Perpendicular vectors
        90.0
        >>> angle_between_vectors([1, 0], [1, 0])  # Parallel vectors
        0.0
        >>> angle_between_vectors([1, 0], [-1, 0])  # Opposite vectors
        180.0
        >>> angle_between_vectors([0, 0], [1, 0])  # Zero vector
        nan

    Note:
        The vectors need not be 2D - this function works for vectors of any dimension.
        The formula uses the arccosine function which returns angles in [0, π],
        so the returned angle is always the smallest angle between the vectors.
    """
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)

    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    if magnitude1 == 0 or magnitude2 == 0:
        # return not a number if a vector has zero magnitude.
        return float('nan')

    cosine_angle = dot_product / (magnitude1 * magnitude2)
    # clip to avoid domain errors.
    angle_radians = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees


def vector_relative_to_origin(
    origin: Union[List[float], Tuple[float, ...], np.ndarray],
    vector: Union[List[float], Tuple[float, ...], np.ndarray],
) -> np.ndarray:
    """
    Translates a vector by subtracting an origin point, creating a vector from origin to the point.

    - equivalent to changing the coordinate system's origin to the specified point.

    Args:
        origin: The origin point as a vector (coordinate)
        vector: The vector (or point) for which the translation vector is required.

    Returns:
        <np.ndarray>: The translated vector, now expressed relative to the new origin.

    Examples:
        >>> vector_relative_to_origin([1, 1], [3, 4])  # Point (3,4) relative to origin (1,1)
        array([2, 3])
        >>> vector_relative_to_origin([0, 0, 0], [5, 5, 5])  # No translation (origin unchanged)
        array([5, 5, 5])

    Note:
        This function doesn't move a vector in space. It provides the vector
        required to 'translate' the original vector
    """
    vector = np.array(vector)
    origin = np.array(origin)

    return vector - origin


def scale_vector(
    vector: Union[List[float], Tuple[float, ...], np.ndarray],
    scale: Union[float, int, List[float], Tuple[float, ...], np.ndarray],
) -> np.ndarray:
    """
    Scales a vector by a scalar value or element-wise by another vector.

    Args:
        vector: The vector to be scaled, as a list, tuple, or numpy array
        scale: The scaling factor, which can be:
               - A single scalar (int or float)
               - A vector with the same dimensions as 'vector'

    Returns:
        <np.ndarray>: The scaled vector

    Examples:
        >>> scale_vector([1, 2, 3], 2)  # Double the magnitude
        array([2, 4, 6])
        >>> scale_vector([1, 2, 3], -1)  # Reverse the direction
        array([-1, -2, -3])
        >>> scale_vector([1, 2, 3], [2, 3, 4])  # Element-wise scaling
        array([2, 6, 12])

    Note:
        When using element-wise scaling, the scale vector must have the same
        dimensions as the input vector.
    """
    vector = np.array(vector)

    return vector * scale


def process_config_files(
    image_feature_path: str, user_feature_list_path: str, chip_map_path: str
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Process configuration files for chip feature analysis by loading data from three sources.

    This function reads and parses three configuration files:
    1. A CSV file containing image feature details
    2. A JSON file containing user-defined feature mappings on the physical chip
    3. A JSON file containing the chip's GDSII mapping information

    The function then filters the chip map to include only the chip type specified
    in the user feature list, providing properly structured data for further analysis.

    Args:
        image_feature_path: Path to the CSV file containing image feature details
        user_feature_list_path: Path to the JSON file containing user-defined features
                                (represents locations on the 'real' physical chip)
        chip_map_path: Path to the JSON file containing GDSII chip mapping data
                       (represents the theoretical chip design)

    Returns:
        A tuple containing three elements:
        - image_feature_details <pd.DataFrame>: DataFrame of image features or None if file not found
        - feature_dict <Dict>: Dictionary of user-defined features or None if file not found/invalid
        - chip_map <Dict>: Filtered chip map dictionary for the specific chip type or None if
                           file not found/invalid or chip type not found

    Notes:
        - The function logs errors to stdout but does not raise exceptions for missing files
        - Missing or invalid files result in None values for the corresponding return items
        - The chip map is filtered to include only the chip type specified in feature_dict
    """
    image_feature_details = None
    feature_dict = None
    chip_map = None

    # Load image feature details from CSV
    try:
        image_feature_details = pd.read_csv(image_feature_path)
    except FileNotFoundError:
        print(f'[INFO]: File not found at {image_feature_path}')
    except Exception as e:
        print(f'[ERROR]: Failed to read CSV file {image_feature_path}: {str(e)}')

    # Load user feature list from JSON (represents locations on the 'real' chip)
    try:
        with open(user_feature_list_path, 'r') as file:
            feature_dict = json.load(file)
    except FileNotFoundError:
        print(f'[ERROR]: File not found at {user_feature_list_path}')
    except json.JSONDecodeError:
        print(f'[ERROR]: Invalid JSON format in {user_feature_list_path}')
    except Exception as e:
        print(f'[ERROR]: Failed to process {user_feature_list_path}: {str(e)}')

    # Load chip map from JSON (represents the chip GDSII file)
    try:
        with open(chip_map_path, 'r') as file:
            chip_map_data = json.load(file)

        # Filter chip map to only include the chip type we are interested in
        if feature_dict and ('chip_type' in feature_dict):
            chip_type_found = False
            for c in chip_map_data.get('chip', []):
                if c.get('chip_type') == feature_dict['chip_type']:
                    chip_map = c
                    chip_type_found = True
                    break

            if not chip_type_found:
                print(f'[WARNING]: Chip type "{feature_dict["chip_type"]}" not found in chip map')
        else:
            print(f'[WARNING]: Cannot filter chip map - missing chip_type in feature dictionary')
            chip_map = chip_map_data

    except FileNotFoundError:
        print(f'[ERROR]: File not found at {chip_map_path}')
    except json.JSONDecodeError:
        print(f'[ERROR]: Invalid JSON format in {chip_map_path}')
    except Exception as e:
        print(f'[ERROR]: Failed to process {chip_map_path}: {str(e)}')

    return image_feature_details, feature_dict, chip_map


def find_all_labels(
    target: np.ndarray,
    image_data: Dict[str, str],
    feature_list_path: Union[str, Path],
    image_feature_path: Union[str, Path],
    chip_map_path: Union[str, Path],
    template_path: Union[str, Path],
    scale_factor: float,
) -> Tuple[float, float, pd.DataFrame]:
    """
    Find and analyze labels in an image based on configuration files and templates.

    Args:
        target: The target image or data to search for labels in
        image_data: Dictionary containing image metadata with 'File Path' and 'Time Stamp' keys
        feature_list_path: Path to the feature list configuration file
        image_feature_path: Path to the image feature details file
        chip_map_path: Path to the chip map configuration file
        template_path: Path to the directory containing template images
        scale_factor: Initial scale factor for image processing

    Returns:
        Tuple containing:
        - scale_factor <float>: Calculated scale factor between chip map and feature distances
        - image_angle <float>: Angle difference between chip map and feature labels in degrees
        - image_feature_details <pd.DataFrame>: Updated image feature details with label locations

    Raises:
        ValueError: If no features are found in FeatureLocation.json when auto_locate_grating is True
    """
    image_feature_details, feature_dict, chip_map = process_config_files(
        image_feature_path, feature_list_path, chip_map_path
    )

    # Extract from chip_map only the features we are interested in
    if chip_map['auto_locate_grating']:
        chip_features = feature_dict.get('features', None)
        if not chip_features:
            raise ValueError('[ERROR] No features found in "FeatureLocation.json"')

    feature_labels = [f['label'].lower() for f in chip_features]
    chip_labels = [l for l in chip_map['labels'] if l['label'].lower() in feature_labels]

    # Update chip features with template path and search region
    for idx, f in enumerate(chip_features):
        chip_features[idx]['template_path'] = Path(template_path, f'{f["label"]}.png')
        if image_feature_details is None:
            _x = f['feature_location'][0]
            _y = f['feature_location'][1]
        else:
            mask = image_feature_details['Label'] == f['label']
            last_row = image_feature_details[mask].iloc[-1]
            _x = last_row['x']
            _y = last_row['y']
        search_window = 100
        chip_features[idx]['template_area'] = {
            'x_slice': slice(_x - search_window, _x + search_window),
            'y_slice': slice(_y - search_window, _y + search_window),
        }

    if image_feature_details is None:
        image_feature_details = pd.DataFrame(columns=['Image', 'Label', 'x', 'y', 'Timestamp'])
    locs = []
    image_name = image_data['File Path']
    timestamp = image_data['Time Stamp']
    for f in chip_features:
        loc = find_label(target, scale_factor, f)
        new_data = {
            'Image': image_name,
            'Label': f['label'],
            'x': loc[0],
            'y': loc[1],
            'Timestamp': timestamp,
        }
        new_row = pd.DataFrame([new_data])
        image_feature_details = pd.concat([image_feature_details, new_row], ignore_index=True)
        locs.append(loc)
    image_feature_details.to_csv(image_feature_path, index=False)

    # Calculate the distance between all combinations of chip_map and chip_feature labels
    loc_combinations = itertools.combinations(range(len(chip_labels)), 2)
    dist_chip_map = []
    dist_chip_feature = []
    for a, b in loc_combinations:
        dist_chip_map.append(
            vector_relative_to_origin(
                chip_labels[a]['label_origin'], chip_labels[b]['label_origin']
            )
        )
        dist_chip_feature.append(vector_relative_to_origin(locs[a], locs[b]))

    # Cacluate average scaling factor using all combinations of chip map and chip feature distances
    scale_factor = np.round(np.linalg.norm(dist_chip_feature) / np.linalg.norm(dist_chip_map), 4)

    # Calculate the angle between the chip map and chip feature labels separately
    chip_map_angle = image_angle_from_horizontal(
        vector_relative_to_origin(chip_labels[0]['label_origin'], chip_labels[1]['label_origin'])
    )
    chip_feature_angle = image_angle_from_horizontal(vector_relative_to_origin(locs[0], locs[1]))

    # Calculate the angle of the image compared to the chip map
    # image can then be rotated by this angle to align with the chip map
    _angle = np.round(chip_map_angle - chip_feature_angle, 1)
    angle_sign = np.sign(_angle)
    image_angle = np.mod(_angle, angle_sign * 180)
    if np.isnan(image_angle):
        image_angle = 0.0

    return scale_factor, image_angle, image_feature_details
