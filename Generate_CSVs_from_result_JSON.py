import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

root = Path('/Volumes/krauss/Lisa/GMR/Array/250424/650/loc1_1')
parent_filepath = Path(root, 'Pos0')
filenames = [
    Path('image_metadata_1.json'),
    Path('image_metadata_2.json'),
    Path('image_metadata_3.json'),
    Path('image_metadata_4.json'),
    Path('image_metadata_5.json'),
    Path('image_metadata_6.json'),
    Path('image_metadata_7.json'),
    Path('image_metadata_8.json'),
    Path('image_metadata_9.json'),
    Path('image_metadata_10.json'),
    Path('image_metadata_11.json'),
    Path('image_metadata_12.json'),
    Path('image_metadata_13.json'),
    Path('image_metadata_14.json'),
]

out_filepath = Path(root, 'Results')


def rotate_JSON_by_ROI(json_file_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Rotates a JSON file containing image data, reorganizing it by ROI instead of image.

    Args:
        json_file_path <Path>: Path to the input JSON file containing image data.

    Returns:
        A dictionary where:
            - Keys: ROI names <str>
            - Values: Lists of dictionaries, each containing image metadata and ROI-specific data
            If the file cannot be read or parsed, returns None implicitly after printing an error.

    Raises:
        FileNotFoundError: Handled internally; prints an error message if the file is not found.
        json.JSONDecodeError: Handled internally; prints an error message if the JSON is invalid.
    """
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f'Error: File not found at {json_file_path}')
        sys.exit(1)
    except json.JSONDecodeError:
        print(f'Error: Invalid JSON format in {json_file_path}')
        sys.exit(1)

    rotated_data = {}
    for _, image_data in data.items():
        if 'Results' not in image_data:
            continue  # skip images without results.
        results = image_data['Results']
        for roi_name, roi_data in results.items():
            if roi_name not in rotated_data:
                rotated_data[roi_name] = []

            # Create a new dictionary that includes the ROI data along
            # with other image-level information.
            image_info = {
                k: v for k, v in image_data.items() if k != 'Results'
            }  # copy all image data except for the results.
            for k, v in roi_data.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        if 'Values' in kk:
                            continue  # skip raw 'Values'
                        image_info[f'{k}_{kk}'] = vv
                else:
                    image_info[k] = v
            rotated_data[roi_name].append(image_info)

    return rotated_data


def create_CSV_per_ROI(dict_list: Dict[str, List[Dict[str, Any]]], out_filepath: Path) -> None:
    """
    Create CSV files for each ROI from a dictionary of ROI data, organized into subdirectories.

    The function processes a dictionary where keys are ROI names and values are lists of
    dictionaries containing ROI data. It saves each ROI's data as a CSV file in one of two
    subdirectories ('1_gratings' or '2_gratings') based on the presence of 'A' in the ROI name.
    The CSV filename is derived from the ROI name, stripping the first 4 characters.

    Args:
        dict_list: Dictionary where keys are ROI names and values are lists of
        dictionaries containing ROI data to be saved as CSV rows.
        out_filepath: Base output directory path where subdirectories and CSV
        files will be created.

    Returns:
        None: The function writes files to disk and prints status messages
        but does not return a value.

    Raises:
        Exception: Caught internally; prints an error message if any step
        (directory creation, DataFrame conversion, or CSV writing) fails for an
        ROI.
    """
    for roi_name, roi_data_list in dict_list.items():
        try:
            if 'A' in roi_name.split('_')[1]:
                filepath = Path(out_filepath, '1_gratings')
                filepath.mkdir(parents=True, exist_ok=True)
            else:
                filepath = Path(out_filepath, '2_gratings')
                filepath.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(roi_data_list)
            csv_filename = f'{roi_name[4:]}.csv'
            df.to_csv(Path(filepath, csv_filename), index=False)
            print(f'ROI "{roi_name}" saved to {csv_filename}')
        except Exception as e:
            print(f'[ERROR] Issue processing ROI "{roi_name}": {e}')


for fn in filenames:
    input_filepath = Path(parent_filepath, fn)
    print(input_filepath)
    rotated_dict = rotate_JSON_by_ROI(input_filepath)
    create_CSV_per_ROI(rotated_dict, out_filepath)
