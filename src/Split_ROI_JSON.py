import json
from pathlib import Path


def split_ROI_JSON(input_path, output_path, roi_metadata_name, num_of_files):
    roi_metadat_name = Path(f'{roi_metadata_name}.json')
    Path(output_path).mkdir(parents=True, exist_ok=True)

    with Path(input_path, roi_metadat_name).open('r') as f:
        data = json.load(f)

    # Extract 'image_angle' if it exists, otherwise set a default or skip
    image_angle = data.get('image_angle', None)
    if image_angle is None:
        print('Warning: "image_angle" not found in the input JSON, setting to 0')
        image_angle = 0

    items = [(k, v) for k, v in data.items() if k != 'image_angle']
    total = len(items)
    print(f'Number of dictionaries (excl. image_angle): {total}')

    base_size = total // num_of_files
    remainder = total % num_of_files
    split_sizes = [base_size + (1 if i < remainder else 0) for i in range(num_of_files)]

    # Split the items into three chunks
    chunks = []
    start = 0
    for size in split_sizes:
        end = start + size
        chunk = dict(items[start:end])
        chunk['image_angle'] = image_angle
        chunks.append(chunk)
        start = end

    # Save each chunk to a separate JSON file
    for i, chunk in enumerate(chunks, 1):
        output_file = Path(output_path, f'{roi_metadata_name}_{i}.json')
        with output_file.open('w') as f:
            json.dump(chunk, f, indent=4)
        print(f'Saved {len(chunk)} dictionaries to {output_file}')


if __name__ == '__main__':
    num_of_files = 14

    input_path = '/Volumes/krauss/Lisa/GMR/Array/250318/sensor_3_Ecoli/loc1_sensor2_1'
    output_path = input_path
    roi_metadata_name = 'ROI_ChirpArray'

    split_ROI_JSON(input_path, output_path, roi_metadata_name, num_of_files)
