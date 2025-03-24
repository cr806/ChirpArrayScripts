import json

from pathlib import Path


def replicate_image_JSON(input_path, output_path, image_metadata_name, num_of_files):
    image_metadata_name = Path(f'{image_metadata_name}.json')
    Path(output_path).mkdir(parents=True, exist_ok=True)

    with Path(input_path, image_metadata_name).open('r') as f:
        data = json.load(f)

    # Save a copy separate numbered JSON files
    for i in range(1, num_of_files + 1):
        output_file = Path(output_path, f'{image_metadata_name}_{i}.json')
        with output_file.open('w') as f:
            json.dump(chunk, f, indent=4)
        print(f'Saved copy {i} to {output_file}')

if __name__ == '__main__':
    num_of_files = 14

    input_path = '/Volumes/krauss/Lisa/GMR/Array/250318/sensor_3_Ecoli/loc1_sensor2_1'
    output_path = input_json_path
    image_metadata_name = 'ROI_ChirpArray'

    replicate_image_JSON(input_path, output_path, image_metadata_name, num_of_files)