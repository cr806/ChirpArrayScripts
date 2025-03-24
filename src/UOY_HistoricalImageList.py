############################################################################
############################################################################
#                         UoY Historical Images List                       #
#                        Author: Christopher Reardon                       #
#                             Date 06/03/2025                              #
#        Description: Creates Image List without Camera Acquisition        #
#                         Project: Phorest Analysis                        #
#                                                                          #
#                       Script Designed for Python 3                       #
#         © Copyright Christopher Reardon, Joshua Male, PhorestDX          #
#                                                                          #
#                       Software Release: Unreleased                       #
############################################################################
############################################################################
import json
import src.path_change  # noqa

from pathlib import Path
from src.Config_loader import load_user_config


def UOY_historical_image_list(path_to_images, image_type,
                              sensor_ID, setup_ID,
                              reader_ID, output_path):

    img_files = [f for f in Path(path_to_images).glob(f'*.{image_type.lower()}') if f.is_file()]
    img_files = sorted(img_files, key=lambda x: int(x.stem.split('_')[1]))

    image_metadata = {}
    for idx, tf in enumerate(img_files):
        image_metadata[tf.stem] = {
            "Sensor ID": sensor_ID,
            "Setup ID": setup_ID,
            "Reader ID": reader_ID,
            "Root Path": tf.parent.as_posix(),
            "File Path": tf.name,
            "Error": "None",
            "Processed": False,
            "Time Stamp": str(idx).zfill(6),
        }
        if not idx % 20:
            print(f'Processed image: {idx}')

        with Path(output_path).open('w') as f:
            json.dump(image_metadata, f, indent=4)


if __name__ == '__main__':
    user_config_filepath = Path('config', 'userCONFIG.yml')
    config_data = load_user_config(user_config_filepath)

    data_path = Path(config_data.ROOT_PATH)
    image_type = str(config_data.IMAGE_TYPE.value).lower()
    sensor_ID = str(config_data.SENSOR_SERIAL_NUMBER).zfill(6)
    setup_ID = str(config_data.SETUP_SERIAL_NUMBER).zfill(6)
    reader_ID = str(config_data.DEVICE_SERIAL_NUMBER).zfill(6)

    output_path = Path(data_path, f'image_metadata_SU{sensor_ID}.json')

    UOY_historical_image_list(data_path, image_type,
                              sensor_ID, setup_ID,
                              reader_ID, output_path)