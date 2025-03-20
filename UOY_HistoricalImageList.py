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

# Set up root path
root = Path().absolute()

user_config_filepath = Path('config', 'userCONFIG.yml')
config_data = load_user_config(user_config_filepath)

data_path = Path(config_data.ROOT_PATH, 'Pos0')
image_type = config_data.IMAGE_TYPE.value
sensor_ID = str(config_data.SENSOR_SERIAL_NUMBER).zfill(6)
setup_ID = str(config_data.SETUP_SERIAL_NUMBER).zfill(6)
reader_ID = str(config_data.DEVICE_SERIAL_NUMBER).zfill(6)

tif_files = [f for f in data_path.glob('*.tif') if f.is_file()]
tif_files = sorted(tif_files, key=lambda x: int(x.stem.split('_')[1]))

image_metadata = {}
for idx, tf in enumerate(tif_files):
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

    output_path = Path(
        data_path, f'image_metadata_SU{sensor_ID}.json')
    with output_path.open("w") as f:
        json.dump(image_metadata, f, indent=4)
