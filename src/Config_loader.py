###############################################################################
###############################################################################
#                     Configuration loader and typechecker                    #
#                          Author: Christopher Reardon                        #
#                               Date: 12/11/2024                              #
#             Description: YAML configuration loader and typechecker          #
#                          Project: Phorest Analysis                          #
#                                                                             #
#                         Script designed for Python 3                        #
#           © Copyright Christopher Reardon, Joshua Male, PhorestDX           #
#                                                                             #
#                        Software release: UNRELEASED                         #
###############################################################################
###############################################################################
import yaml  # type: ignore

from enum import Enum
from pydantic import BaseModel, Field, ValidationError  # type: ignore
from pathlib import Path


class SensorType(Enum):
    UOY = 'UoY'
    IMECI = 'IMEC_I'
    IMECII = 'IMEC_II'


class ImageType(Enum):
    PNG = 'png'
    JPG = 'jpg'


class AnalysisMethod(Enum):
    MAX_INTENSITY = "max_intensity"
    CENTRE = "centre"
    GAUSSIAN = "gaussian"
    FANO = "fano"


class UserConfigModel(BaseModel):
    GMR_ID: str
    SENSOR_SERIAL_NUMBER: int = Field(gt=0)
    SETUP_SERIAL_NUMBER: int = Field(gt=0)
    SENSOR_TYPE: SensorType = Field(default=SensorType.UOY)
    ROOT_PATH: str
    DATA_PATH_FORMAT: str
    IMAGE_TYPE: ImageType
    SAVE_PATH: str
    FIGURE_FILENAME: str
    IMAGES_METADATA_PATH: str
    ROI_METADATA_PATH: str
    ANALYSIS_METHOD: AnalysisMethod = Field(
        default=AnalysisMethod.MAX_INTENSITY
    )
    NUMBER_SUB_ROIS: int = Field(ge=0)
    DEVICE_SERIAL_NUMBER: int = Field(gt=0)
    MEASUREMENT_INTERVAL: int = Field(ge=0)


def load_user_config(config_file_path: Path) -> UserConfigModel:
    with open(config_file_path, 'r') as f:
        config_data = yaml.safe_load(f)

    try:
        return UserConfigModel(**config_data)
    except ValidationError as e:
        for err in e.errors():
            field_name = '.'.join(str(x) for x in err['loc'])
            print(f"Error in field '{field_name}': {err['msg']}")
        exit(1)


if __name__ == '__main__':
    config_file = Path('../config/userCONFIG.yml')
    user_config = load_user_config(config_file)
    print(list(vars(user_config).items()))
