import sys

from src.Replicate_image_JSON import replicate_image_JSON
from src.Split_ROI_JSON import split_ROI_JSON


###############################################################################
## EDIT THESE VALUES ##########################################################

ROOT_PATH = "/Volumes/krauss/Lisa/GMR/Array/250206/Experiment_1_ethanol/loc1_1"
PATH_TO_IMAGES_METADATA = (
    "/Volumes/krauss/Lisa/GMR/Array/250206/Experiment_1_ethanol/loc1_1/20250206"
)

IMAGE_METADATA_NAME = "image_metadata"
ROI_METADATA_NAME = "ROI_ChirpArray"

###############################################################################
###############################################################################

print("\nBeginning multiprocessor set-up...")
num_of_files = input("\tHow many processors do you want to work with? ").lower()
try:
    num_of_files = int(num_of_files)
except ValueError:
    "Enter a valid integer!"
    sys.exit(1)

print("\nSplitting ROI metadata into {num_of_files} files...")
split_ROI_JSON(ROOT_PATH, ROOT_PATH, ROI_METADATA_NAME, num_of_files)

print("\nReplicating image metadata {num_of_files} times...")
replicate_image_JSON(
    PATH_TO_IMAGES_METADATA,
    PATH_TO_IMAGES_METADATA,
    IMAGE_METADATA_NAME,
    num_of_files,
)
