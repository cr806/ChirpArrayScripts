import shutil
import sys
import json

from pathlib import Path

from src.UOY_HistoricalImageList import UOY_historical_image_list
from src.Generate_ROI_JSON import generate_ROI_JSON
from src.Save_plots_of_ROI_locations import save_plots_of_ROI_locations

###############################################################################
## EDIT THESE VALUES ##########################################################

ROOT_PATH = "/Volumes/krauss/Lisa/GMR/Array/250206/Experiment_1_ethanol/loc1_1"
PATH_TO_IMAGES = (
    "/Volumes/krauss/Lisa/GMR/Array/250206/Experiment_1_ethanol/loc1_1/20250206"
)
IMAGE_TYPE = "tif"
IMAGE_NAME_FOR_ROI_PLOTS = "img_000000000_Default_000"

IMAGE_METADATA_NAME = "image_metadata"
ROI_METADATA_NAME = "ROI_ChirpArray"

SERVER_ROOT_PATH = (
    "/home/chris/mnt/storage/Lisa/GMR/Array/250206/Experiment_1_ethanol/loc1_1"
)

###############################################################################
###############################################################################


def edit_image_metadata_file(IMAGE_METADATA_NAME, new_root):
    print("\tEditing metadata files...")
    file_path = Path("Generated_files", f"{IMAGE_METADATA_NAME}.json")
    with file_path.open("r") as f:
        data = json.load(f)

    for k, v in data.items():
        data[k]["Root Path"] = new_root.as_posix()

    with file_path.open("w") as f:
        json.dump(data, f, indent=4)


def move_files(files_src_dest_name):
    """
    Copy a file from src_path to dest_path.

    Args:
        files_src_dest_name (list(tuple)): List containging tuples with src_path,
                                           dest_path, and filename

    Returns:
        bool: True if the copy succeeded, False otherwise.
    """

    message = []
    success = True
    for src_path, dest_path, name in files_src_dest_name:
        try:
            shutil.copy(Path(src_path, name), Path(dest_path, name))
            message.append(
                f"Successfully copied '{Path(src_path, name)}' to '{Path(dest_path, name)}'."
            )
        except PermissionError:
            message.append(
                f"Error: Permission denied while copying '{Path(src_path, name)}' to '{Path(dest_path, name)}'."
            )
            success = False
        except shutil.SameFileError:
            message.append(
                f"Error: Source and destination are the same file: '{Path(src_path, name)}'."
            )
            success = False
        except Exception as e:
            message.append(
                f"Error: Failed to copy '{Path(src_path, name)}' to '{Path(dest_path, name)}': {e}"
            )
            success = False
    print("\n")
    print("\n".join(message))
    return success


# 0. Ask user if they want to skip to ROI metadata generation
skip = input(
    "Step 0: Do you want to skip to the ROI metadata generation step? (y/n): "
)
if skip != "y":
    # 1. Ask user to check images and flip if required
    print(
        "\nStep 1: Please check images to locate two features and flip all images if required."
    )
    input("Press Enter when images are ready...")

    # 2. Confirm FeatureLocation.json is up-to-date
    print(
        "\nStep 2: Please check 'FeatureLocation.json' and update if required."
    )
    print("        NB: 'Labels' should be in alphabetic/numeric order")
    input("Press Enter when 'FeatureLocation.json' is ready...")

    # 3. Confirm userCONFIG.yml is up-to-date
    print("\nStep 3: Please check 'userCONFIG.yml' and update if required.")
    print(
        "        NB: Filepaths should reflect the location from where the analysis script will be ran"
    )
    input("Press Enter when 'userCONFIG.yml' is ready...")

    # 4. Confirm root, image, ROI, and server_root variables
    print("\nStep 4: Are these locations correct?")
    print(f"\t{'Root folder:':<30} {ROOT_PATH}")
    print(f"\t{'Image folder:':<30} {PATH_TO_IMAGES}")
    print(f"\t{'Image type:':<30} {IMAGE_TYPE}")
    print(f"\t{'Image metadata name:':<30} {IMAGE_METADATA_NAME}")
    print(f"\t{'ROI metadata name:':<30} {ROI_METADATA_NAME}")
    print(f"\t{'Root folder (on server):':<30} {SERVER_ROOT_PATH}")
    input("Press Enter to continue...")

    # Create directory for generated files
    Path("Generated_files").mkdir(parents=True, exist_ok=True)

    # 5. Generate image metadata
    print("\nStep 5: Preparing to generate image metadata...")
    proceed = input("\tDo you want to continue? (y/n): ").lower()
    if proceed != "y":
        print("Skipping image metadata generation.")
    else:
        UOY_historical_image_list(
            PATH_TO_IMAGES,
            IMAGE_TYPE,
            "XXXXX",
            "XXXXX",
            "XXXXX",
            Path("Generated_files", f"{IMAGE_METADATA_NAME}.json"),
        )

# 6. Generate ROI metadata
# Make sure directory exists for generated files
Path("Generated_files").mkdir(parents=True, exist_ok=True)
user_scale_factor = 0.73
print("\nStep 6: Generating ROI metadata...")
generate_ROI_JSON(
    PATH_TO_IMAGES,
    Path("Generated_files", f"{IMAGE_METADATA_NAME}.json"),
    Path("Generated_files", f"{ROI_METADATA_NAME}.json"),
    user_scale_factor,
)

# 7. Save plots of ROI locations
print("\nStep 7: Saving ROI location plots for review...")
path_to_first_img = Path(
    PATH_TO_IMAGES, f"{IMAGE_NAME_FOR_ROI_PLOTS}.{IMAGE_TYPE}"
)
save_plots_of_ROI_locations(
    ROOT_PATH,
    path_to_first_img,
    Path("Generated_files", f"{ROI_METADATA_NAME}.json"),
    save_path=Path("Generated_files"),
)

# 8. Confirm ROI locations
print("\nStep 8: Please check the saved ROI location plots.")
roi_correct = input("\tAre the ROI locations correct? (y/n): ").lower()
if roi_correct != "y":
    print("\nROI locations not confirmed. Please adjust and rerun. Exiting.")
    sys.exit(1)

# 9. Ask about analysis location
print("\nStep 9: Analysis location selection...")
location = input(
    "\tWill analysis run locally or on a server? ((l)ocal/(s)erver): "
).lower()
if "s" in location:
    image_metadata_server_path = Path(
        SERVER_ROOT_PATH, Path(PATH_TO_IMAGES).relative_to(ROOT_PATH)
    )
    edit_image_metadata_file(IMAGE_METADATA_NAME, image_metadata_server_path)

    move = input("\n\tMove metadata files to server? (y/n): ").lower()
else:
    move = input(
        "\n\tMove metadata files to approprate directory for analysis? (y/n): "
    ).lower()

src_path = Path("Generated_files")
if move == "y":
    files_src_dest_name = [
        (src_path, PATH_TO_IMAGES, f"{IMAGE_METADATA_NAME}.json"),
        (src_path, ROOT_PATH, f"{ROI_METADATA_NAME}.json"),
    ]
    if not move_files(files_src_dest_name):
        print("\nFailed to move some/all metadata files. Exiting.")
        sys.exit(1)

# 10. Back-up generated files to location of experiment
backup = input(
    "\nStep 10: Do you want to back-up all generated files to experiment directory? (y/n): "
).lower()
if backup == "y":
    dest_path = Path(ROOT_PATH, "Image_analysis_files_BACKUP")
    dest_path.mkdir(parents=True, exist_ok=True)
    files_src_dest_name = []
    src_path = Path("Generated_files")
    for entry in src_path.iterdir():
        if not entry.is_file():
            continue
        files_src_dest_name.append((src_path, dest_path, entry.name))

    src_path = Path("config")
    for entry in src_path.iterdir():
        if not entry.is_file():
            continue
        files_src_dest_name.append((src_path, dest_path, entry.name))

    if not move_files(files_src_dest_name):
        print("\nFailed to move some/all metadata files. Exiting.")
        sys.exit(1)
