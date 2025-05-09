from multiprocessing import Manager, Pool
from pathlib import Path

import numpy as np
from PIL import Image


def init_pool(counter):
    global shared_counter
    shared_counter = counter


def flip_single_image(args):
    input_full_path, output_full_path, flip_direction = args
    try:
        with Image.open(input_full_path) as img:
            img_array = np.array(img)
            if flip_direction == 'UD':
                flipped_array = np.flipud(img_array)
            elif flip_direction == 'LR':
                flipped_array = np.fliplr(img_array)
            flipped_img = Image.fromarray(flipped_array)
            flipped_img.save(output_full_path)
            print(f'Image: {input_full_path.name} processed.')
        return True
    except Exception as e:
        print(f'Error processing {input_full_path.name}: {str(e)}')
        return False


def flip_images_parallel(input_base_path, output_base_path, flip_direction, extension):
    output_base_path.mkdir(parents=True, exist_ok=True)

    # User Manager to create a shared counter for tracking progress
    manager = Manager()
    counter = manager.Value('i', 0)  # 'i' for integer, initialized to 0

    # Prepare list of tasks
    tasks = []
    for input_full_path in input_base_path.glob('*.{extension}'):
        output_full_path = Path(output_base_path, input_full_path.name)
        tasks.append((input_full_path, output_full_path, flip_direction))

    # Process images in parallel with progress tracking
    print(f'Starting to process {len(tasks)} images...')
    with Pool(initializer=init_pool, initargs=(counter,)) as pool:
        results = pool.map(flip_single_image, tasks)

    processed_count = sum(results)
    print(f'Completed! Processed {processed_count} images.')


if __name__ == '__main__':
    input_base_path = Path('/home/chris/mnt/storage/Callum/03_Data/17_array_testing/ethanol_full_array_test_240425')
    output_base_path = Path('/home/chris/mnt/storage/Callum/03_Data/17_array_testing/ethanol_full_array_test_240425/flipped')
    flip_direction = 'UD'
    # flip_direction = 'LR'
    extension = 'png'  # Specify the image file extension
    flip_images_parallel(input_base_path, output_base_path, flip_direction, extension)
