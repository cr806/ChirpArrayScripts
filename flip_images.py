import numpy as np
from PIL import Image
from pathlib import Path
from multiprocessing import Pool, Manager

def init_pool(counter):
    global shared_counter
    shared_counter = counter

def flip_single_image(args):
    input_path, output_path = args
    try:
        with Image.open(input_path) as img:
            img_array = np.array(img)
            flipped_array = np.fliplr(img_array)
            flipped_img = Image.fromarray(flipped_array)
            flipped_img.save(output_path)

        with shared_counter.get_lock():  # Ensure thread-safe increment
            shared_counter.value += 1
            if shared_counter.value % 20 == 0:
                print(f"Processed {counter.value} images so far...")
        return True
    except Exception as e:
        print(f'Error processing {input_path.name}: {str(e)}')
        return False

def flip_images_parallel(input_path, output_path):
    output_path.mkdir(parents=True, exist_ok=True)

    # User Manager to create a shared counter for tracking progress
    manager = Manager()
    counter = manager.Value('i', 0)  # 'i' for integer, initialized to 0

    # Prepare list of tasks
    tasks = []
    for filepath in input_path.glob('*.tif'):
        output_path = Path(output_path, filename.name)
        tasks.append((filepath, output_path))

    # Process images in parallel with progress tracking
    print(f"Starting to process {len(tasks)} images...")
    with Pool(initalizer=init_pool, initargs=(counter,)) as pool:
        results = pool.map(flip_single_image, tasks)

    processed_count = sum(results)
    print(f'Completed! Processed {processed_count} images.')

if __name__ == '__main__':
    input_path = Path('path/to/your/input/folder')
    output_path = Path('path/to/your/output/folder')
    flip_images_parallel(input_path, output_path)