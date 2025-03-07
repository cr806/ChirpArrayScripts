import numpy as np
from PIL import Image
from pathlib import Path
from multiprocessing import Pool

def flip_single_image(args):
    input_path, output_path = args
    try:
        with Image.open(input_path) as img:
            img_array = np.array(img)
            flipped_array = np.fliplr(img_array)
            flipped_img = Image.fromarray(flipped_array)
            flipped_img.save(output_path)
        return True
    except Exception as e:
        print(f'Error processing {input_path.name}: {str(e)}')
        return False

def flip_images_parallel(input_path, output_path):
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # Prepare list of tasks
    tasks = []
    for filename in input_path.glob('*.tif'):
        input_path = Path(input_path, filename)
        output_path = Path(output_path, filename)
        tasks.append((input_path, output_path))
    
    # Process images in parallel
    with Pool() as pool:
        results = pool.map(flip_single_image, tasks)
    
    processed_count = sum(results)
    print(f'Completed! Processed {processed_count} images.')

if __name__ == '__main__':
    input_path = Path('path/to/your/input/folder')
    output_path = Path('path/to/your/output/folder')
    flip_images_parallel(input_path, output_path)