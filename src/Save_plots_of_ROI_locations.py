import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def save_plots_of_ROI_locations(root_path, path_to_first_img,
                                roi_name, save_path=Path('.'),
                                angle=None, offset=None):
    raw_im = cv2.imread(path_to_first_img, cv2.IMREAD_UNCHANGED)
    raw_im = cv2.normalize(raw_im, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)  # type: ignore
    if len(raw_im.shape) > 2:  # Check if the image has more than one channel
        raw_im = cv2.cvtColor(raw_im, cv2.COLOR_BGR2GRAY)

    with open(roi_name, 'r') as file:
        ROIs = json.load(file)

    # Rotate image
    h, w = raw_im.shape
    rot_centre = (raw_im.shape[1] // 2, raw_im.shape[0] // 2)
    if not angle:
        angle = ROIs['image_angle']
    else:
        print(f'Using user defined angle: {angle}')
    rotation_matrix = cv2.getRotationMatrix2D(rot_centre, -angle, 1.0)
    im = cv2.warpAffine(raw_im, rotation_matrix, (w, h))

    # Display grating locations on the image for visual confirmation
    im_color = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

    print(f"Creating 'Grating_locations.png' at {Path(save_path, 'Grating_locations.png')}")
    if offset:
        print(f'Adding offset of: {offset}')
    for k, v in ROIs.items():
        if k == 'image_angle':
            continue
        y, x = v['coords']
        if offset:
            x = x + offset[0]
            y = y + offset[1]
        y_size, x_size = v['size']
        color = (255, 0, 0)
        if 'B' in v['label']:
            color = (0, 255, 0)
        cv2.rectangle(im_color, (x, y + y_size), (x + x_size, y), color, 1)
        cv2.putText(im_color, v['label'], (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)

    _, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(im_color, origin='lower')
    plt.savefig(Path(save_path, 'Grating_locations.png'), dpi=300, bbox_inches='tight')

    # Calculate number of sub-plots required
    num_items = len(ROIs) // 2
    x_plts = int(np.ceil(np.sqrt(num_items)))
    y_plts = int(np.ceil(num_items / x_plts))
    if (x_plts * y_plts) < num_items:
        y_plts += 1

    # Plot ROI_1s on the image for visual confirmation
    print(f"Creating 'ROIs_1.png' at {Path(save_path, 'ROIs_1.png')}")
    _, ax = plt.subplots(x_plts, y_plts, figsize=(x_plts * 2, y_plts * 2))
    a = ax.ravel()
    count = 0
    for k, v in ROIs.items():
        if k == 'image_angle' or 'B' in k:
            continue
        y, x = v['coords']
        if offset:
            x = x + offset[0]
            y = y + offset[1]
        x_slice = slice(x, x + v['size'][1])
        y_slice = slice(y, y + v['size'][0])
        data = im[y_slice, x_slice]
        if v['flip']:
            data = np.fliplr(data)
        try:
            a[count].imshow(data)
            a[count].set_title(v['label'])
            a[count].axis('off')
            count += 1
        except IndexError:
            pass
    plt.savefig(Path(save_path, 'ROIs_1.png'), dpi=300, bbox_inches='tight')

    # PLot ROI_2s on the image for visual confirmation
    print(f"Creating 'ROIs_2.png' at {Path(save_path, 'ROIs_2.png')}")
    _, ax = plt.subplots(x_plts, y_plts, figsize=(x_plts * 2, y_plts * 2))
    a = ax.ravel()
    count = 0
    for k, v in ROIs.items():
        if k == 'image_angle' or 'A' in k:
            continue
        y, x = v['coords']
        if offset:
            x = x + offset[0]
            y = y + offset[1]
        x_slice = slice(x, x + v['size'][1])
        y_slice = slice(y, y + v['size'][0])
        data = im[y_slice, x_slice]
        if v['flip']:
            data = np.fliplr(data)
        try:
            a[count].imshow(data)
            a[count].set_title(v['label'])
            a[count].axis('off')
            count += 1
        except IndexError:
            pass
    plt.savefig(Path(save_path, 'ROIs_2.png'), dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    root_path = Path('/Volumes/krauss/Lisa/GMR/Array/250404/CRP/loc2_1')
    file_path = Path('Pos0', f'img_{0:09d}_Default_000.tif')
    path_to_first_img = Path(root_path, file_path)
    roi_name = Path('..', 'Generated_files', 'ROI_ChirpArray.json')

    save_plots_of_ROI_locations(root_path, path_to_first_img, roi_name)
