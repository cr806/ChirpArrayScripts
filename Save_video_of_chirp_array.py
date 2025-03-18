import json
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.animation import FFMpegWriter
from matplotlib.offsetbox import AnchoredText


root = Path('/Volumes/krauss/Lisa/GMR/Array/250225/loc1_1')
img_dir = Path(root, 'Split/part1')
roi_json_path = Path(root, 'Split/ROI_SU000001.json')

starting_img_num = 0
ending_img_num = 443


writer = FFMpegWriter(fps=30)

file_path = Path(f'img_{starting_img_num:09d}_Default_000.tif')
video_filepath = Path(root, 'Results', 'Chirp_video.mp4')

with open(roi_json_path, 'r') as file:
    ROIs = json.load(file)

img_path = Path(img_dir, file_path)

raw_im = cv.imread(img_path, cv.IMREAD_UNCHANGED)
im = cv.normalize(raw_im, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U) # Normalize to 8-bit range (0-255)

fig, ax = plt.subplots(1, 1, figsize=(7, 7), layout='tight')
ax.axis('off')
ax.set_title(f'{0:03d} / {ending_img_num}', fontsize=20)

# Display grating locations on the image for visual confirmation
image_angle = ROIs['image_angle']
if np.abs(image_angle) > 0:
    h, w = im.shape
    rot_centre = (im.shape[1]//2, im.shape[0]//2)
    rotation_matrix = cv.getRotationMatrix2D(rot_centre, -image_angle, 1.0)
    im = cv.warpAffine(im, rotation_matrix, (w, h))

im_color = cv.cvtColor(im, cv.COLOR_GRAY2BGR)
for k, v in ROIs.items():
    if k == 'image_angle':
        continue
    y, x = v['coords']
    y_size, x_size = v['size']
    color = (255, 0, 0)
    if 'A_' in v['label']:
        color = (0, 255, 0)
    cv.rectangle(im_color, (x, y + y_size), (x + x_size, y), color, 1)
    cv.putText(im_color, v['label'], (x, y + 30),
                cv.FONT_HERSHEY_SIMPLEX, 1, color, 1)


img_chirp = ax.imshow(im_color, aspect='auto', cmap='gray', vmin=0, vmax=1)

with writer.saving(fig, video_filepath, dpi=100):
    for idx in range(starting_img_num, ending_img_num + 1):
        file_path = Path(f'img_{idx:09d}_Default_000.tif')
        img_path = Path(img_dir, file_path)
        temp = cv.imread(img_path, cv.IMREAD_UNCHANGED)
        temp = cv.normalize(temp, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U) # Normalize to 8-bit range (0-255)

        if np.abs(image_angle) > 0: 
            rotation_matrix = cv.getRotationMatrix2D(rot_centre, -image_angle, 1.0)
            temp = cv.warpAffine(temp, rotation_matrix, (w, h))

        temp_color = cv.cvtColor(temp, cv.COLOR_GRAY2BGR)
        for k, v in ROIs.items():
            if k == 'image_angle':
                continue
            y, x = v['coords']
            y_size, x_size = v['size']
            color = (255, 0, 0)
            if 'A_' in v['label']:
                color = (0, 255, 0)
            cv.rectangle(temp_color, (x, y + y_size), (x + x_size, y), color, 1)
            cv.putText(temp_color, v['label'], (x, y + 30),
                        cv.FONT_HERSHEY_SIMPLEX, 1, color, 1)

        img_chirp.set_data(temp_color)
        ax.set_title(f'{idx:03d} / {ending_img_num}', fontsize=20)

        writer.grab_frame()
        print(f'Frame {idx + 1} of {ending_img_num + 1} written', end='\r')