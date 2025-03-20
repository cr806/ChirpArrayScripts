import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
import json
import itertools

from pathlib import Path


file_path = Path('Pos0', f'img_{0:09d}_Default_000.tif')
root_path = Path('/Volumes/krauss/Lisa/GMR/Array/250318/sensor_1_CRP/loc1_sensor1_1')
img_path = Path(root_path, file_path)
save_path = Path(root_path)

raw_im = cv.imread(img_path, cv.IMREAD_UNCHANGED)
raw_im = cv.normalize(raw_im, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)  # Normalize to 8-bit range (0-255)

ROI_path = Path(root_path, 'ROI_ChirpArray.json')
with open(ROI_path, 'r') as file:
    ROIs = json.load(file)

# Rotate image
h, w = raw_im.shape
rot_centre = (raw_im.shape[1]//2, raw_im.shape[0]//2)
rotation_matrix = cv.getRotationMatrix2D(rot_centre, -ROIs['image_angle'], 1.0)
im = cv.warpAffine(raw_im, rotation_matrix, (w, h))

# Display grating locations on the image for visual confirmation
im_color = cv.cvtColor(im, cv.COLOR_GRAY2BGR)

for k, v in ROIs.items():
    if k == 'image_angle':
        continue
    y, x = v['coords']
    y_size, x_size = v['size']
    color = (255, 0, 0)
    if 'B' in v['label']:
        color = (0, 255, 0)
    cv.rectangle(im_color, (x, y + y_size), (x + x_size, y), color, 1)
    cv.putText(im_color, v['label'], (x, y + 30),
                cv.FONT_HERSHEY_SIMPLEX, 1, color, 1)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(im_color)
plt.savefig(Path(save_path, 'Grating_locations.png'), dpi=300, bbox_inches='tight')

# Calculate number of sub-plots required
num_items = len(ROIs) // 2
x_plts = int(np.ceil(np.sqrt(num_items)))
y_plts = int(np.ceil(num_items / x_plts))
if (x_plts * y_plts) < num_items:
    y_plts += 1

# Plot ROI_1s on the image for visual confirmation
fig, ax = plt.subplots(x_plts, y_plts, figsize=(x_plts * 2, y_plts * 2))
a = ax.ravel()
count = 0
for k, v in ROIs.items():
    if k == 'image_angle' or 'B' in k:
        continue
    y_slice = slice(v['coords'][0], v['coords'][0] + v['size'][0])
    x_slice = slice(v['coords'][1], v['coords'][1] + v['size'][1])
    data = im[y_slice, x_slice]
    if v['flip']:
        data = np.fliplr(data)
    try:
        a[count].imshow(data)
        a[count].set_title(v['label'])
        a[count].axis('off')
        count += 1
    except:
        pass
plt.savefig(Path(save_path, 'ROIs_1.png'), dpi=300, bbox_inches='tight')

# PLot ROI_2s on the image for visual confirmation
fig, ax = plt.subplots(x_plts, y_plts, figsize=(x_plts * 2, y_plts * 2))
a = ax.ravel()
count = 0
for k, v in ROIs.items():
    if k == 'image_angle' or 'A' in k:
        continue
    y_slice = slice(v['coords'][0], v['coords'][0] + v['size'][0])
    x_slice = slice(v['coords'][1], v['coords'][1] + v['size'][1])
    data = im[y_slice, x_slice]
    if v['flip']:
        data = np.fliplr(data)
    try:
        a[count].imshow(data)
        a[count].set_title(v['label'])
        a[count].axis('off')
        count += 1
    except:
        pass
plt.savefig(Path(save_path, 'ROIs_2.png'), dpi=300, bbox_inches='tight')