from matplotlib.animation import FFMpegWriter
from matplotlib.offsetbox import AnchoredText

starting_img_num = 0
total_num_imgs = 377

writer = FFMpegWriter(fps=30)

file_path = Path(f'img_{starting_img_num:09d}_Default_000.tif')
root_path = Path('/Volumes/krauss/Lisa/GMR/Array/250206/Experiment_1_ethanol/loc2_1/Pos0')
video_filepath = Path('/Volumes/krauss/Lisa/GMR/Array/250206/Experiment_1_ethanol/Results', 'Chirp_video.mp4')
ROI_path = Path('ROI_ChirpArray.json')

with open(ROI_path, 'r') as file:
    ROIs = json.load(file)

img_path = Path(root_path, file_path)
# print(img_path)

raw_im = cv.imread(img_path, cv.IMREAD_UNCHANGED)
im = cv.normalize(raw_im, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U) # Normalize to 8-bit range (0-255)

fig, ax = plt.subplots(1, 1, figsize=(7, 7), layout='tight')
ax.axis('off')
ax.set_title(f'{0:03d} / {total_num_imgs}', fontsize=20)

# Display grating locations on the image for visual confirmation
im_color = cv.cvtColor(im, cv.COLOR_GRAY2BGR)
for k, v in ROIs.items():
    if k == 'image_angle':
        continue
    y, x = v['coords']
    y_size, x_size = v['size']
    color = (255, 0, 0)
    if '_1_' in v['label']:
        color = (0, 255, 0)
    cv.rectangle(im_color, (x, y + y_size), (x + x_size, y), color, 1)
    cv.putText(im_color, v['label'], (x, y + 30),
                cv.FONT_HERSHEY_SIMPLEX, 1, color, 1)


img_chirp = ax.imshow(im_color, aspect='auto', cmap='gray', vmin=0, vmax=1)

with writer.saving(fig, video_filepath, dpi=100):
    for idx in range(0, total_num_imgs + 1):
        file_path = Path(f'img_{idx:09d}_Default_000.tif')
        img_path = Path(root_path, file_path)
        temp = cv.imread(img_path, cv.IMREAD_UNCHANGED)
        temp = cv.normalize(temp, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U) # Normalize to 8-bit range (0-255)

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
        ax.set_title(f'{idx:03d} / {total_num_imgs}', fontsize=20)

        writer.grab_frame()
        print(f'Frame {idx + 1} of {total_num_imgs} written', end='\r')