expt_path = Path('/Volumes/krauss/Lisa/GMR/Array/250206/Experiment_1_ethanol/loc2_1/Pos0')
img_data_json_name = 'image_metadata_SU000001.json'

starting_img_num = 0
total_num_imgs = 377

user_feature_list_path = Path('FeatureLocation.json')
image_feature_path = Path('ImageFeatures.csv')
chip_map_path = Path('Label_templates/Chip_map.json')
template_path = Path('Label_templates/IMECII/IMEC-II_2')

# Scale factor for template to image (i.e. template larger than image feature)
user_scale_factor = (0.75, 0.75)

image_feature_list = []
for idx in range(0, total_num_imgs + 1):
    file_path = Path(f'img_{idx:09d}_Default_000.tif')
    img_path = Path(root_path, file_path)
    im = cv.imread(img_path, cv.IMREAD_UNCHANGED)
    im = cv.normalize(im, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U) # Normalize to 8-bit range (0-255)

    target = np.fliplr(im)
    target_shape = target.shape

    scale_factor, image_angle, features = find_all_labels(target, row,
                                                          user_feature_list_path,
                                                          image_feature_path,
                                                          chip_map_path, template_path,
                                                          user_scale_factor)

    image_feature_list.append(features)
    print(f'Image {idx + 1} of {total_num_imgs} processed', end='\r')

image_feature_details = pd.concat(image_feature_list, ignore_index=True)
image_feature_details.to_csv(image_feature_path, index=False)