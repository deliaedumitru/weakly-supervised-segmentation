import os
from skimage import io

# [white, blue, cyan, green, yellow]
color_key = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0]]

def label_tile(tile, key=color_key):
    label = [0, 0, 0, 0, 0]
    for i in range(len(key)):
        if (tile == key[i]).all(axis=2).any():
            label[i] = 1
    return label

def gen_tile_ds(img_path, gt_path, save_img_path, save_gt_path, save_labels_path, tile_w=200, tile_h=200, stride=50):
    
    os.makedirs(save_img_path, exist_ok=True)
    os.makedirs(save_gt_path, exist_ok=True)

    labels_out = open(save_labels_path, 'w')

    img_list = os.listdir(img_path)

    for idx in range(len(img_list)):

        print('Splitting image {}/{}...'.format(idx + 1, len(img_list)))

        img = io.imread(os.path.join(img_path, img_list[idx]))
        gt = io.imread(os.path.join(gt_path, img_list[idx]))

        # we use count to number the tiles from each image
        count = 0

        # we skip the corner cases (e.g. i + tile_h >= img.shape[0] or j + tile_w >= img.shape[1]) because we will get enough examples from each image anyway
        for i in range(0, img.shape[0], stride):
            if i + tile_h < img.shape[0]:
                for j in range(0, img.shape[1], stride):
                    if j + tile_w < img.shape[1]:
                        img_tile = img[i:i+tile_h, j:j+tile_w, :]
                        gt_tile = gt[i:i+tile_h, j:j+tile_w, :]

                        # we append to the labels file the name + number of the tile and the label converted to a comma-separated string
                        labels_out.write('{}_{},{}\n'.format(img_list[idx].split('.')[0], count, ','.join(map(str, label_tile(gt_tile)))))

                        io.imsave(os.path.join(save_img_path, '{}_{}.png'.format(img_list[idx].split('.')[0], count)), img_tile, check_contrast=False) 
                        io.imsave(os.path.join(save_gt_path, '{}_{}.png'.format(img_list[idx].split('.')[0], count)), gt_tile, check_contrast=False) 

                        count += 1  

    labels_out.close()
    print('Done.')