import os
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchmetrics

# [white, blue, cyan, green, yellow]
color_key = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0]]

def label_tile(tile, key=color_key, axis=2):
    label_hard = [0, 0, 0, 0, 0]
    label_soft = [0., 0., 0., 0., 0.]
    for i in range(len(key)):
        count = np.count_nonzero((tile == key[i]).all(axis=axis).astype(np.uint8))
        if count > 0:
            label_hard[i] = 1
            label_soft[i] = count / tile.shape[0] / tile.shape[1]
    return label_hard, label_soft

def segmap_rgb_to_classes(segmap_rgb, n_classes=5, normalize_key=False):

    if normalize_key == True:
        key = np.asarray(color_key) / 255.
    else:
        key = np.asarray(color_key)

    segmap_classes = np.zeros((segmap_rgb.shape[0], segmap_rgb.shape[1], n_classes))
    
    white_idx = np.where(np.all(segmap_rgb == key[0], axis=-1))
    segmap_classes[white_idx[0], white_idx[1], 0] = 1

    blue_idx = np.where(np.all(segmap_rgb == key[1], axis=-1))
    segmap_classes[blue_idx[0], blue_idx[1], 1] = 1

    cyan_idx = np.where(np.all(segmap_rgb == key[2], axis=-1))
    segmap_classes[cyan_idx[0], cyan_idx[1], 2] = 1

    green_idx = np.where(np.all(segmap_rgb == key[3], axis=-1))
    segmap_classes[green_idx[0], green_idx[1], 3] = 1

    yellow_idx = np.where(np.all(segmap_rgb == key[4], axis=-1))
    segmap_classes[yellow_idx[0], yellow_idx[1], 4] = 1
    
    return segmap_classes

def segmap_classes_to_rgb(segmap_classes, normalize_key=False):

    if normalize_key == True:
        key = np.asarray(color_key) / 255.
    else:
        key = np.asarray(color_key)   

    segmap_rgb = np.zeros((segmap_classes.shape[0], segmap_classes.shape[1], 3))

    segmap_classes_num = segmap_classes.argmax(axis=-1)

    white_idx = np.where(segmap_classes_num == 0)
    segmap_rgb[white_idx[0], white_idx[1], :] = key[0]
    blue_idx = np.where(segmap_classes_num == 1)
    segmap_rgb[blue_idx[0], blue_idx[1], :] = key[1]
    cyan_idx = np.where(segmap_classes_num == 2)
    segmap_rgb[cyan_idx[0], cyan_idx[1], :] = key[2]
    green_idx = np.where(segmap_classes_num == 3)
    segmap_rgb[green_idx[0], green_idx[1], :] = key[3]
    yellow_idx = np.where(segmap_classes_num == 4)
    segmap_rgb[yellow_idx[0], yellow_idx[1], :] = key[4]
    
    return segmap_rgb



def gen_tile_ds(img_path, gt_path, save_img_path, save_gt_path, save_labels_hard_path, save_labels_soft_path, tile_w=200, tile_h=200, stride=50):
    
    os.makedirs(save_img_path, exist_ok=True)
    os.makedirs(save_gt_path, exist_ok=True)

    labels_out_hard = open(save_labels_hard_path, 'w')
    if save_labels_soft_path:
        labels_out_soft = open(save_labels_soft_path, 'w')

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

                        label_soft, label_hard = label_tile(gt_tile)

                        # we append to the labels file the name + number of the tile and the label converted to a comma-separated string
                        labels_out_hard.write('{}_{},{}\n'.format(img_list[idx].split('.')[0], count, ','.join(map(str, label_hard))))
                        if save_labels_soft_path:
                            labels_out_soft.write('{}_{},{}\n'.format(img_list[idx].split('.')[0], count, ','.join(map(str, label_soft))))

                        io.imsave(os.path.join(save_img_path, '{}_{}.png'.format(img_list[idx].split('.')[0], count)), img_tile, check_contrast=False) 
                        io.imsave(os.path.join(save_gt_path, '{}_{}.png'.format(img_list[idx].split('.')[0], count)), gt_tile, check_contrast=False) 

                        count += 1  

    labels_out_hard.close()
    if save_labels_soft_path:
        labels_out_soft.close()
    print('Done.')

def train(dataloader, model, optimizer, device, loss_fn_strong, loss_fn_weak=None):
    size = len(dataloader.dataset)
    weak_loss_coef = 0.3
    for batch, data in enumerate(dataloader):
        img, gt, labels, weak_supervision = data['image'].to(device), data['ground_truth'].to(device), data['labels'].to(device), data['weak_supervision'].to(device)
        pred = model(img)  
        loss_strong = loss_weak = 0
        if weak_supervision == False:
            loss_strong = loss_fn_strong(pred, gt)
            if loss_fn_weak:
                loss_weak = weak_loss_coef*loss_fn_weak(pred, gt, labels, weak_supervision)
        else:
            loss_weak = weak_loss_coef*loss_fn_weak(pred, gt, labels, weak_supervision)
        loss = loss_strong + loss_weak
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 20 == 0:
            loss_strong, loss_weak, current = loss_strong.item(), loss_weak.item(), batch * len(img)
            print(f"Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, device, loss_fn_strong, loss_fn_weak=None):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    weak_loss_coef=0.3
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data in dataloader:
            img, gt, labels, weak_supervision = data['image'].to(device), data['ground_truth'].to(device), data['labels'].to(device), data['weak_supervision'].to(device)
            pred = model(img)
            loss_strong = loss_weak = 0
            if weak_supervision == False:
                loss_strong = loss_fn_strong(pred, gt)
                if loss_fn_weak:
                    loss_weak = weak_loss_coef*loss_fn_weak(pred, gt, labels, weak_supervision)
            else:
                loss_weak = weak_loss_coef*loss_fn_weak(pred, gt, labels, weak_supervision)
            test_loss += loss_strong + loss_weak
    test_loss /= num_batches
    print(f"Average test loss: {test_loss:>8f} \n")


def get_metrics(model, dataloader, device):
    accuracy = torchmetrics.Accuracy().to(device)
    f1_score = torchmetrics.F1(mdmc_average='samplewise').to(device)

    acc_val = 0.
    f1_val = 0.

    model.to(device)
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            img, gt = data['image'].to(device), data['ground_truth'].to(device)
            pred = (model(img) > 0.5).int()
            gt = gt.int()
            acc_val += accuracy(pred, gt)
            f1_val += f1_score(pred, gt)
    
    return {
        'accuracy': acc_val / len(dataloader),
        'f1_score': f1_val / len(dataloader)
    }

def get_acc_per_class(model, dataloader, device, batch_size=16, n_classes=5):
    model.to(device)
    model.eval()
    acc_per_class = torch.from_numpy(np.zeros((n_classes,))).to(device)
    count_per_class = torch.from_numpy(np.zeros((n_classes,))).to(device)
    with torch.no_grad():
        for data in dataloader:
            pred = model(data['image'].to(device))
            diff=(pred > 0.5).type(torch.float) - data['ground_truth'].to(device)
            nonz=torch.count_nonzero(diff, dim=[2,3])
            acc_per_class += torch.sum(nonz,dim=0)
            count_per_class += torch.sum(torch.count_nonzero(data['ground_truth'].to(device), dim=[2,3]), dim=0)
    return 1. - acc_per_class / count_per_class