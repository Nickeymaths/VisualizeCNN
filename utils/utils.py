import os
import glob
from typing import OrderedDict
from cv2 import sort
import requests
import json
import random

import torch
import torch.nn as nn
from torchvision.transforms import transforms
import numpy as np
import cv2
from functools import partial
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


"""
this file is utils lib
"""


def decode_predictions(preds, top=5):
    """Decode the prediction of an ImageNet model

    # Arguments
        preds: torch tensor encoding a batch of predictions.
        top: Integer, how many top-guesses to return

    # Return
        A list of lists of top class prediction tuples
        One list of turples per sample in batch input.

    """


    class_index_path = 'https://s3.amazonaws.com\
/deep-learning-models/image-models/imagenet_class_index.json'

    class_index_dict = None

    if len(preds.shape) != 2 or preds.shape[1] != 1000:
        raise ValueError('`decode_predictions` expects a batch of predciton'
        '(i.e. a 2D array of shape (samples, 1000)).'
        'Found array with shape: ' + str(preds.shape)
        )

    if not os.path.exists('./data/imagenet_class_index.json'):
        r = requests.get(class_index_path).content
        with open('./data/imagenet_class_index.json', 'w+') as f:
            f.write(r.content)
    with open('./data/imagenet_class_index.json') as f:
        class_index_dict = json.load(f)

    results = []
    for pred in preds:
        top_value, top_indices = torch.topk(pred, top)
        result = [tuple(class_index_dict[str(i.item())]) + (pred[i].item(),) \
                for i in top_indices]
        result = [tuple(class_index_dict[str(i.item())]) + (j.item(),) \
        for (i, j) in zip(top_indices, top_value)]
        results.append(result)

    return results


def load_images(img_path):
    # imread from img_path
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))

    # pytorch must normalize the pic by 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
    
    img = transform(img)
    img.unsqueeze_(0)
    #img_s = img.numpy()
    #img_s = np.transpose(img_s, (1, 2, 0))
    #cv2.imshow("test img", img_s)
    #cv2.waitKey()
    return img

def get_transformer():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    return transform

def store(model):
    """
    make hook for feature map
    """
    def hook(module, input, output, key):
        if isinstance(module, nn.MaxPool2d):
           model.feature_maps[key] = output[0]
           model.pool_locs[key] = output[1]
        elif isinstance(module, nn.Linear):
            model.linear_ft_maps[key] = output
        else:
           model.feature_maps[key] = output
    
    for idx, layer in enumerate(model._modules.get('features')):    
        # _modules returns an OrderedDict
        layer.register_forward_hook(partial(hook, key=idx))
    
    for idx, layer in enumerate(model.classifier):
        if isinstance(layer, nn.Linear):
            layer.register_forward_hook(partial(hook, key=idx))

def construct_layer_pattern(img, layer, model_conv, model_deconv):
    conv_output = model_conv(img)
    
    num_feat = model_conv.feature_maps[layer].shape[1]
    new_feat_map = model_conv.feature_maps[layer].clone()

    # choose the max activations map
    act_lst = []
    for i in range(0, num_feat):
        choose_map = new_feat_map[0, i, :, :]
        activation = torch.max(choose_map)
        act_lst.append(activation.item())

    act_lst = np.array(act_lst)
    mark = np.argmax(act_lst)
    choose_map = new_feat_map[0, mark, :, :]
    new_feat_map[0, mark, :, :] = choose_map

    max_activation = torch.max(choose_map)

    # make zeros for other feature maps
    if mark == 0:
        new_feat_map[:, 1:, :, :] = 0
    else:
        new_feat_map[:, :mark, :, :] = 0
        if mark != model_conv.feature_maps[layer].shape[1] - 1:
            new_feat_map[:, mark + 1:, :, :] = 0
    
    choose_map = torch.where(choose_map==max_activation,
            choose_map,
            torch.zeros(choose_map.shape)
            )
    new_feat_map[0, mark, :, :] = choose_map
    deconv_output = model_deconv(new_feat_map, layer, mark, model_conv.pool_locs)

    new_img = deconv_output.data.numpy()[0].transpose(1, 2, 0)  # (H, W, C)
    # normalize
    new_img = (new_img - new_img.min()) / (new_img.max() - new_img.min()) * 255
    new_img = new_img.astype(np.uint8)

    return new_img, int(max_activation), conv_output


def get_layer_pattern(input_img, layer, channel, vgg16_conv, vgg16_deconv):
    """
    visualing the layer deconv result
    """

    # Forward image to model
    conv_output = vgg16_conv(input_img)

    num_feat = vgg16_conv.feature_maps[layer].shape[1]
    
    # set other feature map activations to zero
    new_feat_map = vgg16_conv.feature_maps[layer].clone()

    # choose the max activations map
    # act_lst = []
    # for i in range(0, num_feat):
    #     choose_map = new_feat_map[0, i, :, :]
    #     activation = torch.max(choose_map)
    #     act_lst.append(activation.item())

    # act_lst = np.array(act_lst)
    # mark = np.argmax(act_lst)

    # choose_map = new_feat_map[0, mark, :, :]
    choose_map = new_feat_map[0, channel, :, :]
    max_activation = torch.max(choose_map)
    
    # make zeros for other feature maps
    # if mark == 0:
    #     new_feat_map[:, 1:, :, :] = 0
    # else:
    #     new_feat_map[:, :mark, :, :] = 0
    #     if mark != vgg16_conv.feature_maps[layer].shape[1] - 1:
    #         new_feat_map[:, mark + 1:, :, :] = 0

    if channel == 0:
        new_feat_map[:, 1:, :, :] = 0
    else:
        new_feat_map[:, :channel, :, :] = 0
        if channel != vgg16_conv.feature_maps[layer].shape[1] - 1:
            new_feat_map[:, channel + 1:, :, :] = 0
    
    choose_map = torch.where(choose_map==max_activation,
            choose_map,
            torch.zeros(choose_map.shape)
            )

    # make zeros for ther activations
    # new_feat_map[0, mark, :, :] = choose_map
    new_feat_map[0, channel, :, :] = choose_map
    
    # print(torch.max(new_feat_map[0, mark, :, :]))    
    print(max_activation)
    
    deconv_output = vgg16_deconv(new_feat_map, layer, channel, vgg16_conv.pool_locs)

    new_img = deconv_output.data.numpy()[0].transpose(1, 2, 0)  # (H, W, C)
    # normalize
    new_img = (new_img - new_img.min()) / (new_img.max() - new_img.min()) * 255
    new_img = new_img.astype(np.uint8)

    return new_img, int(max_activation), conv_output


def load_random_image_path(folder, n=20):
    img_paths = []
    n_per_class = int(n/10)
    class_img_folders = os.listdir(folder)
    for class_folder in class_img_folders:
        class_folder_path = os.path.join(folder, class_folder)
        random.seed(42)
        img_paths += random.sample(glob.glob(f"{class_folder_path}/*.JPEG"), n_per_class)
    
    return img_paths