from cgitb import strong
from curses import keyname
from secrets import choice
import cv2
import matplotlib
from models import alexnet_conv
from models import alexnet_deconv

from models.alexnet_conv import AlexnetConv
from models.alexnet_deconv import AlextnetDeconv
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import random
import glob
import os

from utils import construct_layer_pattern, load_images, store


def plot_exp_result(img_paths, model_conv, model_deconv, img_output, pattern_output):
    k = len(img_paths)
    nrows = np.ceil(np.sqrt(k)).astype(np.int32)
    ncols = np.floor(np.sqrt(k)).astype(np.int32)

    pattern_fig, pattern_axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5, 5))
    img_fig, img_axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5, 5))

    for idx, path in enumerate(img_paths):
        img = load_images(path)
        pattern, _, _ = construct_layer_pattern(img, layer, model_conv, model_deconv)

        i = idx // ncols
        j =  idx % ncols

        img = cv2.imread(path)
        img = cv2.resize(img, (224, 224))
        img_axes[i, j].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), aspect="auto")
        img_axes[i, j].axis("off")

        pattern_axes[i, j].imshow(pattern, aspect="auto")
        pattern_axes[i, j].axis("off")
    
    img_fig.subplots_adjust(wspace=.03, hspace=.03)
    pattern_fig.subplots_adjust(wspace=.03, hspace=.03)

    img_fig.savefig(img_output)
    pattern_fig.savefig(pattern_output)

def load_random_image_path(folder, seed, n=1000):
    img_paths = []
    n_per_class = int(n/10)
    class_img_folders = os.listdir(folder)
    for class_folder in class_img_folders:
        class_folder_path = os.path.join(folder, class_folder)
        random.seed(seed)
        img_paths += random.sample(glob.glob(f"{class_folder_path}/*.JPEG"), n_per_class)
    
    return img_paths


def visualize_top_k_act_layer(layer, folder, k, model_conv, model_deconv, seed):
    act_list = []
    img_paths = load_random_image_path(folder, seed)

    # Find top k image cause maximum activation value
    for img_path in img_paths:
        img_i = load_images(img_path)
        _, act_i, _ = construct_layer_pattern(img_i, layer, model_conv, model_deconv)
        act_list.append(act_i)
    
    sorted_list_index = sorted(range(len(act_list)), key=lambda idx: act_list[idx], reverse=True)

    choice_img_paths = [img_paths[idx] for idx in sorted_list_index[:k]]
    img_output = f"outputs/visualize_exp/imgs/Layer {layer}-seed-{seed}.jpg"
    pattern_output = f"outputs/visualize_exp/patterns/Layer {layer}-seed-{seed}.jpg"

    plot_exp_result(choice_img_paths, model_conv, model_deconv, img_output, pattern_output)


if __name__ == '__main__':
    # You will change seed to generate different image
    seed = 25
    k = 9
    # layer_channel_map = {7: 128, 10:256, 12:256, 19:512, 28:512}
    layer_channel_map = {0: 64, 3:192, 6:384, 8:256, 10:256}

    # forward processing
    alexnet_conv = AlexnetConv()
    alexnet_conv.eval()
    store(alexnet_conv)
    
    # backward processing
    alexnet_deconv = AlextnetDeconv()
    alexnet_deconv.eval()
    for layer in layer_channel_map.keys():
        visualize_top_k_act_layer(layer, "./data/val", k, alexnet_conv, alexnet_deconv, seed)
        print(f"Layer {layer} done!")

