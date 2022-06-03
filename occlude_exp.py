from cProfile import label
from operator import index
from tkinter.messagebox import NO
from cv2 import sort
from models.alexnet_conv import AlexnetConv
from models.alexnet_deconv import AlextnetDeconv
import matplotlib.pyplot as plt
from utils import load_images, construct_layer_pattern, load_random_image_path, decode_predictions, store, get_transformer
from torchvision.transforms import transforms
import cv2
import numpy as np
import os
import pandas as pd


def run_exp(layer, model_conv, model_deconv, k):
    occluded_size = 50
    data_folder_path = "data/val"
    img_paths = load_random_image_path(data_folder_path, n=20)

    act_list = []

    # Get transformer
    transformer = get_transformer()

    for i, img_path in enumerate(img_paths):
        img_i = load_images(img_path)
        _, act_i, _ = construct_layer_pattern(
            img_i, layer, model_conv, model_deconv)
        act_list.append(act_i)

    sort_list_index = sorted(range(len(act_list)),
                             key=lambda idx: act_list[idx], reverse=True)
    top_act_img_paths = [img_paths[idx] for idx in sort_list_index[:k]]
    for path in top_act_img_paths:
        img = cv2.imread(path)
        img = cv2.resize(img, (224, 224))

        pattern, act, output = construct_layer_pattern(
            transformer(img).unsqueeze_(0), layer, model_conv, model_deconv)
        plt.imshow(pattern)
        plt.savefig(
            f"outputs/occlude_exp/pattern_img_{os.path.basename(path)}.jpg")

        act_list = []
        probs = []
        labels = []

        for occluded_img in get_occluded_img(img, occluded_size):
            # plt.imshow(occluded_img)
            occluded_img = transformer(occluded_img)
            occluded_img.unsqueeze_(0)

            pattern, act, output = construct_layer_pattern(
                occluded_img, layer, model_conv, model_deconv)
            predict = decode_predictions(output, top=5)[0]

            probs += [prob for _, _, prob in predict]
            labels += [label for label, _, _ in predict]
            act_list.append(act)
            # plt.imshow(pattern)
            # plt.savefig("occlude_exp_result.jpg")
            # plt.savefig("occlude_img_exp_result.jpg")

            print(f"Predict: {predict}")
            print(f"Activation value: {act}")

        print(f"----------Image path {path}----------")

        h = int(np.sqrt(len(act_list)))

        np.savetxt(
            f"outputs/occlude_exp/img{os.path.basename(path)}-acts.csv", act_list, delimiter=",")
        np.savetxt(
            f"outputs/occlude_exp/img{os.path.basename(path)}-probs.csv", probs, delimiter=",")
        pd.DataFrame({"labels": labels}).to_csv(
            f"outputs/occlude_exp/img{os.path.basename(path)}-labels.csv", sep=",")


def get_occluded_img(img, occlude_size):
    img_tmp = img.copy()
    for i in range(0, len(img)-occlude_size, 10):
        for j in range(0, len(img)-occlude_size, 10):
            img_tmp = img.copy()
            yield cv2.rectangle(img_tmp, (i, j), (i+occlude_size, j+occlude_size), (0, 0, 0), cv2.FILLED)
            # yield cv2.rectangle(img_tmp, (75, 75), (75+50, 75+50), (0, 0, 0), cv2.FILLED)


if __name__ == "__main__":
    examined_layer = 10
    alexnet_conv = AlexnetConv()
    alexnet_conv.eval()
    store(alexnet_conv)

    alexnet_deconv = AlextnetDeconv()
    alexnet_deconv.eval()
    run_exp(examined_layer, alexnet_conv, alexnet_deconv, 3)
