import cv2
import torch.nn as nn
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from models.alexnet_conv import AlexnetConv
from utils.utils import store, get_transformer, decode_predictions


def vtranslatation(img, translation_list):
    for delta_y in translation_list:
        T = np.float32([[1, 0, 0], [0, 1, delta_y]])
        img_translation = cv2.warpAffine(img, T, img.shape[:2])
        yield img_translation


def scale(img, scale_list):
    def central_crop(img, w, h):
        if img.shape[1] <= w or img.shape[0] <= h:
            return img
        center_x, center_y = img.shape[1] / 2, img.shape[0] / 2
        x = center_x - w/2
        y = center_y - h/2

        return img[int(y):int(y+h), int(x):int(x+w)]

    for rate in scale_list:
        new_w, new_h = int(img.shape[1]*rate), int(img.shape[0]*rate)
        transformed_img = central_crop(cv2.resize(
            img, (new_w, new_h), interpolation=cv2.INTER_LINEAR), *img.shape[:2])
        yield transformed_img


def rotation(img, rotation_list):
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    for angle in rotation_list:
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(
            img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
        yield result

def distance(v1, v2):
    v1 = v1.flatten()
    v2 = v2.flatten()
    return np.linalg.norm(v1-v2, ord=2)

def get_ft_maps(img, layer, model_conv):
    transformer = get_transformer()    
    transformed_img = transformer(img).unsqueeze_(0)
    _ = model_conv(transformed_img)

    if layer < len(model_conv.features):
        return model_conv.feature_maps[layer].clone()
    
    layer -= len(model_conv.features)
    if layer < len(model_conv.classifier):
        return model_conv.linear_ft_maps[layer].clone()

def run_vtrans_exp(orig_img, layer, model_conv, translation_list):
    results = []
    # Get feature map of original image
    original_ft = get_ft_maps(orig_img, layer, model_conv).detach().numpy()[0]

    for translated_img in vtranslatation(orig_img, translation_list):
        exp_ft_response = get_ft_maps(translated_img, layer, model_conv).detach().numpy()[0]
        delta = distance(original_ft, exp_ft_response)
        results.append(delta)
    
    return results

def run_scale_exp(orig_img, layer, model_conv, scale_list):
    results = []
    # Get feature map of original image
    original_ft = get_ft_maps(orig_img, layer, model_conv).detach().numpy()[0]

    for scaled_img in scale(orig_img, scale_list):
        exp_ft_response = get_ft_maps(scaled_img, layer, model_conv).detach().numpy()[0]
        delta = distance(original_ft, exp_ft_response)
        results.append(delta)
    
    return results

def run_rotation_exp(orig_img, layer, model_conv, rotation_list):
    results = []
    # Get feature map of original image
    original_ft = get_ft_maps(orig_img, layer, model_conv).detach().numpy()[0]

    for rotated_img in rotation(orig_img, rotation_list):
        exp_ft_response = get_ft_maps(rotated_img, layer, model_conv).detach().numpy()[0]
        delta = distance(original_ft, exp_ft_response)
        results.append(delta)
    
    return results


# # Experiment variance probability (Uncomment when run probability experiments)
# def run_vtrans_exp(orig_img, class_name, model_conv, translation_list):
#     results = []
#     # Get feature map of original image
#     transformer = get_transformer()
#     softmax = nn.Softmax(dim=1)

#     for translated_img in vtranslatation(orig_img, translation_list):
#         translated_img = transformer(translated_img).unsqueeze_(0)
#         exp_ft_response = softmax(model_conv(translated_img))
#         predictions = decode_predictions(exp_ft_response, top=1000)[0]
        
#         has_element = False
#         for item in predictions:
#             name, _, prob = item
#             if name == class_name:
#                 has_element = True
#                 results.append(prob)
#                 break
        
#         if not has_element:
#             print("Fail")
    
#     return results

# def run_scale_exp(orig_img, class_name, model_conv, scale_list):
#     results = []
#     # Get feature map of original image
#     transformer = get_transformer()
#     softmax = nn.Softmax(dim=1)

#     for scaled_img in scale(orig_img, scale_list):
#         scaled_img = transformer(scaled_img).unsqueeze_(0)
#         exp_ft_response = softmax(model_conv(scaled_img))
#         predictions = decode_predictions(exp_ft_response, top=1000)[0]
        
#         has_element = False
#         for item in predictions:
#             name, _, prob = item
#             if name == class_name:
#                 has_element = True
#                 results.append(prob)
#                 break

#         if not has_element:
#             print("Fail")
    
#     return results

# def run_rotation_exp(orig_img, class_name, model_conv, rotation_list):
#     results = []
#     # Get feature map of original image
#     transformer = get_transformer()
#     softmax = nn.Softmax(dim=1)

#     for rotated_img in rotation(orig_img, rotation_list):
#         rotated_img = transformer(rotated_img).unsqueeze_(0)
#         exp_ft_response = softmax(model_conv(rotated_img))
#         predictions = decode_predictions(exp_ft_response, top=1000)[0]
        
#         has_element = False
#         for item in predictions:
#             name, _, prob = item
#             if name == class_name:
#                 has_element = True
#                 results.append(prob)
#                 break
        
#         if not has_element:
#             print("Fail")
    
#     return results


def run_exp(layer):
    translation_list = list(range(-60, 60+1, 1))
    scale_list = np.linspace(1, 2, 50)
    rotate_list = list(range(0, 360+1, 1))
    output_folder = "outputs/stable_exp"

    # prepare results 
    vtrans_exp_results = []
    scale_exp_results = []
    rotation_exp_results = []

    vtrans_exp_output_file = os.path.join(output_folder, f"vtrans-{layer}.jpg")
    scale_exp_output_file = os.path.join(output_folder, f"scale-{layer}.jpg")
    rotation_exp_output_file = os.path.join(output_folder, f"rotation-{layer}.jpg")


    examined_img_paths = ["data/val/n02102040/ILSVRC2012_val_00004650.JPEG",
                          "data/val/n02979186/ILSVRC2012_val_00028911.JPEG",
                          "data/val/n03394916/n03394916_3731.JPEG",
                          "data/val/n03445777/n03445777_461.JPEG",
                          "data/val/n03888257/n03888257_912.JPEG"]

    legend_names = [os.path.basename(os.path.dirname(path)) for path in examined_img_paths]

    model_conv = AlexnetConv()
    model_conv.eval()
    store(model_conv)

    # Start experiment
    for i, path in enumerate(examined_img_paths):
        img = cv2.imread(path)
        img = cv2.resize(img, (224, 224))

        # Uncomment legend and comment layer argument when run probability experiment
        vtrans_exp_results.append(run_vtrans_exp(
            img, 
            layer,
            # legend_names[i],
            model_conv,
            translation_list,
            )
        )
        scale_exp_results.append(run_scale_exp(
            img,
            layer,
            # legend_names[i],
            model_conv,
            scale_list, 
            )
        )
        rotation_exp_results.append(run_rotation_exp(
            img,
            layer,
            # legend_names[i],
            model_conv,
            rotate_list,
            )
        )
    
    save_result(vtrans_exp_results, legend_names, translation_list, vtrans_exp_output_file)
    save_result(scale_exp_results, legend_names, scale_list, scale_exp_output_file)
    save_result(rotation_exp_results, legend_names, rotate_list, rotation_exp_output_file)

def save_result(exp_results, legend_names, value_list, output_file):
    exp_results = np.array(exp_results)
    fig = plt.figure()

    for result_img_i in exp_results:
        plt.plot(value_list, result_img_i)
    
    fig.legend(legend_names, loc="lower left")
    fig.savefig(output_file)


if __name__ == "__main__":
    run_exp(3)
    print("Done exp")
    run_exp(10)
    print("Done exp")
    run_exp(19)
    # # Uncomment when run probability experiment
    # run_exp(20)
    # print("Done exp")
