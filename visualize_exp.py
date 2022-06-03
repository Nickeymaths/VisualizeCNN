from cgitb import strong
from curses import keyname
import cv2
import matplotlib
from models import alexnet_conv
from models import alexnet_deconv

from models.alexnet_conv import AlexnetConv
from models.alexnet_deconv import AlextnetDeconv
matplotlib.use('agg')
import matplotlib.pyplot as plt
import random

from models import Vgg16Conv
from models import Vgg16Deconv
from utils import decode_predictions, load_images, visualize_top_k_act_layer, store
    

if __name__ == '__main__':
    
    k = 9
    n_channel = 16
    # layer_channel_map = {7: 128, 10:256, 12:256, 19:512, 28:512}
    layer_channel_map = {0: 64, 3:192, 6:384, 8:256, 10:256}


    # forward processing
    # img = load_images(img_path)
    # vgg16_conv = Vgg16Conv()
    # vgg16_conv.eval()
    # store(vgg16_conv)
    alexnet_conv = AlexnetConv()
    alexnet_conv.eval()
    store(alexnet_conv)

    
    # conv_output = vgg16_conv(img)
    # pool_locs = vgg16_conv.pool_locs
    # print('Predicted:', decode_predictions(conv_output, top=3)[0])
    

    
    # backward processing
    # vgg16_deconv = Vgg16Deconv()
    # vgg16_deconv.eval()
    alexnet_deconv = AlextnetDeconv()
    alexnet_deconv.eval()
    for layer in layer_channel_map.keys():
        # channel = layer_channel_map[layer]
        # random.seed(42)
        # choice_channels = random.sample(range(channel), n_channel)
        # for channel in choice_channels:
        #     # visualize_top_k_act_layer(layer, channel, "./data/val", k, alexnet_conv, alexnet_deconv)
        #     visualize_top_k_act_layer(layer, channel, "./data/val", k, vgg16_conv, vgg16_deconv)

        # visualize_top_k_act_layer(layer, "./data/val", k, vgg16_conv, vgg16_deconv)
        visualize_top_k_act_layer(layer, "./data/val", k, alexnet_conv, alexnet_deconv)

        print(f"Layer {layer} done!")



    # examined_layers = [0,3,6,8,10]
    # for layer in examined_layers:
    #     visualize_top_k_act_layer(layer, channel, "./data/val", k, alexnet_conv, alexnet_deconv)
    #     print(f"Layer {layer} done!")

