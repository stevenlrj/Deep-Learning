import numpy as np

CONTENT_LAYERS = ('relu4_2', 'relu5_2')

STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
#STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1')

def style_layer_weight_cal(style_layer_weight_exp):
    """
    the factor w_l is always equal to one divided by the number of
    active layers with a non-zero loss-weight w_l

    Input:
    :style_layer_weight_exp: default value set to 1
    """
    layer_weight = 1.0
    sum_layer_weight = 0
    style_layers_weight = {}
    for layer in STYLE_LAYERS:
        style_layers_weight[layer] = layer_weight
        sum_layer_weight += layer_weight
        layer_weight *= style_layer_weight_exp

    # normalize style layer weights
    for layer in STYLE_LAYERS:
        style_layers_weight[layer] /= sum_layer_weight
    return style_layers_weight
