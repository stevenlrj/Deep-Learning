import tensorflow as tf
import numpy as np
import scipy.io


def _conv_layer(input, weights, bias):
    """
    used to calculate the output of con2d layer, Wx+b
    """

    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1),
            padding='SAME')
    return tf.nn.bias_add(conv, bias)


def cnn_net(model_path, input_image):
    """
    :input: 
    :model_path: path to the pre-trained vgg19 mat file
    :input_image: initial image used for training

    :output:
    :net: the neural net weights given specified layer name, either in 'conv' or 'relu' or 'pool'
    :mean_pixel: teh calculated mean_pixel value of 
    """
    # in the original pre-trained VGG19 network, there're 43 layers with 
    # types like conv2d, zero-padding, max-pooling, dropout, 
    # fully-connected layers, etc. Here we're only interested in
    # conv layers, relu activation functions and pooling layers
    vgg19_layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    # First let's load in the pre-trained model
    vgg19_model = scipy.io.loadmat(model_path)
    # extract mean info
    mean = vgg19_model['normalization'][0][0][0]
    # mean is of size (224, 224, 3)
    # then calculate the mean for each channel (RGB)
    mean_pixel = np.mean(mean, axis=(0, 1))
    # mean is of size (3,)
    # exract weights info from 'layer'
    weights = vgg19_model['layers'][0]

    # initilize net variable as an empty dictionary, later when
    # we extract each layer's info, we simply refer to the key value
    net = {}
    layer_output = input_image
    for i, layer in enumerate(vgg19_layers):
        layer_type = layer[:4]
        if layer_type == 'conv':
            # for conv layers, each kernels of size (3, 3, 3, 64), and bias of size (1, 64)
            kernels, bias = weights[i][0][0][0][0]
            # Note that the information stored in tf and mat are different, here
            # we need to switch the first and second dimention of the variable
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            # reshape bias from 2d (1, 64) to 1d (64, )
            bias = bias.reshape(-1)
            layer_output = _conv_layer(layer_output, kernels, bias)
        elif layer_type == 'relu':
            layer_output = tf.nn.relu(layer_output)
        elif layer_type == 'pool':
            # proposed in the original paper, the trained results are better using
            # average pooling instead of max pool, so here we changed the
            # pooling to average pooling.
            layer_output = tf.nn.avg_pool(layer_output, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
        # finally, store the output from each layer into corresponding net (a dictionary)
        net[layer] = layer_output

    assert len(net) == len(vgg19_layers)
    return net, mean_pixel


def preprocess(image, mean_pixel):
    """
    normalize image
    """
    return image - mean_pixel


def unprocess(image, mean_pixel):
    """
    restore image as final output

    """
    return image + mean_pixel
