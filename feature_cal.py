import vgg
import tensorflow as tf
import numpy as np
from collections import defaultdict

CONTENT_LAYERS = ('relu4_2', 'relu5_2')

STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
#STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1')

def compute_feature(network, image_shape, style_shapes, content, styles):
    """
    Used to extract content feature and style feature, this is
    one of the most important key functions used in the paper
    """
    # create a dictionary to store content feature
    content_features = {}
    # for each style image input, create a dictionary to store style feature
    style_features = [{} for _ in styles]

    # compute content features
    #g = tf.Graph()
    with tf.Graph().as_default(), tf.Session() as sess:
        image = tf.placeholder('float', shape=image_shape)
        # load the neural net and mean_pixel from VGG19 architecture
        net, mean_pixel = vgg.cnn_net(network, image)
        # preprocess content image, subtract the mean
        preprocess_content = np.array([vgg.preprocess(content, mean_pixel)])
        # image is the original image, now we calculate P_l, its respective feature representaion in layer l.
        for layer in CONTENT_LAYERS:
            content_features[layer] = net[layer].eval(feed_dict={image: preprocess_content})
            

    # compute style features 
    for i in range(len(styles)):
        #g = tf.Graph()
        with tf.Graph().as_default(), tf.Session() as sess:
            image = tf.placeholder('float', shape=style_shapes[i])
            net, _ = vgg.cnn_net(network, image)
            # preprocess each input style image, subtract the mean
            preprocess_style = np.array([vgg.preprocess(styles[i], mean_pixel)])
            for j in STYLE_LAYERS:
                # compute the feature map in layer j
                feature_map = net[j].eval(feed_dict = {image: preprocess_style})
                feature_map = np.reshape(feature_map, (-1, feature_map.shape[3]))
                # Compute Gram Matrix
                # By definition, gram matrix represents the feature correlations, 
                # which can be computed as inner product btw vectorized feature map i and j
                gram_matrix = np.matmul(feature_map.T, feature_map) / feature_map.size
                style_features[i][j] = gram_matrix

    return content_features, style_features, mean_pixel
