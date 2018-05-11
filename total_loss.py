import vgg
import tensorflow as tf
import numpy as np
from functools import reduce


CONTENT_LAYERS = ('relu4_2', 'relu5_2')
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
#STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1')

def _tensor_size(tensor):
    """
    calculate input tensor size
    """
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)


def loss_computaion(network, initial, image_shape, mean_pixel, initial_content_coeff, initial_noiseblend,
                    content, content_weight_blend, content_weight, content_features, styles, style_layers_weights, 
                    style_features, tv_weight, style_weight, style_blend_weights):
    """
    Define our objective function to optimize over.
    In the original paper, the author used weighted content loss and
    style loss to calculate total loss. In our implementation, we add
    one other term into total loss, the total variation loss.
    By denoising, we can get a much nicer output.

    For more information on total variation denoise, you can refer
    to our final report for detailed introduction and reasons we included it here
    
    'content_weight' refers to alpha value in original paper
    'style_weight' refers to beta value in original paper
    """
    if initial is None:
        noise = np.random.normal(size=image_shape, scale=np.std(content) * 0.1)
        initial = tf.random_normal(image_shape) * 0.256
    else:
        initial = np.array([vgg.preprocess(initial, mean_pixel)])
        initial = initial.astype('float32')
        noise = np.random.normal(size=image_shape, scale=np.std(content) * 0.1)
        initial = (initial) * initial_content_coeff + (tf.random_normal(image_shape) * 0.256) * initial_noiseblend
    image = tf.Variable(initial)
    net, _ = vgg.cnn_net(network, image)

    content_layer_weights = {}
    for layer in CONTENT_LAYERS:
        content_layer_weights[layer] = content_weight_blend
        content_weight_blend = 1 - content_weight_blend
     
    # calculation of content loss 
    content_loss = 0
    loss = 0.0
    for layer in CONTENT_LAYERS:
        loss = content_layer_weights[layer] * content_weight * (2 * tf.nn.l2_loss(net[layer] - content_features[layer]) /content_features[layer].size)
        content_loss += loss
    
    # calculation of style loss 
    style_loss = 0
    loss = 0.0 
    for i in range(len(styles)):
        for style_layer in STYLE_LAYERS:
            layer = net[style_layer]
            _, height, width, number = map(lambda i: i.value, layer.get_shape())
            size = height * width * number
            feats = tf.reshape(layer, (-1, number))
            # compute Gram Matrix
            gram_mat = tf.matmul(tf.transpose(feats), feats) / size
            style_gram = style_features[i][style_layer]
            loss += style_layers_weights[style_layer] * 2 * tf.nn.l2_loss(gram_mat - style_gram) / style_gram.size
        style_loss += style_weight * style_blend_weights[i] * loss

    # total variation denoising
    # one improvement compared to original paper
    tv_y_size = _tensor_size(image[:,1:,:,:])
    tv_x_size = _tensor_size(image[:,:,1:,:])
    tv_loss = tv_weight * 2 * ((tf.nn.l2_loss(image[:,1:,:,:] - image[:,:image_shape[1]-1,:,:]) /tv_y_size) +(tf.nn.l2_loss(image[:,:,1:,:] - image[:,:,:image_shape[2]-1,:]) /tv_x_size))
    # overall loss
    totalloss = content_loss + style_loss + tv_loss
    return image, content_loss, style_loss, tv_loss, totalloss

