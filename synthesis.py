import vgg
import tensorflow as tf
import numpy as np
from sys import stderr
from PIL import Image
from functools import reduce
from total_loss import loss_computaion
from feature_cal import compute_feature
from style_layer_cal import style_layer_weight_cal

##############################################################################################
# in the paper, for the images shown in Fig2, they matched the
# 1. content representation on layer 'conv4_2'
# 2. Style representations on layers 'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'

# Here specify those layer names for furtuer usage
##############################################################################################
CONTENT_LAYERS = ('relu4_2', 'relu5_2')
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
#STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1')

def synthesis(network, initial, initial_noiseblend, content, styles, iterations,
        content_weight, content_weight_blend, style_weight, style_layer_weight_exp, style_blend_weights, tv_weight,
        learning_rate):
    
    """
    
    :input
    :-styles: a list containing one or multiple images used as style image.(art work) 
    """
    # calculate the original image (content) shape
    image_shape = (1,) + content.shape
    # calculate the art image (style) shape
    style_shapes = [(1,) + style.shape for style in styles]
    # style layer weight exponentional increase - weight(layer<n+1>) = weight_exp*weight(layer<n>)
    style_layers_weights = style_layer_weight_cal(style_layer_weight_exp)

    content_features, style_features, mean_pixel = compute_feature(network, image_shape, style_shapes, content, styles)

    initial_content_coeff = 1.0 - initial_noiseblend

    with tf.Graph().as_default():
        # overall loss
        image, content_loss, style_loss, tv_loss, loss = loss_computaion(network, initial, image_shape, mean_pixel,
                                      initial_content_coeff, initial_noiseblend,
                                      content, content_weight_blend, content_weight, content_features,
                                      styles, style_layers_weights, style_features, tv_weight, style_weight, style_blend_weights)

        # optimizer setup
        # The original paper didn't specify which optimization method to use, thus here we choose the classical Adam optimizer
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        # optimization
        # optimization_process(train_step, image, content_loss, style_loss, tv_loss, loss, vgg_mean_pixel, preserve_colors, content)
        def print_progress():
            stderr.write('  content loss: %g\n' % content_loss.eval())
            stderr.write('    style loss: %g\n' % style_loss.eval())
            stderr.write('       tv loss: %g\n' % tv_loss.eval())
            stderr.write('    total loss: %g\n' % loss.eval())

        # optimization
        best_loss = float('inf')
        best = None
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            stderr.write('Optimization started...\n')
            
            for i in range(iterations):
                stderr.write('Iteration %4d/%4d\n' % (i + 1, iterations))
                train_step.run()

                last_step = (i == iterations - 1)
                if last_step:
                    print_progress()

                if last_step:
                    this_loss = loss.eval()
                    if this_loss < best_loss:
                        best_loss = this_loss
                        best = image.eval()

                    # final step, generate output image
                    img_out = vgg.unprocess(best.reshape(image_shape[1:]), mean_pixel)


                    yield (
                        (None if last_step else i),
                        img_out
                    )


