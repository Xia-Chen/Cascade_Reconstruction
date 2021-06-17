import tensorflow as tf
from tflearn.layers.conv import global_avg_pool


def conv_layer(input, filter, kernel, stride, padding='SAME', layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride, padding=padding)
        return network


def Global_Average_Pooling(x):
    return global_avg_pool(x, name='Global_avg_pooling')


def Relu(x):
    return tf.nn.relu(x)


def Sigmoid(x) :
    return tf.nn.sigmoid(x)


def Fully_connected(x, units=100, layer_name='fully_connected'):
    with tf.name_scope(layer_name):
        return tf.layers.dense(inputs=x, use_bias=False, units=units)


def squeeze_excitation_layer(input_x, out_dim, ratio=4, layer_name='SE'):
    with tf.name_scope(layer_name):

        squeeze = Global_Average_Pooling(input_x)

        excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name+'_fcn1')
        excitation = Relu(excitation)
        excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name+'__fcn2')
        excitation = Sigmoid(excitation)

        excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
        scale = input_x * excitation

        return scale