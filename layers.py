import tensorflow as tf

def __weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def __bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape))

def dense(input_tensor, units):
    w = weight_variable([int(input_tensor.get_shape()[1]), units])
    b = bias_variable([units])
    return input_tensor @ w + b

