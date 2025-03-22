import tensorflow as tf
import numpy as np

act = tf.keras.layers.LeakyReLU(0.1)

def class_and_decoder(image_shape, nclasses):
    nlayers = int(np.log2(image_shape[0]))
    x = input_layer = tf.keras.layers.Input(image_shape, name='input')
    filters = 32
    for i in range(nlayers):
        if i % 2 == 0 and i > 0:
            filters *= 2
        x = tf.keras.layers.Conv2D(filters, 3, 2, 'same', activation=act)(x)
    mid = tf.keras.layers.Layer(name='latent')(x)
    x = tf.keras.layers.Flatten()(mid)
    out_classes = tf.keras.layers.Dense(nclasses, name='classes')(x)
    x = mid
    for i in range(nlayers):
        if i % 2 == 0:
            filters //= 2
        x = tf.keras.layers.Conv2DTranspose(filters, 3, 2, 'same', activation=act)(x)
    out_image = tf.keras.layers.Conv2D(1, 1, activation='sigmoid', name='decoder')(x)
    return tf.keras.Model(input_layer, (out_classes, out_image))

def load(filename):
    return tf.keras.models.load_model(filename, {'LeakyReLU': act})

def subset(model, layer1, layer2):
    input_layer = model.get_layer(layer1)
    layers = [model.get_layer(layer2)]
    while layers[0] != input_layer:
        for l in model.layers:
            if l.output is layers[0].input:
                layers.insert(0, l)
                break
    x = input_layer = tf.keras.layers.Input(layers[0].input.shape[1:])
    for l in layers:
        x = l(x)
    return tf.keras.Model(input_layer, x)
