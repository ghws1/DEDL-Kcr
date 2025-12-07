import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential, losses, optimizers, regularizers
import pandas as pd
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer


def concat_layer(intputs):
    return tf.concat(intputs, axis=1)


def multiply_layer(inputs):
    x, y = inputs
    return tf.multiply(x, y)


def res_net_block(input_data, filters, strides=1):
    x = layers.Conv1D(filters, kernel_size=3, strides=strides, padding='same')(input_data)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(filters, kernel_size=3, strides=1, padding='same')(x)
    if strides != 1:
        downsample = layers.Conv1D(filters, kernel_size=1, strides=strides)(input_data)
    else:
        downsample = input_data
    x = layers.Add()([x, downsample])
    x = layers.BatchNormalization()(x)
    output = layers.Activation('relu')(x)

    return output


def CNN(encode):
    inputs = tf.keras.Input(shape=(encode.shape[1], encode.shape[2]))
    x = layers.Conv1D(64, kernel_size=3)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool1D(pool_size=2, strides=1, padding='same')(x)
    x = layers.Dropout(0.4)(x)

    x = res_net_block(x, 64)
    x = layers.MaxPool1D(2)(x)
    x = layers.Dropout(0.4)(x)

    x = res_net_block(x, 64)
    x = layers.MaxPool1D(2)(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Flatten()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation=tf.nn.gelu)(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation=tf.nn.gelu, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(x)

    x = layers.Dense(32, activation=tf.nn.gelu)(x)
    output = layers.Dense(1, activation=tf.nn.sigmoid)(x)
    model = Model(inputs=[inputs], outputs=[output])
    model.compile(optimizer=optimizers.AdamW(),
                     loss=tf.keras.losses.BinaryCrossentropy(),
                     metrics='acc',
                     experimental_run_tf_function=False)
    return model


def BiGRU(windows=31):
    # 双向GRU
    input_1 = layers.Input(shape=(windows,))

    embedding = layers.Embedding(20, 200, input_length=31)

    x_embedding = embedding(input_1)

    x_2 = layers.Bidirectional(layers.GRU(100, return_sequences=True))(x_embedding)
    x_2 = layers.Bidirectional(layers.GRU(50, return_sequences=True))(x_2)

    x_2 = layers.Flatten()(x_2)

    x = layers.BatchNormalization()(x_2)

    x = layers.Dense(units=256, activation="sigmoid")(x)

    x = layers.Dropout(0.4)(x)

    x = layers.Dense(units=128, activation="sigmoid")(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dense(units=1, activation="sigmoid")(x)

    inputs = [input_1]
    outputs = [x]

    model = Model(inputs=inputs, outputs=outputs, name="Kcr")

    optimizer = optimizers.AdamW()

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=optimizer, metrics=['acc'])

    return model


def ensemble_model():

    inputs = layers.Input(shape=(3,), name='model_outputs')

    x = layers.Dense(3, activation='tanh', name='attention_weights', use_bias=False)(inputs)

    attention_scores = layers.Activation('softmax')(x)

    attention_output = layers.Multiply(name='weighted_outputs')([inputs, attention_scores])

    output = layers.Lambda(lambda x: K.sum(x, axis=1, keepdims=True), name='final_output')(attention_output)

    model = Model(inputs=inputs, outputs=output)

    model.compile(optimizer=optimizers.AdamW(),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['acc'])
    return model

