import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, BatchNormalization
from spektral.layers import GCNConv
import pandas as pd
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy
from tensorflow.python.keras.optimizer_v2.adam import Adam


def build_model(number_nodes:  int, number_features: int, num_classes: int, channels: int = 256,
                dropout: float =  0.4, compile_args:  dict = None):
    compile_args = {} if compile_args is None else compile_args
    if "optimizer" not in compile_args:
        compile_args["optimizer"] = Adam()
    if "loss" not in compile_args:
        compile_args["loss"] = SparseCategoricalCrossentropy()
    if "metrics" not in compile_args:
        compile_args["metrics"] = ["accuracy"]

    x_inp = Input(shape=(number_features,))
    a_inp = Input((number_nodes,), sparse=True)
    x_1 = GCNConv(channels, activation="relu")([x_inp, a_inp])
    x_1 = BatchNormalization()(x_1)
    x_1 = Dropout(dropout)(x_1)
    x_2 = GCNConv(channels, activation="relu")([x_1, a_inp])
    x_2 = BatchNormalization()(x_2)
    x_2 = Dropout(dropout)(x_2)
    predictions = GCNConv(num_classes, activation="softmax")([x_2, a_inp])
    model = tf.keras.Model(inputs=x_inp, outputs= predictions)
    model.compile(**compile_args)
    return model


def train_model(model, generator, train_label, train_idx, test_label, test_idx):
    model.fit(generator.flow(train_idx, train_label), epochs=5)

    loss, accuracy = model.evaluate(generator.flow(test_idx, test_label))
    return loss,  accuracy


