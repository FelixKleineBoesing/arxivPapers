import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, BatchNormalization
from spektral.layers import GCNConv
import pandas as pd
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy
from tensorflow.python.keras.optimizer_v2.adam import Adam


def build_model(number_nodes:  int, number_features: int, num_classes: int, channels: int = 256,
                dropout: float =  0.4):

    x_inp = Input(shape=(number_features,))
    a_inp = Input((number_nodes,), sparse=True)
    x_1 = GCNConv(channels, activation="relu")([x_inp, a_inp])
    x_1 = BatchNormalization()(x_1)
    x_1 = Dropout(dropout)(x_1)
    x_2 = GCNConv(channels, activation="relu")([x_1, a_inp])
    x_2 = BatchNormalization()(x_2)
    x_2 = Dropout(dropout)(x_2)
    predictions = GCNConv(num_classes, activation="softmax")([x_2, a_inp])
    model = tf.keras.Model(inputs=[x_inp, a_inp], outputs= predictions)
    optimizer = Adam(0.5e-2)
    loss = SparseCategoricalCrossentropy()
    return model, optimizer, loss


def train_model(model, train_label, train_idx, test_label, test_idx):
    model.fit(generator.flow(train_idx, train_label), epochs=5)

    loss, accuracy = model.evaluate(generator.flow(test_idx, test_label))
    return loss,  accuracy


def get_train_function(model, optimizer, loss_func):
    @tf.function
    def train(inputs, target, mask):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = loss_func(target[mask], predictions[mask]) + sum(model.losses)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss
