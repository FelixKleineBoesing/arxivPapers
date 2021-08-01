import tensorflow as tf
import stellargraph as sg
import pandas as pd


def build_model(nodes, edges, num_classes: int, compile_args:  dict = None):
    compile_args = {} if compile_args is None else compile_args
    if "optimizer" not in compile_args:
        compile_args["optimizer"] = "adam"
    if "loss" not in compile_args:
        compile_args["loss"] = "categorical_crossentropy"
    if "metrics" not in compile_args:
        compile_args["metrics"] = ["accuracy"]
    graph = sg.StellarGraph(nodes, edges)

    generator = sg.mapper.FullBatchNodeGenerator(graph, method="gcn")

    gcn = sg.layer.GCN(layer_sizes=[64, 32], generator=generator)
    x_inp, x_out = gcn.in_out_tensors()
    predictions = tf.keras.layers.Dense(units=num_classes, activation="softmax")(x_out)
    model = tf.keras.Model(inputs=x_inp, outputs=predictions)
    model.compile(**compile_args)
    return model, generator


def train_model(model, generator, train_label, train_idx, test_label, test_idx):
    model.fit(generator.flow(train_idx, train_label), epochs=5)

    loss, accuracy = model.evaluate(generator.flow(test_idx, test_label))
    return loss,  accuracy


