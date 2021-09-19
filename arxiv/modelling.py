import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, BatchNormalization
from spektral.layers import GCNConv
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy
from tensorflow.python.keras.optimizer_v2.adam import Adam
from spektral.data.loaders import SingleLoader


def evaluate(graph, model, masks, evaluator):
    x, adj, y = graph.x, graph.a, graph.y
    p = model([x, adj], training=False)
    p = p.numpy().argmax(-1)[:, None]
    tr_mask, va_mask, te_mask = masks["train"], masks["val"], masks["test"]
    tr_auc = evaluator.eval({"y_true": y[tr_mask], "y_pred": p[tr_mask]})["acc"]
    va_auc = evaluator.eval({"y_true": y[va_mask], "y_pred": p[va_mask]})["acc"]
    te_auc = evaluator.eval({"y_true": y[te_mask], "y_pred": p[te_mask]})["acc"]
    return tr_auc, va_auc, te_auc


def build_model(number_nodes:  int, number_features: int, num_classes: int, channels: int = 256,
                dropout: float = 0.4, compile_args: dict = None):
    compile_args = {} if compile_args is None else compile_args
    if "optimizer" not in compile_args:
        compile_args["optimizer"] = Adam(0.5e-2)
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
    model = tf.keras.Model(inputs=[x_inp, a_inp], outputs= predictions)
    model.compile(**compile_args)
    print(model.summary())
    return model


def train_model(model, dataset, masks, epochs: int = 2000, early_stopping_patience: int = 50):
    train_loader = SingleLoader(dataset=dataset, sample_weights=masks["train"], epochs=epochs)
    val_loader = SingleLoader(dataset=dataset, sample_weights=masks["val"], epochs=epochs)

    model.fit(train_loader.load(),
              steps_per_epoch=train_loader.steps_per_epoch,
              validation_data=val_loader.load(),
              validation_steps=val_loader.steps_per_epoch,
              epochs=epochs,
              callbacks=[EarlyStopping(patience=early_stopping_patience, restore_best_weights=True)]
              )
    return model