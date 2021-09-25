import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, BatchNormalization
from spektral.layers import GCNConv
import pandas as pd


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
                dropout: float = 0.4):
    x_inp = Input(shape=(number_features,))
    a_inp = Input((number_nodes,), sparse=True)
    x_1 = GCNConv(channels, activation="relu")([x_inp, a_inp])
    x_1 = BatchNormalization()(x_1)
    x_1 = Dropout(dropout)(x_1)
    x_2 = GCNConv(channels, activation="relu")([x_1, a_inp])
    x_2 = BatchNormalization()(x_2)
    x_2 = Dropout(dropout)(x_2)
    predictions = GCNConv(num_classes, activation="softmax")([x_2, a_inp])
    model = tf.keras.Model(inputs=[x_inp, a_inp], outputs=predictions)
    return model


def train_model(model, optimizer, loss, dataset, masks, epochs: int = 2000, early_stopping_patience: int = 50,
                batch_size: int = 32):
    train = get_training_function(model, loss, optimizer)
    x, adj, y = dataset[0].x, dataset[0].a, dataset[0].y
    for i in range(1, 1 + epochs):
        all_data_used = False
        while not all_data_used:
            tr_loss = train([x, adj], y, mask_tr)
            tr_acc, va_acc, te_acc = evaluate(x, adj, y, model, masks, evaluator)
            print(
                "Ep. {} - Loss: {:.3f} - Acc: {:.3f} - Val acc: {:.3f} - Test acc: "
                "{:.3f}".format(i, tr_loss, tr_acc, va_acc, te_acc)
            )

    return model


def get_training_function(model, loss_func, optimizer):
    @tf.function
    def train(inputs, target, mask):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = loss_func(target[mask], predictions[mask]) + sum(model.losses)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss
    return train
