import csv
from pathlib import Path

import numpy as np
import yaml
from tensorflow import keras
import tensorflow as tf


def main():
    with open("params.yaml", encoding="utf-8") as file:
        params = yaml.safe_load(file)["train"]
    keras.utils.set_random_seed(params["seed"])
    tf.config.experimental.enable_op_determinism()
    model = keras.Sequential([
        keras.Input(shape=(28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(params["dense_units"], activation="relu"),
        keras.layers.Dropout(params["dropout_rate"]),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(params["learning_rate"]),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    with np.load("data/processed/fashion_mnist.npz") as data:
        history = model.fit(
            data["x_train"], data["y_train"],
            validation_data=(data["x_val"], data["y_val"]),
            epochs=params["epochs"], batch_size=params["batch_size"])
    Path("models").mkdir(exist_ok=True)
    model.save("models/model.h5")
    with open("models/history.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", *history.history])
        for epoch, values in enumerate(zip(*history.history.values()), start=1):
            writer.writerow([epoch, *values])


if __name__ == "__main__":
    main()
