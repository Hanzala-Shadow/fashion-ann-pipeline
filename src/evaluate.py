import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorflow import keras


def main():
    with open("params.yaml", encoding="utf-8") as file:
        params = yaml.safe_load(file)["train"]
    model = keras.models.load_model("models/model.h5", compile=False)
    model.compile(optimizer=keras.optimizers.Adam(params["learning_rate"]),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    with np.load("data/processed/fashion_mnist.npz") as data:
        loss, accuracy = model.evaluate(data["x_test"], data["y_test"],
                                        batch_size=params["batch_size"], verbose=0)
        predictions = model.predict(data["x_test"],
                                    batch_size=params["batch_size"], verbose=0).argmax(axis=1)
        matrix = confusion_matrix(data["y_test"], predictions, labels=range(10))
    labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
              "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(matrix, display_labels=labels).plot(
        ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    fig.tight_layout()
    fig.savefig("confusion_matrix.png", dpi=150)
    plt.close(fig)
    metrics = {"test_loss": float(loss), "test_accuracy": float(accuracy)}
    with open("metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
        file.write("\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
