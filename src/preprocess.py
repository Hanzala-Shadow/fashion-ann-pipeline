from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import train_test_split


def main():
    with open("params.yaml", encoding="utf-8") as file:
        params = yaml.safe_load(file)["preprocess"]
    with np.load("data/raw/fashion_mnist.npz") as raw:
        x_train = raw["x_train"].astype("float32") / 255.0
        x_test = raw["x_test"].astype("float32") / 255.0
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, raw["y_train"], test_size=params["test_size"],
            random_state=params["seed"], stratify=raw["y_train"])
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        np.savez("data/processed/fashion_mnist.npz", x_train=x_train,
                 y_train=y_train, x_val=x_val, y_val=y_val,
                 x_test=x_test, y_test=raw["y_test"])
    print(f"Train: {len(y_train)}, validation: {len(y_val)}, test: {len(x_test)}")


if __name__ == "__main__":
    main()
