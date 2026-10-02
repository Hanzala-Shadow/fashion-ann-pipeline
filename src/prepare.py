from pathlib import Path

import numpy as np
from tensorflow import keras


def main():
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    np.savez("data/raw/fashion_mnist.npz", x_train=x_train, y_train=y_train,
             x_test=x_test, y_test=y_test)
    print(f"Saved {len(y_train)} training and {len(y_test)} test images.")


if __name__ == "__main__":
    main()
