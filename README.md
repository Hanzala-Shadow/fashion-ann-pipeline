# Fashion-MNIST ANN

This project classifies clothing images using a fully connected neural network.
The four scripts download data, preprocess it, train the model, and evaluate it.
Run commands from the project root. Hyperparameters are stored in params.yaml.

Install requirements in a virtual environment, then run `dvc repro`.
Configure the Google Drive remote and use `dvc push` to store the artifacts.

The target test accuracy is 85%. Check metrics.json after training.

Run all scripts from the project root so relative paths resolve correctly.
