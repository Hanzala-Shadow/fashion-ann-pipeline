# Fashion-MNIST ANN

The four scripts download Fashion-MNIST, preprocess the images, train an ANN,
and evaluate it. Run commands from the project root. Settings are in params.yaml.

Install requirements in a virtual environment:

```text
python -m pip install -r requirements.txt
dvc pull
dvc repro
dvc status
```

Google Drive remote:
https://drive.google.com/drive/folders/1WPFVOYX7_GiPG3pDvg4VASi2UU2PJdR3

The default DVC Google sign-in app was blocked. A custom OAuth client must be
configured locally before dvc pull or dvc push can authenticate. Follow:
https://doc.dvc.org/user-guide/data-management/remote-storage/google-drive

Do not commit OAuth credentials. Use dvc remote modify --local for credentials.
Windows verification completed on 3 October 2026 (Asia/Karachi) using a custom
OAuth client. dvc fetch --all-commits retrieved 16 files, dvc checkout restored
the outputs, and dvc push --all-commits returned Everything is up to date.

Measured test accuracy: v1 (128 units) 87.84%; v2 (256 units) 87.87%.
Both exceed the required 85%. Final results are in metrics.json.
