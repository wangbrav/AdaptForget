# AdaptForget

This repository contains the implementation of AdaptForget :

AdaptForget: A Domain-Adaptive Feature-Level Unlearning Approach for Achieving Single-Entry Data Forgetting in Healthcare

# Prerequisites:
```python
python == 3.7.0
pytorch == 1.10.1+cu113
torchvision == 0.11.2+cu113
numpy == 1.21.6
pillow == 9.2.0
tqdm == 4.64.1


conda env create --file environment.yaml
```

# Dataset:

We used seven datasets for testing, namely DermaMNIST, PathMNIST, RetinaMNIST, OCTMNIST, ASD, MDD, and Diabetes.



# Training:

1. Train the original model.

Run the corresponding dataset training code.

```python
cd ./train_the_original_models
python orresponding_dataset_train.py
```

2. Perform forgetting, auditing, and evaluation.

Run the corresponding dataset unlearningcode.

```python
cd ..
cd ./adversarial
python orresponding_dataset_unlearning.py
```

# Contact

If you have any problem about our code, feel free to contact wb1696843361@gmail.com
