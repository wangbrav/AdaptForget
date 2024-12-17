# AdaptForget

This repository contains the implementation of AdaptForget :

Adaptive Feature Unlearning for Trustworthy Healthcare Data Privacy

# Prerequisites:
```python
python == 3.7.0
pytorch == 1.10.1+cu113
torchvision == 0.11.2+cu113
numpy == 1.21.6
pillow == 9.2.0
tqdm == 4.64.1


conda env create --f environment.yaml
```

# Dataset:

We used seven datasets for testing, namely DermaMNIST, PathMNIST, RetinaMNIST, OCTMNIST, ASD, MDD, and Diabetes.
All seven datasets used in this work are publicly available. The RetinaMNIST, OCTMNIST, DermaMNIST, and PathMNIST datasets are available at https://medmnist.com. The ASD dataset can be accessed at https://www.kaggle.com/datasets/fabdelja/autism-screening-for-toddlers. The MDD dataset is stored at https://physionet.org/content/mimiciii/1.4. The Diabetes dataset is stored at https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/data.



# Train,forget,audit and comparative experiment:

```python
cd machine_unlearning/
conda activate adaptforget
cd adaptforget/
python Summary_pathmnist_batch.py
```

# Contact

If you have any problem about our code, feel free to contact wb1696843361@gmail.com
