# Big Data Project - Image Classification with PySpark

This project builds an image classification pipeline using PySpark to classify images of **cats**, **dogs**, and **horses**. It includes steps for data loading, preprocessing, model training, prediction, and visualization.

## 📁 Dataset

We use the following datasets downloaded from Kaggle:

- [Cats and Dogs Dataset](https://www.kaggle.com/datasets/tongpython/cat-and-dog)
- [Horse Breeds Dataset](https://www.kaggle.com/datasets/olgabelitskaya/horse-breeds)

To download the datasets via code:
```python
import kagglehub

# Download datasets
path1 = kagglehub.dataset_download("tongpython/cat-and-dog")
path2 = kagglehub.dataset_download("olgabelitskaya/horse-breeds")

print("Path to dataset files:", path1, path2)
