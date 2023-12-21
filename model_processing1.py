import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

iris = load_iris()

X = iris.data
y = iris.target

# print(f"Feature names : {iris.feature_names}")
# print(f"Target names : {iris.target_names}")
# print("First 10 rows : \n", X[:10])
