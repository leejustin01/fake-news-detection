import numpy as np

X_train = np.load("./data/processed/X_train.npy")
X_test = np.load("./data/processed/X_test.npy")
y_train = np.load("./data/processed/y_train.npy")
y_test = np.load("./data/processed/y_test.npy")

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)