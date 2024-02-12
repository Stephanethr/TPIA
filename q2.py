from sklearn.datasets import fetch_openml
import numpy as np

# Télécharger le dataset MNIST
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist['data'], mnist['target']

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Permutation aléatoire des données d'entraînement
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# Affichage des dimensions pour vérification
print("Dimensions de X_train:", X_train.shape)
print("Dimensions de y_train:", y_train.shape)
