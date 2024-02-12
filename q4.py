from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import fetch_openml
import numpy as np

# Télécharger le dataset MNIST
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist['data'], mnist['target']

# Convertir les étiquettes en booléens : True pour les '5', False pour les autres chiffres
y_5 = (y == '5')

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train_5, y_test_5 = X[:60000], X[60000:], y_5[:60000], y_5[60000:]

# Initialiser le SGDClassifier
sgd_clf = SGDClassifier(random_state=42)

# Effectuer la validation croisée
scores = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")

print("Scores d'accuracy :", scores)
