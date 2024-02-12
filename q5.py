from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import fetch_openml
import numpy as np

# Téléchargement du dataset MNIST
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist['data'], mnist['target']

# Convertir les étiquettes en booléens : True pour les '5', False pour les autres chiffres
y_5 = (y == '5')

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train_5, y_test_5 = X[:60000], X[60000:], y_5[:60000], y_5[60000:]

# Initialiser le SGDClassifier
sgd_clf = SGDClassifier(random_state=42)

# Générer les prédictions avec cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

# Calculer la matrice de confusion
conf_matrix = confusion_matrix(y_train_5, y_train_pred)

print("Matrice de confusion :\n", conf_matrix)
