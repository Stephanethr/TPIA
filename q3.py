# -*-coding:Utf-8 -*
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
import numpy as np

# Télécharger le dataset MNIST
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist['data'], mnist['target']

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Permutation aléatoire des données d'entraînement pour mélanger les données
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# Création des vecteurs cibles pour l'entraînement et le test
y_train_5 = (y_train == '5')  # True pour les 5, False pour les autres chiffres
y_test_5 = (y_test == '5')

# Sélection du classificateur SGD
sgd_clf = SGDClassifier(random_state=42)

# Entraînement du classificateur sur l'ensemble d'entraînement
sgd_clf.fit(X_train, y_train_5)

# Utilisation du classificateur pour prédire si une image représente un 5
prediction = sgd_clf.predict([X[36000]])

print("La prédiction pour l'image sélectionnée est :", prediction)
