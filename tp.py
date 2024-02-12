# Importation des bibliothèques nécessaires
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

# Chargement du jeu de données MNIST
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]
y = y.astype(np.uint8)  # Conversion des étiquettes en entiers

# Division du jeu de données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Mélange de l'ensemble d'entraînement (si nécessaire, convertir en tableaux NumPy d'abord)
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# Entraînement d'un classificateur binaire pour le chiffre '5'
y_train_5 = (y_train == 5)
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

# Évaluation du classificateur avec la validation croisée
scores = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")

# Calcul de la matrice de confusion
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
conf_matrix = confusion_matrix(y_train_5, y_train_pred)

# Précision et rappel
precision = precision_score(y_train_5, y_train_pred)
recall = recall_score(y_train_5, y_train_pred)

# Sélection d'un échantillon pour la prédiction
some_digit = X_test[0]  # Modification ici pour utiliser un échantillon de X_test

# Classification multi-classes avec SGDClassifier
sgd_clf.fit(X_train, y_train)  # Entraînement sur l'ensemble complet des étiquettes
predicted_digit = sgd_clf.predict([some_digit])  # Prédiction de l'échantillon

# Classification multi-classes avec RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
forest_clf.fit(X_train, y_train)
forest_prediction = forest_clf.predict([some_digit])

# Évaluation du classificateur multi-classes
multi_scores = cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")

# Visualisation d'un échantillon de l'ensemble de données
plt.imshow(some_digit.reshape(28, 28), cmap=plt.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()

# Affichage de la prédiction
print("Prédiction du classificateur SGD :", predicted_digit[0])
print("Prédiction du classificateur RandomForest :", forest_prediction[0])
