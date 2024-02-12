from sklearn.linear_model import SGDClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Téléchargement du dataset MNIST
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist['data'], mnist['target']

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialiser le SGDClassifier
sgd_clf = SGDClassifier(random_state=42)

# Entraîner le modèle sur l'ensemble d'entraînement avec les classes originales (0-9)
sgd_clf.fit(X_train, y_train)

# Faire une prédiction
some_digit = X[0]  # Vous pouvez choisir une autre image si vous le souhaitez
print("Prédiction du SGDClassifier :", sgd_clf.predict([some_digit]))
