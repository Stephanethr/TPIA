from sklearn.datasets import fetch_openml
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
import numpy as np

# Assurez-vous d'obtenir des arrays NumPy
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist['data'], mnist['target']

# Convertir les étiquettes en booléens : True pour les '5', False pour les autres chiffres
y_5 = (y == '5')

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train_5, y_test_5 = X[:60000], X[60000:], y_5[:60000], y_5[60000:]

# Définir le classificateur "Never 5"
class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

# Initialiser le classificateur "Never 5"
never_5_clf = Never5Classifier()

# Évaluer le classificateur "Never 5" à l'aide d'une validation croisée
never_5_scores = cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")

print("Scores d'accuracy pour le classificateur 'Never 5':", never_5_scores)
