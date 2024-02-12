from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Charger le jeu de données MNIST
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mise à l'échelle des caractéristiques
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(float))
X_test_scaled = scaler.transform(X_test.astype(float))

# Créer un classificateur SGD avec un nombre maximal d'itérations plus élevé
sgd_clf = SGDClassifier(max_iter=1000, random_state=42)

# Évaluer le classificateur à l'aide de la validation croisée
scores = cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")

# Afficher les résultats de l'évaluation
print("Exactitude moyenne:", scores.mean())
print("Exactitude pour chaque fold:", scores)
