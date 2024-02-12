from sklearn.datasets import fetch_openml

# Télécharger le dataset MNIST
mnist = fetch_openml('mnist_784', version=1)

# Accéder aux données (images) et aux étiquettes (labels)
X, y = mnist['data'], mnist['target']

# Afficher la forme des données pour confirmation
print(f"Forme des données (images): {X.shape}")
print(f"Forme des labels: {y.shape}")
