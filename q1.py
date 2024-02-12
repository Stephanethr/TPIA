from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np

# Assurez-vous d'obtenir des arrays NumPy
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist['data'], mnist['target']

some_digit_index = 36000
some_digit = X[some_digit_index]
some_digit_image = some_digit.reshape(28, 28)  # Redimensionner de 784 à 28x28 pixels pour l'affichage

plt.imshow(some_digit_image, cmap='binary')
plt.axis("off")
plt.show()

print(f"Label de l'image: {y[some_digit_index]}")
