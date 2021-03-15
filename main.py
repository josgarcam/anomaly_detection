from detection import anomaly_detector
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

# Generación del dataset de prueba
blobs_params = dict(random_state=0, n_samples=50, n_features=2)
dataset = make_blobs(centers=[[0.5, 0.5], [0.5, 0.5]], cluster_std=0.3, **blobs_params)[0]

# Generación del detector con los parámetros deseados
detector_params = dict(n_neighbors=15)
detector = anomaly_detector('ABOD', detector_params)

# Se entrena el modelo
detector.fit(dataset)

# Resultado de la detección en el dataset de entrenamiento
y_pred = detector.labels_

# Se aplica el modelo sobre dos observaciones extras
extra = np.array([[0.6, 0.6],
                  [0, 0]])
y_pred_extra = detector.predict(extra)

# Representación de los resultados
color = np.array(['#00ff00', '#ff0000'])
plt.figure()
plt.scatter(dataset[:, 0], dataset[:, 1], color=color[y_pred])
plt.scatter(extra[:, 0], extra[:, 1], color=color[y_pred_extra], marker='x')
plt.show()


