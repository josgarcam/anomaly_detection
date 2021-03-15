from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.lscp import LSCP


def anomaly_detector(algorithm, parameters):
    # Recibe una cadena con el nombre del algoritmo a emplear, y un diccionario con los parámetros de este.

    algorithms = {'ABOD': ABOD, 'CBLOF': CBLOF, 'FeatureBagging': FeatureBagging, 'HBOS': HBOS, 'IForest': IForest,
                  'KNN': KNN, 'LOF': LOF, 'MCD': MCD, 'OCSVM': OCSVM, 'PCA': PCA, 'LSCP': LSCP}
    model = algorithms[algorithm](**parameters)
    return model


# class AnomalyDetector:
#
#     def __init__(self, algorithm, parameters):
#         self.algorithm = algorithm
#         self.parameters = parameters
#
#         algorithms = {'ABOD': ABOD, 'CBLOF': CBLOF, 'FeatureBagging': FeatureBagging, 'HBOS': HBOS, 'IForest': IForest,
#                       'KNN': KNN, 'LOF': LOF, 'MCD': MCD, 'OCSVM': OCSVM, 'PCA': PCA, 'LSCP': LSCP}
#         self.model = algorithms[algorithm](**parameters)

