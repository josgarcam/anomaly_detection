# anomaly_detection
Detección de anomalías. Emplea un dataset para entrenar el modelo. Detecta anomalías en él. Se le pasan nuevos valores y detecta si es o no outlier en base al modelo entrenado.

## Descripción
Los algoritmos para la detección de outliers pertenecen al módulo [PyOD](https://pyod.readthedocs.io/en/latest/index.html). 
En la siguiente tabla se recogen todos los que se incluyen en dicho módulo. En negrita aparecen aquellos que han sido probados y considerados en la implementación.

|Type|Abbr|Algorithm|Year|Class|
|--- |--- |--- |--- |--- |
|Linear Model|**PCA**|**Principal Component Analysis (the sum of weighted projected distances to the eigenvector hyperplanes)**|2003|pyod.models.pca.PCA|
|Linear Model|**MCD**|**Minimum Covariance Determinant (use the mahalanobis distances as the outlier scores)**|1999|pyod.models.mcd.MCD|
|Linear Model|**OCSVM**|**One-Class Support Vector Machines**|2001|pyod.models.ocsvm.OCSVM|
|Linear Model|LMDD|Deviation-based Outlier Detection (LMDD)|1996|pyod.models.lmdd.LMDD|
|Proximity-Based|**LOF**|**Local Outlier Factor**|2000|pyod.models.lof.LOF|
|Proximity-Based|COF|Connectivity-Based Outlier Factor|2002|pyod.models.cof.COF|
|Proximity-Based|**CBLOF**|**Clustering-Based Local Outlier Factor**|2003|pyod.models.cblof.CBLOF|
|Proximity-Based|LOCI|LOCI: Fast outlier detection using the local correlation integral|2003|pyod.models.loci.LOCI|
|Proximity-Based|**HBOS**|**Histogram-based Outlier Score**|2012|pyod.models.hbos.HBOS|
|Proximity-Based|**kNN**|**k Nearest Neighbors (use the distance to the kth nearest neighbor as the outlier score)**|2000|pyod.models.knn.KNN|
|Proximity-Based|AvgKNN|Average kNN (use the average distance to k nearest neighbors as the outlier score)|2002|pyod.models.knn.KNN|
|Proximity-Based|MedKNN|Median kNN (use the median distance to k nearest neighbors as the outlier score)|2002|pyod.models.knn.KNN|
|Proximity-Based|SOD|Subspace Outlier Detection|2009|pyod.models.sod.SOD|
|Proximity-Based|ROD|Rotation-based Outlier Detection|2020|pyod.models.rod.ROD|
|Probabilistic|**ABOD**|**Angle-Based Outlier Detection**|2008|pyod.models.abod.ABOD|
|Probabilistic|FastABOD|Fast Angle-Based Outlier Detection using approximation|2008|pyod.models.abod.ABOD|
|Probabilistic|COPOD|COPOD: Copula-Based Outlier Detection|2020|pyod.models.copod.COPOD|
|Probabilistic|MAD|Median Absolute Deviation (MAD)|1993|pyod.models.mad.MAD|
|Probabilistic|SOS|Stochastic Outlier Selection|2012|pyod.models.sos.SOS|
|Outlier Ensembles|**IForest**|**Isolation Forest**|2008|pyod.models.iforest.IForest|
|Outlier Ensembles| |**Feature Bagging**|2005|pyod.models.feature_bagging.FeatureBagging|
|Outlier Ensembles|**LSCP**|**LSCP: Locally Selective Combination of Parallel Outlier Ensembles**|2019|pyod.models.lscp.LSCP|
|Outlier Ensembles|XGBOD|Extreme Boosting Based Outlier Detection (Supervised)|2018|pyod.models.xgbod.XGBOD|
|Outlier Ensembles|LODA|Lightweight On-line Detector of Anomalies|2016|pyod.models.loda.LODA|
|Neural Networks|AutoEncoder|Fully connected AutoEncoder (use reconstruction error as the outlier score)|2015|pyod.models.auto_encoder.AutoEncoder|
|Neural Networks|VAE|Variational AutoEncoder (use reconstruction error as the outlier score)|2013|pyod.models.vae.VAE|
|Neural Networks|Beta-VAE|Variational AutoEncoder (all customized loss term by varying gamma and capacity)|2018|pyod.models.vae.VAE|
|Neural Networks|SO_GAAL|Single-Objective Generative Adversarial Active Learning|2019|pyod.models.so_gaal.SO_GAAL|
|Neural Networks|MO_GAAL|Multiple-Objective Generative Adversarial Active Learning|2019|pyod.models.mo_gaal.MO_GAAL|

Cada uno tiene sus propios parámetros, pero comparten los siguientes métodos y atributos:

* fit(X): Entrena el modelo
* predict(X): Predice si una o varias observaciones son outliers o no según el modelo previamente entrenado.
* predict_proba(X): Predice la probabilidad de que una o varias observaciones sean outliers.
* decision_function(X): Predice la puntuación de anomalía de X

* Parameters: Parámetros dados en la generación del modelo.
* decision_scores_: La puntuación para ser outliers de los datos de entrenamiento (Mayor mientras más anormal sea cada observación).
* threshold_: Umbral usado para generar la etiqueta binaria (outlier o inlier) en función de la puntuación de cada observación. Se calcula como las *n_samples * contamination* observaciones más anormales según decision_scores_.
* labels_ : Etiqueta binaria de los datos de entrenamiento (0 inliers, 1 outliers)

### Implementación
[detection.py](https://github.com/josgarcam/anomaly_detection/blob/main/detection.py)

Los modelos anteriores en negrita se han recogido bajo la función **anomaly_detector(algorithm, parameters)**, cuyos argumentos de entrada son:

* algorithm: String con la abreviatura del modelo que se quiere emplear (segunda columna).
* parameters: Diccionario con los parámetros del modelo.

La función devuelve un objeto que integra el modelo. Este debe ser entrenado previamente antes de ser usado para predecir.

Nota: Si se quieren añadir más modelos tan solo es necesario incluirlos en el diccionario algorithms de la línea 17:

`algorithms = {'ABOD': ABOD, 'CBLOF': CBLOF, 'FeatureBagging': FeatureBagging, 'HBOS': HBOS, 'IForest': IForest,
                  'KNN': KNN, 'LOF': LOF, 'MCD': MCD, 'OCSVM': OCSVM, 'PCA': PCA, 'LSCP': LSCP}`

## Ejemplo de funcionamiento

En [main.py](https://github.com/josgarcam/anomaly_detection/blob/main/main.py) se recoge un ejemplo de uso. En este se genera un dataset ficticio para el entrenamiento de un modelo de tipo "ABOD" y, posteriormente, es empleado para predecir si dos observaciones extras son outliers o inliers.

![res_ejemplo](https://user-images.githubusercontent.com/80322524/111287564-a1c67200-8643-11eb-9c93-66ad795a35e1.png)

Los puntos corresponden con las observaciones del conjunto de entrenamiento, mientras que las X son las dos extras que se quieren predecir. El color verde indica que el modelo considera que es inlier y el rojo que es outlier.

## Efecto del parámetro de entrada contaminación
A cada muestra del conjunto de datos se le asigna un valor de anormalidad cuando el modelo es entrenado. Este valor se puede consultar con el atributo *decision_scores_*. Luego, haciendo uso del parámetro de contaminación, que debe estar en el intervalo (0, 0.5], el algoritmo establece un umbral que separa el porcentaje del total de muetras indicado por la contaminación con mayor valor de anormalidad con el objetivo de marcarlas como anomalías.

![Figure_2](https://user-images.githubusercontent.com/80322524/118446984-8550bf80-b6f0-11eb-8271-dab17c32bbdb.png)

En la figura anterior, cada barra representa una muestra del conjunto de entrenamiento y la altura de estas indica el nivel de anormalidad. La barra horizontal marca el umbral calculado que indica el límite de anormalidad a partir del cual las muestras se consideran anomalías. Para generar la gráfica, el parámetro contaminación se ha ajustado al mínimo valor posible y, por ello, solo una barra supera el umbral. Si se incrementa el parámetro, la altura de las barras permanecería igual, pero el umbral descendería hasta situarse por debajo del porcentaje especificado de muestras. 

