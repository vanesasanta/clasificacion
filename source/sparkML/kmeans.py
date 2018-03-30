import sys
import os
from pyspark import SparkContext, SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import StringIndexer, OneHotEncoder,VectorIndexer, VectorAssembler

import datetime

from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors, VectorUDT

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("Kmeans")\
        .getOrCreate()

#tiempo preparación de los datos
timestartINI = datetime.datetime.now()

# Load the data stored as dataframe
df = spark.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("/FileStore/tables/corrected_heading")

df.cache()

from pyspark.sql import functions as F
#solo 2 clases, normal y attack (todos los que no son normal)
#df = df.withColumn("class", F.when(df["class"]!="normal","attack").otherwise(df["class"]))

#df.printSchema()

#38 classes
df_cluster = df.groupBy("class").count()

df_cluster.show(40)
nm_cluster = df_cluster.count()

print("nm_cluter " + str(nm_cluster))

##########################################
#preparación de los datos
# fichero tipo label y features
##########################################

#categoricas
categorical_columns = ["protocol_type","service","flag","land","logged_in","is_host_login","is_guest_login"]

#numericas = todas las columnas menos las categoricas (incluyendo el class)
numeric_columns = [col for col in df.columns if col not in categorical_columns]

#eliminamos la columna class-label
numeric_columns = [c for c in numeric_columns if c not in ["class"]]
#print(numeric_columns)

# StringIndexer para las variables categóricas
stringindexer_stages = [StringIndexer(inputCol=c, outputCol='strindexed_' + c) for c in categorical_columns]

# los nombres de los indices de las categóricas para añadir a las features
stringindexer_columns = ['strindexed_' + c for c in categorical_columns]

#vector de stringIndexer, utilizando OneHotEncoder
onehotencoder_stages = [OneHotEncoder(inputCol='strindexed_' + c, outputCol='onehotencoder_' + c) for c in categorical_columns]

onehotencoder_columns = ['onehotencoder_' + c for c in categorical_columns]

# StringIndexer para la columna class-label
labelindexer_stages = [StringIndexer(inputCol='class', outputCol='label')]

# feature columns (numericas + categoricas_indexed) para el vectorAssembler de features
feature_columns = numeric_columns + onehotencoder_columns

vectorassembler_stage = VectorAssembler(inputCols=feature_columns, outputCol='features') 

# para el pipeline del dataframe inicial
all_stages = stringindexer_stages + onehotencoder_stages + [vectorassembler_stage] + labelindexer_stages

#tiempo preparación de los datos
timestart = datetime.datetime.now()

pipeline_data = Pipeline(stages=all_stages)

pipeline_model = pipeline_data.fit(df)

# transform the data
final_columns = ['features', 'label']

data_df = pipeline_model.transform(df).select(final_columns)

#fin medicion de tiempo de preparacion de datos
timeend = datetime.datetime.now()

timedelta = round((timeend-timestart).total_seconds(), 2) 
print "Time taken to prepare data: " + str(timedelta) + " seconds";

data_df.show(10)
##########################################
# FIN preparación de los datos
# fichero tipo label y features
##########################################

##########################################
# Generación del modelo
##########################################

timestart = datetime.datetime.now()

# kmeans
kmeans = KMeans().setK(nm_cluster).setSeed(1).setMaxIter(10)#.setFeaturesCol("features").setPredictionCol("prediction")
model = kmeans.fit(data_df)

predictions = model.transform(data_df)

timeend = datetime.datetime.now()

timedelta = round((timeend-timestart).total_seconds(), 2) 
print "Time taken to generate the model with kmeans: " + str(timedelta) + " seconds";

predictions.groupBy("prediction").count().show(40)

#número de predicciones total
total = float(predictions.count())

#predicciones erróneas
errorPredictions = predictions.filter(predictions['prediction'] != predictions['label']).count()
#predicciones correctas
correctPredictions = predictions.filter(predictions['prediction'] == predictions['label']).count()

#ratios de predicciones erróneas y correctas
ratioKO = errorPredictions / total
ratioOK = correctPredictions / total

print("Predictions total: " + str(total))
print("Correct predictions: " + str(correctPredictions))
print("Incorrect predictions: " + str(errorPredictions))
print("Ratio incorrect: " + str(ratioKO))
print("Ratio correct: " + str(ratioOK))
   
# Evaluate clustering by computing Within Set Sum of Squared Errors.
wssse = model.computeCost(data_df)
print("Within Set Sum of Squared Errors = " + str(wssse))

# Shows the result.
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)
    
timeendFIN = datetime.datetime.now()

timedelta = round((timeendFIN-timestartINI).total_seconds(), 2) 
print "Time taken to run the algorithm: " + str(timedelta) + " seconds";
