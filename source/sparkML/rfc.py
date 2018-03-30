import sys
import os
from pyspark import SparkContext, SQLContext
from pyspark.ml import Pipeline
#from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import DecisionTreeClassifier

from pyspark.ml.feature import StringIndexer, OneHotEncoder,VectorIndexer, VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator

import datetime

from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors, VectorUDT


if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("DecisionTreeClassification")\
        .getOrCreate()

#tiempo preparación de los datos
timestartGlobal = datetime.datetime.now()

# Load the data stored as dataframe
df = spark.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("/FileStore/tables/kdd_training.txt")

df.cache()

##########################################
#preparación de los datos
# fichero tipo label y features
##########################################
from pyspark.sql import functions as F
#solo 2 clases, normal y attack (todos los que no son normal)
df = df.withColumn("class", F.when(df["class"]!="normal","attack").otherwise(df["class"]))

#df.printSchema()
#df.show(15)
#print("dataset count " + str(df.count()))

#tiempo INI preparación de los datos
timestartDatos = datetime.datetime.now()

#categoricas
categorical_columns = ["protocol_type","service","flag","land","logged_in","is_guest_login"]

#numericas = todas las columnas menos las categoricas (incluyendo el class)
numeric_columns = [col for col in df.columns if col not in categorical_columns]

#eliminamos la columna class-label
numeric_columns = [c for c in numeric_columns if c not in ["class","is_host_login"]]
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

#print(feature_columns)

vectorassembler_stage = VectorAssembler(inputCols=feature_columns, outputCol='features') 

# para el pipeline del dataframe inicial
all_stages = stringindexer_stages + onehotencoder_stages + [vectorassembler_stage] + labelindexer_stages

pipeline_data = Pipeline(stages=all_stages)

pipeline_model = pipeline_data.fit(df)

# transform the data
final_columns = ['features', 'label']

data_df = pipeline_model.transform(df).select(final_columns)

#fin medicion de tiempo de preparacion de datos
timeendDatos = datetime.datetime.now()
 
data_df.show(10)
##########################################
# FIN preparación de los datos
# fichero tipo label y features
##########################################

##########################################
# Generación del modelo
##########################################

#tiempo generación del modelo
timestartModel = datetime.datetime.now()

labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data_df)

# Indexación de las variables categóricas como numéricas a partir de 4 valores
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data_df)

# Dividir el dataset en datos de entrenamiento (70%) y datos para validacion del modelo (30%)
(trainingData, testData) = data_df.randomSplit([0.7, 0.3])

trainingData.cache()
testData.cache()

print(data_df.count())
print(trainingData.count())
print(testData.count())

# Entrenar un RandomForestClassifier model.
#dt = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=5)
dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")

# parameter grid
from pyspark.ml.tuning import ParamGridBuilder
    
# feature 35 tiene 65 valores diferentes, por defecto 32,se amplia maxBins (número de hojas) a minimo 65
param_grid = ParamGridBuilder().addGrid(dt.maxBins, [65, 68, 71]).addGrid(dt.maxDepth, [4, 6, 8]).build()

# evaluator binario
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", metricName="areaUnderROC")

# build cross-validation model, 4 iteracciones
from pyspark.ml.tuning import CrossValidator
cv = CrossValidator(estimator=dt, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=4)

# construccion del Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, cv])

# entrenar el pipeline, con los datos de entrenamiento el estimador genera el modelo.
model = pipeline.fit(trainingData)

#tiempo generación del modelo
timeendModel = datetime.datetime.now()

# Generamos el dataframe con las predicciones a partir de los datos de test y del modelo anterior

#tiempo validación del modelo
timestartTest = datetime.datetime.now()

#predictions = model.transform(trainingData)
predictions = model.transform(testData)

#tiempo validación del modelo
timeendTest = datetime.datetime.now()

# Select de las columnas a mostrar
predictions.select("prediction", "indexedLabel", "features").show(5)

#exactitud
accuracy = evaluator.evaluate(predictions)

#número de predicciones total
total = float(predictions.count())

#predicciones erróneas
errorPredictions = predictions.filter(predictions['prediction'] != predictions['indexedLabel']).count()
#predicciones correctas
correctPredictions = predictions.filter(predictions['prediction'] == predictions['indexedLabel']).count()

#ratios de predicciones erróneas y correctas
ratioKO = errorPredictions / total
ratioOK = correctPredictions / total

#calculo de true positive, true negative, false positive, false negative
tp = (predictions.filter(predictions['prediction'] == 1.0).filter(predictions['indexedLabel'] == 1.0).count())/total
tn = (predictions.filter(predictions['prediction'] == 0.0).filter(predictions['indexedLabel'] == 0.0).count())/total
fp = predictions.filter(predictions['prediction'] == 1.0).filter(predictions['indexedLabel'] == 0.0).count()/total
fn = predictions.filter(predictions['prediction'] == 0.0).filter(predictions['indexedLabel'] == 1.0).count()/total

print("Exactitud (accuracy) = %g " % accuracy)
print("Error rate = %g " % (1.0 - accuracy))
print("Predictions total: " + str(total))
print("Correct predictions: " + str(correctPredictions))
print("Incorrect predictions: " + str(errorPredictions))
print("Ratio incorrect: " + str(ratioKO))
print("Ratio correct: " + str(ratioOK))
print("TP: " + str(tp))
print("TN: " + str(tn))
print("FP: " + str(fp))
print("FN: " + str(fn))

timeendGlobal = datetime.datetime.now()

print "Time taken to prepare data: " + str(round((timeendDatos-timestartDatos).total_seconds(), 2)) + " seconds";
print "Time taken to generate the model: " + str(round((timeendModel-timestartModel).total_seconds(), 2)) + " seconds";
print "Time taken to validate the model: " + str(round((timeendTest-timestartTest).total_seconds(), 2)) + " seconds";
print "Time taken to run the algorithm: " + str(round((timeendGlobal-timestartGlobal).total_seconds(), 2)) + " seconds";