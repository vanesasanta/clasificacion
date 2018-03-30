# Databricks notebook source
library(SparkR)

sparkR.session()

#tiempo inicio del algoritmo
timestartINIGlobal <- Sys.time()

##########################################
# Load the data stored as SparkDataframe
##########################################
df <- read.df("FileStore/tables/kdd_training.txt", "csv", header = "true", inferSchema = "true")

##########################################
# Preparación de los datos
##########################################

#tiempo INI preparación de los datos
timestartINIDatos <- Sys.time()

#normal -> 1.0, attack -> 0.0
df <- withColumn(df,"class", ifelse(df$class != "normal", 0.0, 1.0))
df <- drop(df, "is_host_login")
#drop categorical column
df <- drop(df, "service")

#convertir categoricas en numericas
df <- withColumn(df,"protocol_type", ifelse(df$protocol_type == "tcp", 0, ifelse(df$protocol_type == "udp", 1, 2)))
df <- withColumn(df,"flag", ifelse(df$flag == "SF", 0, ifelse(df$flag == "S0", 1, ifelse(df$flag == "S1", 2, ifelse(df$flag == "S2", 3, ifelse(df$flag == "S3", 4, ifelse(df$flag == "SH", 5, ifelse(df$flag == "OTH", 6, ifelse(df$flag == "REJ", 7, ifelse(df$flag == "RSTO", 8, ifelse(df$flag == "RSTOS0", 9, 10)))))))))))

#print(paste(c("distinct ", collect(distinct(select(df, 'protocol_type')))), collapse = " "))
#print(paste(c("distinct ", collect(distinct(select(df, 'service')))), collapse = " "))
#print(paste(c("distinct ", collect(distinct(select(df, 'flag')))), collapse = " "))
print(head(df))
#printSchema(df)
#print(count(df))

splitDF_list <- randomSplit(df, c(0.7, 0.3))
df_training <- splitDF_list[[1]]
df_test <- splitDF_list[[2]]

#tiempo FIN preparación de los datos
timestartFINDatos <- Sys.time()

#tiempo INI generación del modelo
timestartINIModel <- Sys.time()

#2 classes
#count classes y lo convierte a Rdataframe para operar con el, como número de clusters
nm_output <- collect(select(df, countDistinct(df$class)))
nm_input<-ncol(df) - 1 #eliminar la clase

print(paste(c("nm_output ", nm_output), collapse = " "))
print(paste(c("nm_input", nm_input), collapse = " "))

# capas definidas para la red neuronal:
# capa de entrada con nm_input entradas (1 por feature), 
# dos capas intermedias de 10 neuronas
# y una capa de salida con nm_output neuronas (1 por cada clase de salida)
layers = c(nm_input, 10, 10, nm_output)

# Fit a multilayer perceptron classification model with mlp
model <- spark.mlp(df_training, class ~ ., blockSize = 128, layers = layers, maxIter = 100)#, seed=1234

# Model summary
#print(summary(model))

#tiempo FIN generación del modelo
timestartFINModel <- Sys.time()

#tiempo INI generación de las predicciones/validacion del modelo
timestartINITest <- Sys.time()

# Prediction
predictions <- predict(model, df_test)

#printSchema(predictions)
print(head(predictions))

#tiempo FIN generación de las predicciones/validacion del modelo
timestartFINTest <- Sys.time()

#predicciones erróneas
errorPredictions = count(filter(predictions, predictions$prediction != predictions$class))

#predicciones correctas
correctPredictions = count(filter(predictions, predictions$prediction == predictions$class))

total <- count(predictions)
positive = count(filter(predictions, predictions$prediction == 1.0))
negative = count(filter(predictions, predictions$prediction == 0.0))

#ratios de predicciones erróneas y correctas
ratioKO <- errorPredictions / total
ratioOK <- correctPredictions / total

#calculo de true positive, true negative, false positive, false negative
tp = count(filter(filter(predictions, predictions$prediction == 1.0), predictions$class == 1.0))
tn = count(filter(filter(predictions, predictions$prediction == 0.0), predictions$class == 0.0))
fp = count(filter(filter(predictions, predictions$prediction == 1.0), predictions$class == 0.0))
fn = count(filter(filter(predictions, predictions$prediction == 0.0), predictions$class == 1.0))

tp_rate = tp/positive
tn_rate = tn/negative
fp_rate = fp/negative
fn_rate = fn/positive

print(paste(c("total dataset: ", count(df)), collapse = " "))
print(paste(c("total entrenamiento: ", count(df_training)), collapse = " "))
print(paste(c("total validacion: ", count(df_test)), collapse = " "))

print(paste(c("Predictions total: ", total), collapse = " "))
print(paste(c("Correct predictions: ", correctPredictions), collapse = " "))
print(paste(c("Incorrect predictions: ", errorPredictions), collapse = " "))
print(paste(c("Ratio incorrect: ", ratioKO), collapse = " "))
print(paste(c("Ratio correct: ", ratioOK), collapse = " "))

print(paste(c("TP: ",tp_rate), collapse = " "))
print(paste(c("TN: ",tn_rate), collapse = " "))
print(paste(c("FP: ",fp_rate), collapse = " "))
print(paste(c("FN: ",fn_rate), collapse = " "))

print(paste(c("Time taken to prepare the data",difftime(timestartFINDatos, timestartINIDatos, units = "secs")), collapse = " "))
print(paste(c("Time taken to generate the model",difftime(timestartFINModel, timestartINIModel, units = "secs")), collapse = " "))
print(paste(c("Time taken to test the model",difftime(timestartFINTest, timestartINITest, units = "secs")), collapse = " "))
print(paste(c("Time taken to run the algorithm",difftime(Sys.time(), timestartINIGlobal, units = "secs")), collapse = " "))
