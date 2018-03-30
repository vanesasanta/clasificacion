library(SparkR)

sparkR.session()

#tiempo inicio del algoritmo
timestartINIGlobal <- Sys.time()

##########################################
# Load the data stored as SparkDataframe
##########################################
df <- read.df("FileStore/tables/kdd_training.txt", "csv", header = "true", inferSchema = "true")
df <- drop(df, "is_host_login")
print(count(df))

##########################################
# Preparación de los datos
##########################################

#tiempo INI preparación de los datos
timestartINIDatos <- Sys.time()

df <- withColumn(df,"class", ifelse(df$class != "normal", "attack", df$class))

#print(head(filter(df, df$class != "normal")))

splitDF_list <- randomSplit(df, c(0.7, 0.3), seed = 1)
df_training <- splitDF_list[[1]]
df_test <- splitDF_list[[2]]

#print(count(df_training))

#print(count(df_test))

#tiempo FIN preparación de los datos
timestartFINDatos <- Sys.time()

#tiempo INI generación del modelo
timestartINIModel <- Sys.time()

# Fit a random forest classification model with spark.randomForest
model <- spark.randomForest(df_training, class ~ ., "classification", maxDepth = 6, maxBins = 68, numTrees = 5)

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
#ratios de predicciones erróneas y correctas
ratioKO <- errorPredictions / total
ratioOK <- correctPredictions / total

#calculo de true positive, true negative, false positive, false negative
tp = count(filter(filter(predictions, predictions$prediction == "normal"), predictions$class == "normal"))/total
tn = count(filter(filter(predictions, predictions$prediction == "attack"), predictions$class == "attack"))/total

fp = count(filter(filter(predictions, predictions$prediction == "normal"), predictions$class == "attack"))/total
fn = count(filter(filter(predictions, predictions$prediction == "attack"), predictions$class == "normal"))/total

print(paste(c("Predictions total: ", total), collapse = " "))
print(class(correctPredictions))
print(paste(c("Correct predictions: ", correctPredictions), collapse = " "))
print(paste(c("Incorrect predictions: ", errorPredictions), collapse = " "))
print(paste(c("Ratio incorrect: ", ratioKO), collapse = " "))
print(paste(c("Ratio correct: ", ratioOK), collapse = " "))

print(paste(c("TP: ",tp), collapse = " "))
print(paste(c("TN: ",tn), collapse = " "))
print(paste(c("FP: ",fp), collapse = " "))
print(paste(c("FN: ",fn), collapse = " "))

print(paste(c("Time taken to prepare the data",difftime(timestartFINDatos, timestartINIDatos, units = "secs")), collapse = " "))
print(paste(c("Time taken to generate the model",difftime(timestartFINModel, timestartINIModel, units = "secs")), collapse = " "))
print(paste(c("Time taken to test the model",difftime(timestartFINTest, timestartINITest, units = "secs")), collapse = " "))
print(paste(c("Time taken to run the algorithm",difftime(Sys.time(), timestartINIGlobal, units = "secs")), collapse = " "))