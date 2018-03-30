library(SparkR)
library(magrittr)

sparkR.session()

#tiempo inicio del algoritmo
timestartGlobal <- Sys.time()

##########################################
# Load the data stored as SparkDataframe
##########################################
df <- read.df("FileStore/tables/kdd_training.txt", "csv", header = "true", inferSchema = "true")

##########################################
# Preparación de los datos
##########################################

#tiempo INI preparación de los datos
timestartDatos <- Sys.time()
df <- drop(df, "is_host_login")

#df_cluster <- summarize(groupBy(df, df$class), count = n(df$class))
#print(head(select(df_cluster, "class", "count"),num=40))

#solo dos clases
df = withColumn(df,"class", otherwise(when(df$class!="normal","attack"),df$class))

print(head(df))

#count classes y lo convierte a Rdataframe para operar con el, como número de clusters
nmcluster <- collect(select(df, countDistinct(df$class)))
#print(nmcluster)

splitDF_list <- randomSplit(df, c(0.7, 0.3), seed = 3)
df_training <- splitDF_list[[1]]
df_test <- splitDF_list[[2]]

print(count(df))
print(count(df_training))
print(count(df_test))

#print(head(select(summarize(groupBy(df_training, df_training$class), count = n(df_training$class)), "class", "count"),num=2))

#print(head(select(summarize(groupBy(df_test, df_test$class), count = n(df_test$class)), "class", "count"),num=2))

#tiempo FIN preparación de los datos
timeendDatos <- Sys.time()

#print(head(select(df_cluster,df_cluster$count, count = n(df_cluster$class))))

#tiempo INI generación del modelo
timestartModel <- Sys.time()

#entrenamos el modelo con todas las features
model <- spark.kmeans(df_training, class ~ ., k = nmcluster, maxIter = 10, initMode = "random")
print(summary(model))

#tiempo FIN generación del modelo
timeendModel <- Sys.time()

#tiempo INI generación de las predicciones/validacion del modelo
timestartTest <- Sys.time()

# fitted values on training data
fitted <- predict(model, df_test)
print(head(select(fitted, "class", "prediction"),num=10))

#sumdf <- summarize(groupBy(fitted, fitted$prediction,fitted$class), count_class = n(fitted$class))
#print(head(select(sumdf, "prediction","class","count_class"), num=100))

#tiempo FIN generación de las predicciones/validacion del modelo
timeendTest <- Sys.time()

# save fitted model to input path
#path <- "path/to/model"
#write.ml(model, path)

# can also read back the saved model and print
#savedModel <- read.ml(path)
#summary(savedModel)
#print(summary(model))

print(paste(c("Time taken to prepare the data",difftime(timeendDatos, timestartDatos, units = "secs")), collapse = " "))
print(paste(c("Time taken to generate the model",difftime(timeendModel, timestartModel, units = "secs")), collapse = " "))
print(paste(c("Time taken to test the model",difftime(timeendTest, timestartTest, units = "secs")), collapse = " "))
print(paste(c("Time taken to run the algorithm",difftime(Sys.time(), timestartGlobal, units = "secs")), collapse = " "))