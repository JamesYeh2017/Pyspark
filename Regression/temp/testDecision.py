from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
import os
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.ml.linalg import Vectors

os.environ["PYSPARK_PYTHON"] = "python3.6"

sc = SparkContext("spark://quickstart.cloudera:7077","app")

# Load the data stored in LIBSVM format as a DataFrame.
data = sc.textFile("hdfs://localhost/user/cloudera/Project/0801.csv")
#data = SQLContext(sc).read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

# parsedData = data.map(lambda x : x.split(',')).map(lambda x: (int(x[2]), float(x[6]),  float(x[8])  ) )
parsedData = data.map(lambda x : x.split(',')).map(lambda x: (int(x[2]), float(x[10]), int(x[12]), float(x[13])))
newRDD = parsedData.map(lambda x:{'label':x[3], 'features': Vectors.sparse(3,{0:x[0],1:x[1],2:x[2]})})
sqlContext = SQLContext(sc)
newDF = sqlContext.createDataFrame(newRDD, ['features', 'label'])
print("newDF : ",newDF)
print("newDF.show(3) : ",newDF.show(3))
print("newDF.take(2) : ",newDF.take(2))

# Automatically identify categorical features, and index them.
# We specify maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(newDF)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = newDF.randomSplit([0.7, 0.3])

# Train a DecisionTree model.
dt = DecisionTreeRegressor(featuresCol="indexedFeatures")

# Chain indexer and tree in a Pipeline
pipeline = Pipeline(stages=[featureIndexer, dt])

# Train model.  This also runs the indexer.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("prediction", "label", "features").show(3)

# Select (prediction, true label) and compute test error
evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

treeModel = model.stages[1]
# summary only
print(treeModel)