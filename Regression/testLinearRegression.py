from pyspark.ml.regression import LinearRegression
from pyspark import SparkContext, SparkConf
import time
import os
from pyspark.sql import SQLContext
from pyspark.ml.linalg import Vectors

start = time.time()
PATH = "hdfs://localhost/user/cloudera/Project/0801.csv"

os.environ["PYSPARK_PYTHON"] = "python3.6"

#create sparkContext
sc = SparkContext("spark://quickstart.cloudera:7077","app")


# Load and parse the data
data = sc.textFile(PATH)
# parsedData = data.map(lambda x : x.split(',')).map(lambda x: (int(x[2]), float(x[6]), float(x[8])))
parsedData = data.map(lambda x : x.split(',')).map(lambda x: (int(x[2]), float(x[10]), int(x[12]), float(x[13])))
print("parsedData : ",parsedData.take(2))


sqlContext = SQLContext(sc)
newRDD = parsedData.map(lambda x:{'label':x[3], 'features': Vectors.sparse(3,{0:x[0],1:x[1],2:x[2]})})
newDF = sqlContext.createDataFrame(newRDD, ['features', 'label'])
print('newDF : ',newDF)

#LinearRegression model
lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Fit the model
lrModel = lr.fit(newDF)

# Print the coefficients and intercept for linear regression
print("Coefficients: %s" % str(lrModel.coefficients))
print("Intercept: %s" % str(lrModel.intercept))

# Summarize the model over the training set and print out some metrics
trainingSummary = lrModel.summary
print("numIterations: %d" % trainingSummary.totalIterations)
print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
trainingSummary.residuals.show(5)
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)

end = time.time()
elapsed = end - start
# print("Time taken :{} seconds.".format(elapsed))
