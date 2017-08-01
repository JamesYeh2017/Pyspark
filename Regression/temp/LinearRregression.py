from pyspark.ml.regression import LinearRegression
from pyspark import SparkContext
from pyspark.sql import SQLContext
# Load training data
sc = SparkContext("spark://quickstart.cloudera:7077","app")
training = SQLContext(sc).read.format("libsvm")\
     .load("/home/cloudera/spark-2.1.1-bin-hadoop2.6/data/mllib/sample_linear_regression_data.txt")

lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
print(training)
print(training.show(2))
print(training.take(2))
# Fit the model
lrModel = lr.fit(training)

# Print the coefficients and intercept for linear regression
print("Coefficients: %s" % str(lrModel.coefficients))
print("Intercept: %s" % str(lrModel.intercept))

# Summarize the model over the training set and print out some metrics
trainingSummary = lrModel.summary
print("numIterations: %d" % trainingSummary.totalIterations)
print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
trainingSummary.residuals.show()
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)
