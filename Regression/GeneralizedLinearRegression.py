from pyspark.ml.regression import GeneralizedLinearRegression
import os
from pyspark import SparkContext, SparkConf
import time
from pyspark.sql import SQLContext
from pyspark.ml.linalg import Vectors

os.environ["PYSPARK_PYTHON"] = "python3.6"

sc = SparkContext("spark://quickstart.cloudera:7077","app")

start = time.time()

# Load training data
data = SQLContext(sc).read.format("libsvm")\
    .load("/home/cloudera/spark-2.1.1-bin-hadoop2.6/data/mllib/sample_linear_regression_data.txt")

print('data : ',data)
print('data.show() : ',data.show(3))
print(data.take(1))

#glr model
glr = GeneralizedLinearRegression(family="gamma", link="power", maxIter=10, regParam=0.3)
                #family="gaussian", link="identity"
# Fit the model
model = glr.fit(data)

# Print the coefficients and intercept for generalized linear regression model
print("Coefficients: " + str(model.coefficients))
print("Intercept: " + str(model.intercept))

# Summarize the model over the training set and print out some metrics
summary = model.summary
print("Coefficient Standard Errors: " + str(summary.coefficientStandardErrors))
print("T Values: " + str(summary.tValues))
print("P Values: " + str(summary.pValues))
print("Dispersion: " + str(summary.dispersion))
print("Null Deviance: " + str(summary.nullDeviance))
print("Residual Degree Of Freedom Null: " + str(summary.residualDegreeOfFreedomNull))
print("Deviance: " + str(summary.deviance))
print("Residual Degree Of Freedom: " + str(summary.residualDegreeOfFreedom))
print("AIC: " + str(summary.aic))
print("Deviance Residuals: ")
summary.residuals().show()