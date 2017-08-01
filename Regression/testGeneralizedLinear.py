from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.regression import LinearRegression
from pyspark import SparkContext, SparkConf
import time
from pyspark.mllib.regression import LabeledPoint
import os
from pyspark.sql import SQLContext
from pyspark.ml.linalg import Vectors

os.environ["PYSPARK_PYTHON"] = "python3.6"

sc = SparkContext("spark://quickstart.cloudera:7077","app")

start = time.time()

# Load training data
dataset = SQLContext(sc).read.format("libsvm")\
    .load("/home/cloudera/spark-2.1.1-bin-hadoop2.6/data/mllib/sample_linear_regression_data.txt")

glr = GeneralizedLinearRegression(family="gaussian", link="identity", maxIter=10, regParam=0.3)

# Fit the model
model = glr.fit(dataset)
print('dataset : ',dataset)
print('dataset.show() : ',dataset.show(3))
print("dataset.take(1) : ",dataset.take(1))

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