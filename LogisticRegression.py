import os
from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark import SparkContext, SparkConf
import time
import numpy as np
from sklearn.linear_model import LinearRegression

os.environ["PYSPARK_PYTHON"] = "python3.6"

sc = SparkContext("spark://quickstart.cloudera:7077","app")

def f(x): print(x)

start = time.time()

# Load and parse the data
data = sc.textFile("hdfs://localhost/user/cloudera/Project/carReg.txt")
parsedData = data.map(lambda x : x.split(',')).map(lambda x: (float(x[2]), float(x[7]), float(x[15].split('%')[0])))
print(parsedData.take(5))


end = time.time()
elapsed = end - start
print("Time taken :{} seconds.".format(elapsed))



