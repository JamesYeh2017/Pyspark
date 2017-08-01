from pyspark.mllib.clustering import KMeans, KMeansModel
import os
from numpy import array
from pyspark import SparkContext, SparkConf
from math import sqrt

spark_home = "/home/cloudera/spark-2.1.1-bin-hadoop2.6"
if not spark_home:
    raise ValueError('SPARK_HOME environment Variable is not set')

os.environ["PYSPARK_PYTHON"] = "python3.6"
# sys.path.insert(0, os.path.join(spark_home, '/usr/local/lib/python3.6/site-packages/pyspark/examples/src/main/python'))#/usr/lib64/cmf/agent/build/env/bin/python
# sys.path.insert(0, os.path.join(spark_home, '/usr/src/Python-3.6.1/python.tgz'))
sc = SparkContext("spark://quickstart.cloudera:7077","app")

def f(x): print(x)

# data = sc.parallelize(array([0.0,0.0, 1.0,1.0, 9.0,8.0, 8.0,9.0]).reshape(4, 2))
# aa = sc.parallelize([0.0,0.0, 1.0,1.0, 9.0,8.0, 8.0,9.0]).take(2)
# print(aa)

# Load and parse the data
data = sc.textFile("hdfs://localhost/user/cloudera/Project/kmeans_data.txt")

parsedData = data.map(lambda line: array([float(x) for x in line.split(' ') if x != ' ']))
print(parsedData.take(5))

# Build the model (cluster the data)
clusters = KMeans.train(parsedData, 2, maxIterations=10, initializationMode="random")


# Evaluate clustering by computing Within Set Sum of Squared Errors
def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))

WSSSE = parsedData.map(lambda point: error(point)).reduce(lambda x, y: x + y)
print("Within Set Sum of Squared Error = " + str(WSSSE))

# Save and load model
# clusters.save(sc, "target/org/apache/spark/PythonKMeansExample/KMeansModel")
# sameModel = KMeansModel.load(sc, "target/org/apache/spark/PythonKMeansExample/KMeansModel")

