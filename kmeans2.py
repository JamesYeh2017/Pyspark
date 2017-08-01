import os
from pyspark import SparkContext
from pyspark.ml.clustering import KMeans
from numpy import array
from math import sqrt
from pyspark import SparkContext, SparkConf

import numpy as np
from pyspark.mllib.linalg import Vectors

spark_home = "/home/cloudera/spark-2.1.1-bin-hadoop2.6"
if not spark_home:
    raise ValueError('SPARK_HOME environment Variable is not set')

os.environ["PYSPARK_PYTHON"] = "python3.6"
# sys.path.insert(0, os.path.join(spark_home, '/usr/local/lib/python3.6/site-packages/pyspark/examples/src/main/python'))#/usr/lib64/cmf/agent/build/env/bin/python
# sys.path.insert(0, os.path.join(spark_home, '/usr/src/Python-3.6.1/python.tgz'))
sc = SparkContext("spark://quickstart.cloudera:7077","app")


dataset = sc.textFile("hdfs://localhost/user/cloudera/Project/kmeans_data.txt")
print(dataset.take(5))
kmeans = KMeans().setK(2).setSeed(1)
RDD = sc.parallelize(kmeans)
print(RDD)
model = kmeans.fit(RDD)

# Evaluate clustering by computing Within Set Sum of Squared Errors.
wssse = model.computeCost(dataset)
print("Within Set Sum of Squared Errors = " + str(wssse))

# Shows the result.
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)
# $example off$



