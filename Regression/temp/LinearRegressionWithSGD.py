from pyspark.mllib.regression import LinearRegressionWithSGD
import os
from numpy import array
from pyspark import SparkContext
import time
from pyspark.mllib.regression import LabeledPoint

os.environ["PYSPARK_PYTHON"] = "python3.6"

sc = SparkContext("spark://quickstart.cloudera:7077","app")

def f(x): print(x)

start = time.time()

# Load and parse the data
data = sc.textFile("hdfs://localhost/user/cloudera/Project/0722.csv")
parsedData = data.map(lambda x : x.split(',')).map(lambda x: (int(x[2]), float(x[6]),  float(x[8])))
print(parsedData.take(5))


Y_x = parsedData.map(lambda x : LabeledPoint(x[2],x[0:2]))
print(Y_x.take(5))



linear_model = LinearRegressionWithSGD.train(Y_x, iterations=1000, step=0.1, intercept =False)
# pre = linear_model.predict([1,26000.0])
# print(pre)

preData = Y_x.map(lambda x:(x.label, x.features, linear_model.predict(x.features)))
print(preData.take(5))


end = time.time()
elapsed = end - start
print("Time taken :{} seconds.".format(elapsed))