from pyspark.mllib.regression import LinearRegressionWithSGD
from pyspark.mllib.tree import DecisionTree
import os
from pyspark import SparkContext, SparkConf
import time
from pyspark.mllib.regression import LabeledPoint
import numpy as np

os.environ["PYSPARK_PYTHON"] = "python3.6"

sc = SparkContext("spark://quickstart.cloudera:7077","app")

def f(x): print(x)

start = time.time()

# Load and parse the data
data = sc.textFile("hdfs://localhost/user/cloudera/Project/0801.csv")
parsedData = data.map(lambda x : x.split(',')).map(lambda x: (int(x[2]), float(x[10]), int(x[12]), float(x[13])))
print("parsedData : ",parsedData.take(2))


Y_x = parsedData.map(lambda x : LabeledPoint(x[3],x[0:3]))
print("Y_x : ",Y_x.take(2))


print("================================================")
# LinearRegressionWithSGD
linear_model = LinearRegressionWithSGD.train(Y_x, iterations=2, step=0.1, intercept =False)
# pre = linear_model.predict([1,26000.0])
# print(pre)

                                     # (label , predict)
label_vs_predcited = Y_x.map(lambda p:(p.label,linear_model.predict(p.features)))
print('Linear Model predictions: ', str(label_vs_predcited.take(2)) )
#
# preData = Y_x.map(lambda p:(p.label, p.features, linear_model.predict(p.features)))
# print("preData : ",preData.take(2))

print("================================================")
#DecisionTree
(trainingData, testData) = Y_x.randomSplit([0.7, 0.3])

dt_model = DecisionTree.trainRegressor(trainingData,{})
#1.(trainingData, categoricalFeaturesInfo={},impurity='variance', maxDepth=5, maxBins=32)
preds = dt_model.predict(testData.map(lambda p: p.features))
actual = testData.map(lambda p:p.label)
true_vs_predicted_dt = actual.zip(preds)

testMSE = true_vs_predicted_dt.map(lambda lp: (lp[0] - lp[1]) * (lp[0] - lp[1])).sum() /\
    float(testData.count())

print('Decision Tree labelsAndPredictions: '+str(true_vs_predicted_dt.take(100)))
print('Decision Tree depth: '+ str(dt_model.depth()))
print('Decision Tree number of nodes: '+str(dt_model.numNodes()))
print('Test Mean Squared Error = ' + str(testMSE))
# print('Learned regression tree model:')
# print(dt_model.toDebugString())

print("================================================")
#Linear Regression
def squared_error(actual, pred):
   return (pred-actual)*2
def abs_error(actual, pred):
   return np.abs(pred-actual)
def squared_log_error(pred, actual):
   return (np.log(pred+1)-np.log(actual+1))*2

mse = label_vs_predcited.map(lambda t: squared_error(t[0],t[1])).mean()
mae = label_vs_predcited.map(lambda t: abs_error(t[0],t[1])).mean()
rmsle = np.sqrt(label_vs_predcited.map(lambda t: squared_log_error(t[0],t[1])).mean())
print("Linear Model - Mean Squared Error: %2.4f"% mse)
print("Linear Model - Mean Absolute Error: %2.4f"% mae)
print("Linear Model - Root Mean Squared Log Error: %2.4f"% rmsle)

print("================================================")


end = time.time()
elapsed = end - start
print("Time taken :{} seconds.".format(elapsed))
