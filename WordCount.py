import os
import sys
from pyspark import SparkContext, SparkConf

spark_home = "/home/cloudera/spark-2.1.1-bin-hadoop2.6"
if not spark_home:
    raise ValueError('SPARK_HOME environment Variable is not set')

os.environ["PYSPARK_PYTHON"] = "python3.6"
# sys.path.insert(0, os.path.join(spark_home, '/usr/local/lib/python3.6/site-packages/pyspark/examples/src/main/python'))#/usr/lib64/cmf/agent/build/env/bin/python
# sys.path.insert(0, os.path.join(spark_home, '/usr/src/Python-3.6.1/python.tgz'))
sc = SparkContext("spark://quickstart.cloudera:7077","app")

lines = sc.textFile("hdfs://localhost/user/cloudera/spark101/wordcount/book")

def parse_data(textRDD):
    words = lines.flatMap(lambda x : x.split(" "))

    wordCounts = words.map(lambda x : (x, 1))
    wordCounts.reduceByKey(lambda x,y : x + y)
    return (wordCounts.collect())

WC = parse_data(lines)#.foreach(print)
for idx in WC:
    print(idx)


