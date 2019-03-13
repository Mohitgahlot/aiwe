
m pyspark import sql
import math
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
spark = SparkSession.builder.appName("s").getOrCreate()

lines = spark.read.text("train.dat").rdd
parts = lines.map(lambda row: row.value.split("\t"))
parts.collect()
ratingsRDD = parts.map(lambda a: Row(userId=int(a[0]),movieId=int(a[1]), rating=float(a[2]), timestamp=int(a[3])))


ratings = spark.createDataFrame(ratingsRDD).cache()
als = ALS(maxIter=20, regParam=0.128, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="nan")
model = als.fit(ratings)
lines1 = spark.read.text("test.dat").rdd
parts1 = lines1.map(lambda row: row.value.split("\t"))
parts1.collect()
ratingsRDD1 = parts1.map(lambda a: Row(userId=int(a[0]),movieId=int(a[1])))
ratingsRDD1.collect()
ratings1 = spark.createDataFrame(ratingsRDD1).cache()
res = ratings1.withColumn("id", monotonically_increasing_id())
predictions = model.transform(res)
ta = predictions.alias("ta")
tb = res.alias("tb")
inner = tb.join(ta,["id"]).select(ta.id,ta.movieId,ta.userId,ta.prediction)
inner = inner.sort(inner.id)
a = inner.select("prediction").rdd.flatMap(list).take(2154)
fa = []
for a1 in a:
    if math.isnan(a1):
        fa.append(4)
    else:
        fa.append(int(round(a1, 0)))

f = open("assignments/homework.txt","w")
for f1 in fa:
    f.write(str(f1)+"\n")
