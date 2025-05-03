from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("TestImageLoad").getOrCreate()

df = spark.read.format("image").option("dropInvalid", True).load("data/train")
df.show(5, truncate=False)
