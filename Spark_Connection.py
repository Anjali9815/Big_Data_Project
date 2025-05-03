from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("TestImageLoad").getOrCreate()

train_path = "/Users/anjalijha/Desktop/UMBC/Semester_2/DS602/data/Train/"
train_data = spark.read.format("image").option("dropInvalid", True).load(train_path)
train_data.show(5, truncate=False)
