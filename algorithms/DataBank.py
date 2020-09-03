import pandas as pd
from pyspark.sql import SparkSession


class DataBank:
    def __init__(self):
        self.data = None

    def read_csv(self, path):
        self.data = pd.read_csv(path)

    def read_parquet(self, path):
        self.data = pd.read_parquet(path)

    def spark_sql(self, sql):
        spark = SparkSession \
            .builder \
            .config("spark.driver.memory", "4g") \
            .appName("Python spark") \
            .enableHiveSupport() \
            .getOrCreate()

        self.data = spark.sql(sqlQuery=sql).toPandas()

    def spark_read_csv(self, path):
        spark = SparkSession \
            .builder \
            .config("spark.driver.memory", "4g") \
            .appName("Python spark") \
            .enableHiveSupport() \
            .getOrCreate()
        self.data = spark.read.csv(path).toPandas()

    def spark_read_parquet(self, path):
        spark = SparkSession \
            .builder \
            .config("spark.driver.memory", "4g") \
            .appName("Python spark") \
            .enableHiveSupport() \
            .getOrCreate()
        self.data = spark.read.parquet(path).toPandas()
