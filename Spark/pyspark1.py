from pyspark.sql import SparkSession


class Spark:
    def __init__(self):
        from pyspark.sql import SparkSession
        self.spark = SparkSession \
            .builder \
            .appName("Dataframe") \
            .getOrCreate()


from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import DoubleType, StructField, LongType, StringType, TimestampType, StructType

if __name__ == '__main__':
    import warnings
    import pandas as pd
    #data = pd.read_csv('/Users/libaihe/Desktop/amh_data/tmp_dealcargo_price_feature_0615_0714.csv')
    #print(data.head())
    warnings.filterwarnings('ignore')
    from pyspark.sql import SparkSession
    spark = SparkSession.builder\
        .appName("demo3")\
        .config("spark.some.config.option", "some-value")\
        .getOrCreate()

    df = spark.createDataFrame(
        [(18862669710, '/未知类型', 'IM传文件', 'QQ接收文件', 39.0, '2018-03-08 21:45:45', 178111558222, 1781115582),
         (18862669710, '/未知类型', 'IM传文件', 'QQ接收文件', 39.0, '2018-03-08 21:45:45', 178111558222, 1781115582),
         (18862228190, '/移动终端', '移动终端应用', '移动腾讯视频', 292.0, '2018-03-08 21:45:45', 178111558212, 1781115582),
         (18862669710, '/未知类型', '访问网站', '搜索引擎', 52.0, '2018-03-08 21:45:46', 178111558222, 1781115582)],
        ('online_account', 'terminal_type', 'action_type', 'app', 'access_seconds', 'datetime', 'outid', 'class'))
    df.show()


    def compute(x):
        result = x[
            ['online_account', 'terminal_type', 'action_type', 'app', 'access_seconds', 'datetime', 'outid', 'class',
             'start_time', 'end_time']]
        return result


    schema = StructType([
        # StructField("index", DoubleType()),
        StructField("online_account", LongType()),
        StructField("terminal_type", StringType()),
        StructField("action_type", StringType()),
        StructField("app", StringType()),
        StructField("access_seconds", DoubleType()),
        StructField("datetime", StringType()),
        StructField("outid", LongType()),
        StructField("class", LongType()),
        StructField("end_time", TimestampType()),
        StructField("start_time", TimestampType()),
    ])


    @pandas_udf(schema, functionType=PandasUDFType.GROUPED_MAP)
    def g(df):
        print('ok')
        mid = df.groupby(['online_account']).apply(lambda x: compute(x))
        result = pd.DataFrame(mid)
        return result


    df = df.withColumn("end_time", df['datetime'].cast(TimestampType()))
    df = df.withColumn('end_time_convert_seconds', df['end_time'].cast('long').cast('int'))
    time_diff = df.end_time_convert_seconds - df.access_seconds
    df = df.withColumn('start_time', time_diff.cast('int').cast(TimestampType()))
    df = df.drop('end_time_convert_seconds')
    df.printSchema()
    aa = df.groupby(['online_account']).apply(g)
    aa.show()




