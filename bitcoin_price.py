from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_unixtime, col
import requests

# Parse arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--table', required=True)
parser.add_argument('--bucket', required=True)
parser.add_argument('--api', default="https://api.coinbase.com/api/v3/brokerage/market/products/BTC-USD/candles?granularity=ONE_MINUTE")

args = vars(parser.parse_args())
table_id = args['table']
temp_bucket = args['bucket']
api = args['api']

# Initialize Spark session
sc = SparkContext()
spark = SparkSession(sc)

# Fetch data from API
response = requests.get(api)
data = response.json()  # assuming JSON response

# Convert API data to Spark DataFrame
data_df = spark.createDataFrame(data['candles'])
data_df = data_df.limit(1) # select only 1 datapoint per request

# Perform data transformations
data_df = data_df.withColumn('timestamp', from_unixtime('start'))
data_df = data_df.drop('start')
# Convert column names to lower case
for coll in data_df.columns:
    if coll.lower()=="timestamp":
        continue
    else:
        data_df = data_df.withColumn(coll, col(coll).cast("double"))
    data_df = data_df.withColumnRenamed(coll, coll.lower())

data_df.show()

# Save DataFrame to BigQuery
data_df.write \
    .format("bigquery") \
    .option("table", table_id) \
    .option("temporaryGcsBucket", temp_bucket) \
    .mode("append") \
    .save()

print("Data stored in BigQuery!")
spark.stop()