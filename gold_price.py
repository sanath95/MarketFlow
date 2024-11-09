from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import lag, avg, concat, col, lit
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# Parse arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--table', required=True)
parser.add_argument('--data', required=True)
parser.add_argument('--bucket', required=True)
parser.add_argument('--target', default="High")
parser.add_argument('--train_start', default='2020.01.01')
parser.add_argument('--train_end', default='2023.12.31')
parser.add_argument('--test_start', default='2024.01.01')
parser.add_argument('--test_end', default='2024.09.20')

args = vars(parser.parse_args())
table_id = args['table']
data_path = args['data']
temp_bucket = args['bucket']
target = args['target']
train_start = args['train_start']
train_end = args['train_end']
test_start = args['test_start']
test_end = args['test_end']

# Initialize Spark session
sc = SparkContext()
spark = SparkSession(sc)

# Load historical data
historical_df = spark.read.format("csv").option("header", "true").load(data_path)
print('-'*30)
print('Total dataset size:')
print((historical_df.count(), len(historical_df.columns)))
print('-'*30)

# Create features to train model: Lag and moving average of target column
windowSpec = Window.orderBy(["Date", "Time"]).partitionBy('Date')
df = historical_df.withColumn("lag", lag(target, 1).over(windowSpec).cast("double")) \
                             .withColumn("moving_avg", avg(target).over(windowSpec.rowsBetween(-6, -1)))
# Drop NA rows
df = df.na.drop()
# Cast target column as double
df = df.withColumn(target, df[target].cast("double"))

# Split train and test data
train_data = df.filter(df.Date.between(*(train_start, train_end)))
test_data = df.filter(df.Date.between(*(test_start, test_end)))

# Print train and test data info
print('-'*30)
print(f"Train from {train_start} to {train_end}")
print(f"Test from {test_end} to {test_end}")
print(f'Train size: {train_data.count()}')
print(f'Test size: {test_data.count()}')
print('-'*30)

# Describe target column column
print('-'*30)
print('Train Data:')
train_data.describe([target]).show()
print('Test Data:')
test_data.describe([target]).show()
print('-'*30)

# Initialize VectorAssembler and StandardScaler
assembler = VectorAssembler(inputCols=["lag", "moving_avg"], outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

# Transform using VectorAssembler
train_data = assembler.transform(train_data)
test_data = assembler.transform(test_data)

# Fit StandardScaler model and transform the data
scaler_model = scaler.fit(train_data)
train_df = scaler_model.transform(train_data)
test_df = scaler_model.transform(test_data)

# Initialize LinearRegression and RandomForestRegressor models
lr = LinearRegression(regParam=0.2, featuresCol="scaled_features", labelCol=target)
rf = RandomForestRegressor(featuresCol="scaled_features", labelCol=target)

# Fit models on train data and transform test data
lr_model = lr.fit(train_df)
lr_predictions = lr_model.transform(test_df)
rf_model = rf.fit(train_df)
rf_predictions = rf_model.transform(test_df)

# Evaluate on test data and print RMSE, R² values
rmse_evaluator = RegressionEvaluator(labelCol=target, predictionCol="prediction", metricName="rmse")
r2_evaluator = RegressionEvaluator(labelCol=target, predictionCol="prediction", metricName="r2")
lr_rmse = rmse_evaluator.evaluate(lr_predictions)
lr_r2 = r2_evaluator.evaluate(lr_predictions)

print('-'*30)
print(f"Root Mean Squared Error (RMSE) for Linear Regression = {lr_rmse}")
print(f"R² for Linear Regression = {lr_r2}")
print('-'*30)
rf_rmse = rmse_evaluator.evaluate(rf_predictions)
rf_r2 = r2_evaluator.evaluate(rf_predictions)
print(f"Root Mean Squared Error (RMSE) for Random Forest Regressor = {rf_rmse}")
print(f"R² for Random Forest Regressor = {rf_r2}")
print('-'*30)

# Create Timestamp column
historical_df = historical_df.withColumn('Timestamp', concat(col("Date"), lit(" "), col("Time")))
historical_df = historical_df.drop('Date')
historical_df = historical_df.drop('Time')

# Write dataset into bigquery
historical_df.write.format("bigquery") \
    .option("table", table_id) \
    .option("temporaryGcsBucket", temp_bucket) \
    .mode("overwrite") \
    .save()

print("Data stored in BigQuery!")
spark.stop()