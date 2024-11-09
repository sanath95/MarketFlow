# MarketFlow

## Introduction

This project is focused on developing a robust ETL (Extract, Transform, Load) pipeline on Google Cloud Platform (GCP) to process and analyze financial data from multiple sources. The pipeline integrates real-time Bitcoin price data from the Coinbase API and a comprehensive historical dataset of gold prices sourced from Kaggle. PySpark is utilized to extract and transform this data, perform data cleansing, and build regression-based predictive models for enhanced analytical insights. The transformed data is then stored in a BigQuery data warehouse for scalable and efficient access. Additionally, dbt (data build tool) is employed to carry out further transformations on the processed data within BigQuery, optimizing it for downstream analysis and reporting. This ETL pipeline aims to streamline financial data management, support predictive modeling, and enable data-driven decision-making.

**Motivation:**

The choice of Bitcoin and gold price data is driven by the high volatility of financial markets, which generates a massive volume of data in real time. This volatility demands a system capable of processing and analyzing big data at scale, providing timely insights to support decision-making. By designing the pipeline to handle large, streaming datasets effectively, this project ensures that real-time and historical data is efficiently processed, enabling rapid analysis and predictive modeling in a fast-paced market environment.

**Data Sources:**

1. Static data: [XAU/USD Gold Price Historical Data (2004-2024)](https://www.kaggle.com/datasets/novandraanugrah/xauusd-gold-price-historical-data-2004-2024?resource=download)
2. Live data: [Coinbase: Get product candles](https://docs.cdp.coinbase.com/exchange/reference/exchangerestapi_getproductcandles)
    - [API](https://api.coinbase.com/api/v3/brokerage/market/products/BTC-USD/candles?granularity=ONE_MINUTE) for real-time data.

---

## Architecture

![architecture](./assets/marketflow_architecture.png)

The pipeline architecture for processing Bitcoin and gold price data on **Google Cloud Platform** consists of several components, as shown in the image:

1. **ETL using PySpark and BigQuery**: A Google Cloud Dataproc cluster is set up to run PySpark jobs. This cluster performs two main tasks:
   - It extracts historical gold price data from a Google Cloud Storage bucket, then calculates features like lag and moving averages, which are used to train a regression model. The transformed data is then stored in BigQuery for further analysis.
   - It calls the Coinbase API at regular intervals to gather real-time Bitcoin price data, which is also stored in BigQuery.

2. **Automation of real-time data collection**: Real-time data collection from the Coinbase API is automated using a combination of Cloud Scheduler, Pub/Sub, and Cloud Functions:
   - **Cloud Scheduler** triggers the process at defined intervals, sending a message to Pub/Sub.
   - **Pub/Sub** forwards this message to Cloud Functions.
   - **Cloud Functions** then initiates the Dataproc job to fetch and process the latest Bitcoin data, ensuring that the data collection happens without manual intervention.

3. **DBT for Data Transformation**: DBT (data build tool) is used for additional transformations within BigQuery. After the initial processing and storage in BigQuery, dbt refines the data further, applying business logic and creating analytics-ready datasets.

This architecture enables scalable, real-time data processing, storage, and transformation, ensuring that Bitcoin and gold price data is available for analysis in BigQuery with minimal latency.

---

## Execution

1. **Create Project**
    
    Create a new project on Google Cloud Platform and link it to a billing account.

2. **Enable APIs**

    Enable the following APIs.
    - Cloud Dataproc API
    - BigQuery API
    - Cloud Resource Manager API
    - Cloud Scheduler API
    - Cloud Pub/Sub API
    - Cloud Build API
    - Cloud Functions API
    - Cloud Run Admin API
    - Eventarc API

3. **Add Roles**

    Navigate to "IAM" and add roles to the service account. 
    - Click on "Edit principal"(pencil icon) next to the "Compute Engine default service account". Add "Storage Admin" role.
    - Click on "Edit principal"(pencil icon) next to the "App Engine default service account". Add "Service Account Token Creator" role.

4. **Create Dataproc Cluster**

    Navigate to "Cloud Dataproc" and create a cluster for running PySpark jobs.
    - Select "Cluster on Compute Engine".
    - Give the cluster a name. 
    - Select "europe-west3" region.
    - Select "Cluster type" as "Single Node".
    - Go to "Customize cluster" tab. Uncheck "Internal IP only".
    - Click on create.

5. **Create Bucket**

    Navigate to "Buckets" and create a bucket to store code, and data files.
    - Give the bucket a name and click "Continue".
    - Select region "europe-west3" and click "Continue".
    - Under "Data Proctection", unckeck "Soft delete policy" and click "Create".

6. **File Storage**

    Store the following files in the bucket:
    - `bitcoin_price.py`
    - `gold_price.py`
    - `XAU_15m_data_2004_to_2024-20-09.csv`

7. **Create BigQuery dataset and tables**

    Navigate to "BigQuery" and create a dataset and tables to store in the data warehouse.
    - Click on the kebab menu button next to the project id and select "Create dataset".
    - Give the "Dataset ID".
    - Select region "europe-west3" and click "Create dataset".
    - Click on the kebab menu button next to the dataset id and select "Create table".
    - Give the table a name for gold prices.
    - Create another table for bitcoin prices.

8. **ETL for historical data**

    Navigate to "Cloud Dataproc" > "Jobs" and submit a job to extract, transform and load historial gold prices data.
    - Select region "europe-west3"
    - Select cluster created in step 4.
    - Select "Job type" as "PySpark".
    - Enter "Main python file" as `gs://{bucket-name}/gold_price.py`.
    - Give the following arguments:
        - `--table={table-id}`
        - `--data=gs://{bucket-name}/XAU_15m_data_2004_to_2024-20-09.csv`
        - `--bucket={bucket-name}`
        > Table ID is usually of the form: `{project-id}.{dataset-name}.{table-name}`
    - Submit job.

    `gold_price.py` accepts the following arguments:
    |Argument|Description|Deafult value|Required|
    |---|---|---|---|
    |--table|BigQuery Table ID to store the Gold Price data||&#x2713;|
    |--data|gsutil URI of the datset stored in the bucket||&#x2713;|
    |--bucket|Temp storage bucket name when writing to BigQuery||&#x2713;|
    |--target|Target feature to train the model on|High|&#x2718;|
    |--train_start|Start date for the train data in yyyy.mm.dd format|2020.01.01|&#x2718;|
    |--train_end|End date for the train data in yyyy.mm.dd format|2023.12.31|&#x2718;|
    |--test_start|Start date for the test data in yyyy.mm.dd format|2024.01.01|&#x2718;|
    |--test_end|End date for the test data in yyyy.mm.dd format|2024.09.20|&#x2718;|

9. **ETL for streaming data**

    Navigate to "Pub/Sub" and create a topic for a publish-subscribe messaging service.
    - Give "Topic ID" and click "Create".

    Navigate to "Cloud Run Functions" to create a function that subscribes to messages and runs a PySpark job to extract, transform and load live bitcoin prices data.
    - Give the function a name.
    - Select "europe-west3" region.
    - Select "Trigger type" as "Cloud Pub/Sub".
    - Select the "Cloud Pub/Sub Topic" that was created and click "Next".
    - For "Runtime" select "Python 3.11".
    - For "Entry point" write "trigger_dataproc".
    - Replace the contents in "main.py" and give the following variables: `project_id`, `region`, `cluster_name`, `script_uri`.
        ```python
        from google.cloud import dataproc_v1 as dataproc
    
        def trigger_dataproc(event, context):
            project_id = "" # GCP Project ID
            region = "europe-west3" # Region
            cluster_name = "" # Name of the Dataproc Cluster
            script_uri = ""  # Path to your PySpark script: gs://{bucket-name}/bitcoin_price.py
            table_id = "" # BigQuery Table ID to store the Bitcoin Price data
            bucket = "" # Temp storage bucket name when writing to BigQuery

        
            # Initialize Dataproc client
            job_client = dataproc.JobControllerClient(client_options={"api_endpoint": f"{region}-dataproc.googleapis.com:443"})
        
            # Set up the PySpark job
            job = {
                "placement": {"cluster_name": cluster_name},
                "pyspark_job": {
                    "main_python_file_uri": script_uri, 
                    "args": [f"--table={table_id}", f"--bucket={bucket}"],}
            }
        
            # Submit the job to Dataproc
            operation = job_client.submit_job_as_operation(
                request={"project_id": project_id, "region": region, "job": job}
            )
            response = operation.result()
            print(f"Submitted job ID {job_id}")
        ```
    - Replace the contents in "requirements.txt"
        ```
        google-cloud-dataproc==5.15.1
        ```
    - Click "Deploy".

    Navigate to "Cloud Scheduler" and schedule a job to send a publish a message.
    - Give the job a name.
    - Select region "europe-west3".
    - Set the frequency. Ex: `* * * * *` to schedule every minute.
    - Select a "Timezone" and click "Continue".
    - Select "Target Type" as "Pub/Sub".
    - Select the "Cloud Pub/Sub Topic" that was created.
    - Give a message and click "Create".
    - Select the created job and click on "Force Run".

10. **Data Transformation using DBT**

    TODO

---

## Results

1. Data can be seen in BigQuery

    Gold Prices Data

    ![gold sample](./assets/gold_price_sample.png)

    Bitcoin Prices Data

    ![bitcoin sample](./assets/bitcoin_price_sample.png)

2. Regression models on historical data:
    |Model| RMSE | R² |
    |---|---|---|
    |Linear Regression|2.42|0.99|
    |Random Forest Regressor|296.72|-2.05|
    Linear Regression maybe overfit whereas the Random Forest Regressor is very underfit.

---

## Future Scope

- Get both historical data and live data for same asset with same time intervals.
- Use the trained model to forecast and compare with live values.

---

## References

1. [XAU/USD Gold Price Historical Data (2004-2024)](https://www.kaggle.com/datasets/novandraanugrah/xauusd-gold-price-historical-data-2004-2024?resource=download)
2. [Coinbase: Get product candles](https://docs.cdp.coinbase.com/exchange/reference/exchangerestapi_getproductcandles)
3. [PySpark](https://spark.apache.org/docs/latest/api/python/index.html#:~:text=PySpark%20is%20the%20Python%20API,for%20interactively%20analyzing%20your%20data.)
4. [Use the BigQuery connector with Spark](https://cloud.google.com/dataproc/docs/tutorials/bigquery-connector-spark-example)
5. [ChatGPT](https://chatgpt.com/)

---

> **Remember to stop and delete all resources after completion.**

---
