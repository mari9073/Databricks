# Databricks notebook source
# MAGIC %md
# MAGIC # Data Exploration
# MAGIC This notebook performs exploratory data analysis on the dataset.
# MAGIC To expand on the analysis, attach this notebook to the **MLCluster1** cluster,
# MAGIC edit [the options of pandas-profiling](https://pandas-profiling.github.io/pandas-profiling/docs/master/rtd/pages/advanced_usage.html), and rerun it.
# MAGIC - Explore completed trials in the [MLflow experiment](#mlflow/experiments/1829960369523491/s?orderByKey=metrics.%60val_f1_score%60&orderByAsc=false)
# MAGIC - Navigate to the parent notebook [here](#notebook/1829960369523492) (If you launched the AutoML experiment using the Experiments UI, this link isn't very useful.)
# MAGIC 
# MAGIC Runtime Version: _10.4.x-cpu-ml-scala2.12_

# COMMAND ----------

import os
import uuid
import shutil
import pandas as pd
import databricks.automl_runtime

from mlflow.tracking import MlflowClient

# Download input data from mlflow into a pandas DataFrame
# Create temporary directory to download data
temp_dir = os.path.join(os.environ["SPARK_LOCAL_DIRS"], "tmp", str(uuid.uuid4())[:8])
os.makedirs(temp_dir)

# Download the artifact and read it
client = MlflowClient()
training_data_path = client.download_artifacts("f7a73f26e72f45c392dc9a5e94438874", "data", temp_dir)
df = pd.read_parquet(os.path.join(training_data_path, "training_data"))

# Delete the temporary data
shutil.rmtree(temp_dir)

target_col = "CT_TRMNT_DT_Target"

# Convert Spark date columns to Pandas datetime
date_columns = ['CT_CNCLS_DT', 'CT_END_DT_2']
df[date_columns] = df[date_columns].apply(pd.to_datetime, errors="coerce")

# Convert columns detected to be of semantic type numeric
numeric_columns = ["POL_SN", "JOIN_SN", "COM_SN"]
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Semantic Type Detection Alerts
# MAGIC 
# MAGIC For details about the definition of the semantic types and how to override the detection, see
# MAGIC [Databricks documentation on semantic type detection](https://docs.microsoft.com/azure/databricks/applications/machine-learning/automl#semantic-type-detection).
# MAGIC 
# MAGIC - Semantic type `numeric` detected for columns `COM_SN`, `JOIN_SN`, `POL_SN`. Training notebooks will convert each column to a numeric type and encode features based on numerical transformations.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Profiling Results

# COMMAND ----------

from pandas_profiling import ProfileReport
df_profile = ProfileReport(df, minimal=True, title="Profiling Report", progress_bar=False, infer_dtypes=False)
profile_html = df_profile.to_html()

displayHTML(profile_html)
