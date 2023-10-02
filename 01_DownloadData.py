# Databricks notebook source
# MAGIC %md
# MAGIC # Download instructions
# MAGIC Go [here](https://forms.office.com/pages/responsepage.aspx?id=AGTMMj0V90K8VXflpQ8bz0Fhowp3sHlPsHuc9DMJSEBURFFQTkNWVFhBVUxZRVU1NVJFU0wxR0taUS4u) and fill out the form to get the root url for downloading the zip files and place it in the root_location

# COMMAND ----------

dbutils.widgets.dropdown("reset_all","False",["True","False"])
dbutils.widgets.text("root_location","https://xxxx.blob.core.windows.net/xxxx/")

# COMMAND ----------

# MAGIC %run ./00-Configuration

# COMMAND ----------

import urllib.request
import os, shutil
import json
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf
import requests



download_location = dbutils.widgets.get("root_location")

spark.sql(f"create catalog if not exists {CATALOG}")
spark.sql(f"create schema if not exists {CATALOG}.{SCHEMA}")
if dbutils.widgets.get("reset_all") == 'True':
  if spark.sql(f"show volumes from {CATALOG}.{SCHEMA}").where(f"volume_name='{VOLUME}'").count() >0:
    dbutils.fs.rm(f'{project_location}label_data/', recurse=True)
    dbutils.fs.rm(f'{project_location}raw_images/', recurse=True)
    dbutils.fs.rm(f'{project_location}training/', recurse=True)
  spark.sql(f"drop VOLUME if exists {CATALOG}.{SCHEMA}.{VOLUME}")

spark.sql(f"create volume if not exists {CATALOG}.{SCHEMA}.{VOLUME}")  





# COMMAND ----------

from datetime import datetime, timezone

os.makedirs(f"/tmp/utility_accelerator/raw_images/", exist_ok=True)

def data_download(location):
  with requests.get(f"{download_location}{location}", stream=True) as r:
    with open(f"/tmp/utility_accelerator/raw_images/{location}", 'wb') as f:
      shutil.copyfileobj(r.raw,f)

for file in files:
  print(f"Starting {file} at {datetime.now(timezone.utc).astimezone().isoformat()}")
  data_download(file)
  print(f"Ending {file} at {datetime.now(timezone.utc).astimezone().isoformat()}")
  

# COMMAND ----------

def unzip(location):
  import zipfile
  with zipfile.ZipFile(f"/tmp/utility_accelerator/raw_images/{location}", 'r') as zip_ref:
    zip_ref.extractall(f"/tmp/utility_accelerator/raw_images/")
  os.remove(f"/tmp/utility_accelerator/raw_images/{location}")

for file in files:
  unzip(file)


# COMMAND ----------

import shutil
shutil.copytree('/tmp/utility_accelerator/raw_images/',f'{project_location}raw_images/', dirs_exist_ok=True)

# COMMAND ----------

import urllib.request, os
os.makedirs(f"{project_location}label_data", exist_ok=True)

urllib.request.urlretrieve(f"{download_location}Overhead-Distribution-Labels.csv", f"{project_location}label_data/Overhead_Distribution_Labels.csv")

# COMMAND ----------


