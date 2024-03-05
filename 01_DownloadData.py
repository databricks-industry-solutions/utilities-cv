# Databricks notebook source
# MAGIC %md
# MAGIC # Download instructions
# MAGIC This dataset is taken from an open epri inspection dataset. Details can be found [here](https://www.kaggle.com/datasets/dexterlewis/epri-distribution-inspection-imagery)
# MAGIC
# MAGIC In order to download the dataset you need to agree to their license terms found [here](https://forms.office.com/pages/responsepage.aspx?id=AGTMMj0V90K8VXflpQ8bz0Fhowp3sHlPsHuc9DMJSEBURFFQTkNWVFhBVUxZRVU1NVJFU0wxR0taUS4u) and fill out the form to get the root url for downloading the zip files and place it in the root_location
# MAGIC
# MAGIC ### Attribution
# MAGIC - Dataset: Drone-based Distribution Inspection Imagery 1.0
# MAGIC - DOI: 10.34740/kaggle/dsv/3803175
# MAGIC - Creator: EPRI, P. Kulkarni, D. Lewis,
# MAGIC - License: CC BY-SA 4.0

# COMMAND ----------

# MAGIC %md
# MAGIC ## Variable Configuration
# MAGIC Replace the xxxx values in the root_location with the values that you received via email after accepting the license terms

# COMMAND ----------

dbutils.widgets.dropdown("reset_all","False",["True","False"])
dbutils.widgets.text("root_location","https://xxxx.blob.core.windows.net/xxxx/")

# COMMAND ----------

# MAGIC %run ./00-Configuration

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Unity Catalog Assets

# COMMAND ----------

# MAGIC %sql show catalogs like 'mfg_utilities_accelerator'

# COMMAND ----------

import urllib.request
import os, shutil
import json
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf
import requests



download_location = dbutils.widgets.get("root_location")

exists_count = spark.sql(f"show catalogs like '{CATALOG}'").count()
if exists_count ==0:
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

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download and Extract
# MAGIC These next two cells will download zip files from EPRI and unzip them locally to save network IO

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

# MAGIC %md
# MAGIC ## Copy Unzipped Assets
# MAGIC Copy the files to the unity catalog volume

# COMMAND ----------

import shutil
shutil.copytree('/tmp/utility_accelerator/raw_images/',f'{project_location}raw_images/', dirs_exist_ok=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download Label CSV

# COMMAND ----------

import urllib.request, os
os.makedirs(f"{project_location}label_data", exist_ok=True)

urllib.request.urlretrieve(f"{download_location}Overhead-Distribution-Labels.csv", f"{project_location}label_data/Overhead_Distribution_Labels.csv")

# COMMAND ----------


