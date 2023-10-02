# Databricks notebook source
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")

# COMMAND ----------

# MAGIC %run "../00-Configuration"

# COMMAND ----------

# MAGIC %md 
# MAGIC # Setup notebook
# MAGIC This notebook will ensure the cluster has the proper settings, and install required lib.
# MAGIC
# MAGIC It'll also automatically download the data from kaggle if it's not available locally (please set your kaggle credential in the `_kaggle_credential` companion notebook)
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection. View README for more details.  -->
# MAGIC <img width="1px" src="https://www.google-analytics.com/collect?v=1&gtm=GTM-NKQ8TT7&tid=UA-163989034-1&cid=555&aip=1&t=event&ec=field_demos&ea=display&dp=%2F42_field_demos%2Fmedia%2Fgaming_toxicity%2Fnotebook_setup&dt=MEDIA_USE_CASE_GAMING_TOXICITY">
# MAGIC <!-- [metadata={"description":"Companion notebook to setup libs and dependencies.</i>",
# MAGIC  "authors":["duncan.davis@databricks.com"]}] -->

# COMMAND ----------

# MAGIC %run ../../../_resources/00-global-setup $reset_all_data=$reset_all_data $db_prefix=manufacturing

# COMMAND ----------

try:
  path = "/mnt/field-demos/manufacturing/satellite-imaging"
  dbutils.fs.ls(path)
  print(f"Using default path {path} as raw data location")
  #raw data is set to the current user location where we just uploaded all the data
  raw_data_location = "/mnt/field-demos/manufacturing/satellite-imaging"
except:
  print(f"Couldn't find data saved in the default mounted bucket. Will download it from Kaggle instead under {cloud_storage_path}.")
  print("Note: you need to specify your Kaggle Key under ./_resources/_kaggle_credential ...")
  result = dbutils.notebook.run("./_resources/01_download", 3600, {"cloud_storage_path": cloud_storage_path + "/satellite-imaging"})
  if result is not None and "ERROR" in result:
    print("-------------------------------------------------------------")
    print("---------------- !! ERROR DOWNLOADING DATASET !!-------------")
    print("-------------------------------------------------------------")
    print(result)
    print("-------------------------------------------------------------")
    raise RuntimeError(result)
  else:
    print(f"Success. Dataset downloaded from kaggle and saved under {cloud_storage_path}.")
  raw_data_location = cloud_storage_path + "/satellite-imaging"


# COMMAND ----------

from PIL import Image
import io
from pyspark.sql.functions import col, substring_index, collect_list, size, pandas_udf
import mlflow
from typing import Iterator
import requests as r


# COMMAND ----------

# MAGIC %md ### Image encoded as base64 for example

# COMMAND ----------

mask_base64 = '/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAEAAQABAREA/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/9oACAEBAAA/APn+iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiivSPBHgXw/4n+HnifWbm9vk1fR4pZlghZRHsERaMtlDnLJIDhug7cE+b0UUUUUUUUUUUUUUUUUUUUUUUUUUUUUV6p8LNW03T/AvxEt73ULS2nu9MCW0c0yo0zeVOMICcscsBgeo9a8roooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooor1TQtJ02b9nLxPqkun2j6hDqaJFdtCplRd1vwr4yB8zcA9z615XRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRWxb+KdZtPC934agvNmkXcomnt/KQ73BUg7iNw+4vQ9vrWPRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRX//Z'
content_base64='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'

# COMMAND ----------

# MAGIC %run ./02-image-udf

# COMMAND ----------

# DBTITLE 1,Helper for realtime inference
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get("browserHostName").get()
headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {token}'}
import mlflow

local_model = None
def display_realtime_inference(image_base64, model_answer):
  #Currently we mock endpoint answer because it might not always be running during the demos, rest api answer is ignored
  global local_model
  if local_model == None:
    local_model = mlflow.pyfunc.load_model("models:/field_demos_satellite_segmentation/Production")
  pd2=local_model.predict(pd.DataFrame({'content': [image_base64]}))
  displayHTML(f"""
  <div style="float: left; margin-right: 10px">
    <h4>Model input</h4>
    <img src="data:image/png;base64, {image_base64}"/>
  </div>
  <div>
    <h4>Boat detected</h4>
    <img src="data:image/png;base64, {pd2[0]}"/>
  </div>""")


# COMMAND ----------

# DBTITLE 1,Image metadata for notebook display
# add the metadata to enable the image preview
image_meta = {"spark.contentAnnotation" : '{"mimeType": "image/jpeg"}'}
