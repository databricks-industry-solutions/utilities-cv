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
