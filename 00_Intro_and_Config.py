# Databricks notebook source
# MAGIC %pip install pytorch-lightning==1.5.4 opencv-python==4.8.0.74 segmentation-models-pytorch shapely deltalake

# COMMAND ----------

IMAGE_RESIZE=640
IMAGE_SIZE=640
NUM_EPOCHS=5
BATCH_SIZE=16 
CATALOG ="mfg_utilities_accelerator"
SCHEMA = "utility_vision_accelerator"
VOLUME = "drone_data"

project_location = f'/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/'

files = [
  'Circuit3.zip',
]

# COMMAND ----------

# MAGIC %run ./_resources/03-DL-helpers

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | pytorch-lightning  | lightweight PyTorch wrapper for ML researchers | Apache Software License (Apache-2.0)   | https://pypi.org/project/pytorch-lightning/   |
# MAGIC | opencv-python  | Wrapper package for OpenCV python bindings| Apache Software License (Apache 2.0)   | https://pypi.org/project/opencv-python/   |
# MAGIC | segmentation-models-pytorch  | Image segmentation models with pre-trained backbones. PyTorch. | MIT License (MIT)    | https://pypi.org/project/segmentation-models-pytorch/  |
# MAGIC | shapely  |Manipulation and analysis of geometric objects |BSD License (BSD 3-Clause)    | https://pypi.org/project/shapely/   |
