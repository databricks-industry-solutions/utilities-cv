# Databricks notebook source
# MAGIC %md
# MAGIC # Analyzing Electrical Grid Assets Using Computer Vision
# MAGIC
# MAGIC Utilities can have millions of individual distribution assets in circulation to help keep their grid operation smoothly. Many times these assets are out of sync with their digital versions on systems like GIS, or they deteriorate over time and become damaged. Drones have become the defacto way to take imagery of these assets, but it is a daunting task to manually review this imagery to correct GIS or identify assets that need repaired or replaced. These activities need to be automated through a subfield of machine learning called computer vision. This involved teaching a machine how to look at images and be able to identify items in that image.
# MAGIC
# MAGIC In this accelerator we will explore how to use Databricks for computer vision use cases that involve drone imagery and power distribution assets. This framework can easily be substitued with transmission assets, or solar generation assets. Addtionally we can extend this accelerator by cropping identified assets in order to apply other fine tuned models for things like damage detection.
# MAGIC
# MAGIC ## Notebook Outline
# MAGIC **01_DownloadData** will create some [UnityCatalog](https://www.databricks.com/product/unity-catalog) assets for us to download and store the images and labels needed to train our computer vision model
# MAGIC
# MAGIC **02_CreateTables** will manipulate our downloaded data so that labels will be applied to images for teaching the computer how to identify these assets
# MAGIC
# MAGIC **03_MultiGPU_ModelTraining** will utilize our table data to train our model through multiple rounds of training and testing. This will be done on more than one machine to make it faster
# MAGIC
# MAGIC **04_MLInference** will take the model trained in the previous notebook and look at a never before seen drone image to see if it can identify the assets properly

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
