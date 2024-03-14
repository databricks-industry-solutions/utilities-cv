# Databricks notebook source
# MAGIC %md
# MAGIC # Inference
# MAGIC Model inference involves using the the model that has learned how to find the distribution assets in an image to feed in new images for detection of these assets. The model is stored in Unity Catalog right next to the data and can be applied as data is loaded into the system.

# COMMAND ----------

# MAGIC %run ./00_Intro_and_Config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Model From Unity Catalog

# COMMAND ----------


import mlflow


mlflow.set_registry_uri("databricks-uc") # set the model registry to Unity Catalog
model_name = "utility_asset_accelerator"
logged_model = f'models:/{CATALOG}.{SCHEMA}.{model_name}@Production'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model) 



# COMMAND ----------

# MAGIC %md
# MAGIC # Sample image
# MAGIC
# MAGIC This is a sample image that does not exist in the original dataset that we will use for testing our inference
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/Power_Utilities_AssetID_Semantic_Seg_inference_test.jpg'>

# COMMAND ----------

# MAGIC %md
# MAGIC # Perform Inference
# MAGIC This does simple inference on a sample image and draws bounding boxes around the assets. The image is loaded from a url into a PIL image, ran through the trained model and then has labels drawn on it.

# COMMAND ----------

import json
from PIL import Image 
import pandas as pd
import io
import requests
import cv2
from matplotlib import colors
from dbruntime.patches import cv2_imshow
import numpy as np

url = 'https://brysmiwasb.blob.core.windows.net/demos/images/Power_Utilities_AssetID_Semantic_Seg_inference_test.jpg'
response = requests.get(url) # use python requests to get the image data

image = Image.open(io.BytesIO(response.content)).resize((640,640)) # convert the image data into PIL format and resize it so our model understands it

output = io.BytesIO() # convert back into bytes for the model to understand
image.save(output, format='JPEG')
df = pd.DataFrame([{"data_input":output.getvalue()}])
_labels = json.loads(loaded_model.predict(df)[0]) # send the image bytes through the model to get predictions
image = np.array(image) # convert the image into numpyarray so opencv can draw the labels onto it



from PIL import ImageDraw
draw_labels(image, _labels)
cv2_imshow(image)

# COMMAND ----------

# MAGIC %md
# MAGIC # Scaling
# MAGIC The above function won't scale well because it is using a single node with pandas. To distribute this we will load the same model on spark and perform distributed inference for a few records.
# MAGIC

# COMMAND ----------

# get the trained model as a spark user defined function instead so that we can apply it to a spark dataframe
spark_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model, result_type='string')

#feed the image data from the gold table through the model and display the labels for each image
display(spark.table(
  f"{CATALOG}.{SCHEMA}.gold_asset_inventory")
  .select("image_binary")
  .limit(5)
  .withColumn("labels",spark_model("image_binary"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Scaling Round 2
# MAGIC Finally we will run inference across the entire source table to simulate real production inference. Ideally each image would have geo locations attached to it as well to save as part of the inference. In this dataset example the geo locations were removed.

# COMMAND ----------

(
  spark.table(f"{CATALOG}.{SCHEMA}.gold_asset_inventory")
 .select("image_binary")
 .withColumn("labels",spark_model("image_binary"))
 .write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.gold_asset_inventory_predictions")
 )

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
