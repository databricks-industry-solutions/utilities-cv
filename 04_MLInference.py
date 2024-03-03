# Databricks notebook source
# MAGIC %pip install pytorch-lightning==1.5.4 opencv-python==4.8.0.74 segmentation-models-pytorch shapely
# MAGIC

# COMMAND ----------

# MAGIC %run ./00-Configuration

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Model From Unity Catalog

# COMMAND ----------


import mlflow


mlflow.set_registry_uri("databricks-uc")
model_name = "utility_asset_accelerator"
logged_model = f'models:/{CATALOG}.{SCHEMA}.{model_name}@Production'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)



# COMMAND ----------

# MAGIC %md
# MAGIC # Perform Inference
# MAGIC This does simple inference on a sample image and draws bounding boxes around the assets
# MAGIC
# MAGIC ## Sample image
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/Power_Utilities_AssetID_Semantic_Seg_inference_test.jpg'>

# COMMAND ----------

import json
from PIL import Image 
import pandas as pd
import io
import requests

url = 'https://brysmiwasb.blob.core.windows.net/demos/images/Power_Utilities_AssetID_Semantic_Seg_inference_test.jpg'
response = requests.get(url)

image = Image.open(io.BytesIO(response.content)).resize((640,640))

output = io.BytesIO()
image.save(output, format='JPEG')
df = pd.DataFrame([{"data_input":output.getvalue()}])
_labels = json.loads(loaded_model.predict(df)[0])

from PIL import ImageDraw
color_map = {
  '1': "blue",
  '2':"green",
  '3':"red",
  '4':'purple',
  '5':'pink',
}
draw = ImageDraw.Draw(image)
for k in _labels.keys():
  for x in _labels[k]:
    color = color_map[k]
    draw.rectangle(x, outline=color, width=3)
image

# COMMAND ----------

# MAGIC %md
# MAGIC # Scaling
# MAGIC The above function won't scale well because it is using a single node with pandas. To distribute this we will load the same model on spark and perform distributed inference for a few records.
# MAGIC

# COMMAND ----------

spark_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model, result_type='string')
display(spark.table(f"{CATALOG}.{SCHEMA}.gold_asset_inventory").select("image_binary").limit(5).withColumn("labels",spark_model("image_binary")))

# COMMAND ----------

# MAGIC %md
# MAGIC # Scaling Round 2
# MAGIC Finally we will run inference across the entire source table to simulate real production inference. Ideally each image would have geo locations attached to it as well to save as part of the inference. In this dataset example the geo locations were removed.

# COMMAND ----------

(
  spark.table(f"{CATALOG}.{SCHEMA}.gold_asset_inventory")
 .select("image_binary").limit(5)
 .withColumn("labels",spark_model("image_binary"))
 .write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.gold_asset_inventory_predictions")
 )

# COMMAND ----------


