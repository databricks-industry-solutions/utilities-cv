# Databricks notebook source
# MAGIC %pip install pytorch-lightning==1.5.4 opencv-python==4.8.0.74 segmentation-models-pytorch shapely
# MAGIC

# COMMAND ----------

# MAGIC %run ./00-Configuration

# COMMAND ----------


import mlflow


mlflow.set_registry_uri("databricks-uc")
model_name = "utility_asset_accelerator"
logged_model = f'models:/{CATALOG}.{SCHEMA}.{model_name}@Production'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)



# COMMAND ----------

import json
from PIL import Image 
import pandas as pd
import io
# with Image.open('/Workspace/Users/david.radford@databricks.com/Demos/ComputerVision/Startover/Pole_Transformer_petastorm_stuck_new/10.jpg') as image:
image = Image.open('/Workspace/Users/david.radford@databricks.com/Demos/ComputerVision/Startover/Pole_Transformer_petastorm_stuck_new/10.jpg').resize((640,640))

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
    # draw.polygon(x, outline=color, width=3)
image

# COMMAND ----------

spark_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model, result_type='string')
display(spark.table(f"{CATALOG}.{SCHEMA}.gold_asset_inventory").select("image_binary").limit(5).withColumn("labels",spark_model("image_binary")))

# COMMAND ----------

(
  spark.table(f"{CATALOG}.{SCHEMA}.gold_asset_inventory")
 .select("image_binary").limit(5)
 .withColumn("labels",spark_model("image_binary"))
 .write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.gold_asset_inventory_predictions")
 )

# COMMAND ----------


