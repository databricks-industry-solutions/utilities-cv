# Databricks notebook source
# MAGIC %pip install opencv-python==4.8.0.74

# COMMAND ----------

# MAGIC %run ./00-Configuration

# COMMAND ----------

import cv2
import os
import numpy as np
import json
from pyspark.sql.functions import udf, from_json

@udf('binary')
def create_mask_from_polygons(image_bytes, polygons):
    polygons = json.loads(polygons)
    # Read the original image to get its dimensions
    x = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(x, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    # h = 3888
    # w = 5184

    # Initialize a blank grayscale image of the same dimensions
    mask = np.zeros((h, w), np.uint8)
    
    # Draw polygons
    for class_id, class_polygons in polygons.items():
        for polygon in class_polygons:
            cv2.fillPoly(mask, [np.array(polygon, np.int32)], color=int(class_id))
    
    # Save the mask
    _, encoded_img = cv2.imencode('.png', mask)
    masked_byte_array = encoded_img.tobytes()
    return masked_byte_array




# COMMAND ----------

obj_mapping = {
  "insulator": 4,
  "crossarm":3,
  "conductor":5,
  "pole":1,
  "cutouts":0,
  "other_wire":0,
  "guy_wires":0,
  "transformers":2,
  "background_structure":0
}

@udf("string")
def my_conv(objects):
  if objects is None:
    return {}
  obj = json.loads(objects)
  obj_dict = {}
  for v in obj['objects']:
    cls = v['value']
    cls_int = obj_mapping[cls]
    if obj_dict.get(cls_int) is None:
      obj_dict[cls_int] = []
    pnts = v.get('line') if v.get('line') is not None else v.get('polygon')
    
    obj_list = []
    for pnt in pnts:
      pnt_list = []
      pnt_list.append(float(pnt['x']))
      pnt_list.append(float(pnt['y']))
      obj_list.append(pnt_list)
    obj_dict[cls_int].append(obj_list)
  return json.dumps(obj_dict)

# COMMAND ----------


raw_images = f"{project_location}raw_images/"

raw_df = spark.read.format("binaryfile").load(raw_images).selectExpr("*","replace(_metadata.file_name,'%20') as file_name")

raw_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.raw_images_bronze")

# COMMAND ----------

from pyspark.sql.functions import to_json, from_json
label_df = (
  spark.read.csv(f"{project_location}label_data", header=True)
  .withColumnRenamed("External ID","file_name")
  .withColumn("Label",from_json("Label","objects array<struct<value:string, line:array<struct<x:float,y:float>>,polygon:array<struct<x:float,y:float>>>>"))
  .selectExpr("Label","replace(file_name, ' ') as file_name")
)
label_df.write.mode('overwrite').saveAsTable(f"{CATALOG}.{SCHEMA}.label_data")

# COMMAND ----------

mask_df = (
  spark.sql(f"""
          select A.file_name,content, to_json(label) as label from {CATALOG}.{SCHEMA}.raw_images_bronze A
          left join {CATALOG}.{SCHEMA}.label_data B using(file_name)
          """).withColumn("mask_labels",my_conv('label'))
  .withColumn("mask_binary", create_mask_from_polygons("content", "mask_labels"))
  .drop("content")
)

# COMMAND ----------

mask_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.silver_mask")

# COMMAND ----------

gold_df = (
  spark.sql(f"""
            select mask_binary, content as image_binary, A.file_name
            from {CATALOG}.{SCHEMA}.raw_images_bronze A join {CATALOG}.{SCHEMA}.silver_mask B using(file_name)
            """)
)
gold_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.gold_asset_inventory")

# COMMAND ----------

gold_satellite_image =spark.table(f"{CATALOG}.{SCHEMA}.gold_asset_inventory")
(images_train, images_test) = gold_satellite_image.randomSplit([0.8, 0.2 ], 42)
images_train.write.mode("overwrite").save(f"{project_location}gold_asset_train")
images_test.write.mode("overwrite").save(f"{project_location}gold_asset_test")


# COMMAND ----------


