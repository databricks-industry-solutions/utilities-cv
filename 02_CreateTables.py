# Databricks notebook source
# MAGIC %md
# MAGIC # Creating Tables
# MAGIC To prepare for training we are going to generate masks from our label data and combine that with our raw images to create a mask and image pair. This will be the foundation that will be fed into our pytorch model in the next notebook
# MAGIC
# MAGIC We will be utilizing bytes stored directly in delta tables so that we can pack more records into a file and not be as bound to IO if we were to keep each image in it's raw form on blob storage
# MAGIC
# MAGIC ![here](https://brysmiwasb.blob.core.windows.net/demos/images/Power_Utilities_AssetID_Table_Creation.png)

# COMMAND ----------

# MAGIC %pip install opencv-python==4.8.0.74

# COMMAND ----------

# MAGIC %run ./00-Configuration

# COMMAND ----------

# MAGIC %md
# MAGIC ## UDF to create image masks from class labels

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

    # Initialize a blank grayscale image of the same dimensions
    mask = np.zeros((h, w), np.uint8)
    
    # Draw polygons
    for class_id, class_polygons in polygons.items():
        for polygon in class_polygons:
            cv2.fillPoly(mask, [np.array(polygon, np.int32)], color=int(class_id))
    
    # Return the mask to be saved in delta
    _, encoded_img = cv2.imencode('.png', mask)
    masked_byte_array = encoded_img.tobytes()
    return masked_byte_array




# COMMAND ----------

# MAGIC %md
# MAGIC ## Convert Raw Label Data to Structured Values

# COMMAND ----------

# This dict will map the original class names to numeric for pytorch to consume
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
def transform_labels(objects):
  if objects is None: # if there are no objects in the image then exit early
    return {}
  obj = json.loads(objects) # load the objects from string to json
  obj_dict = {} # instantiate an empty return dict
  for v in obj['objects']: # loop through all objects in the image
    cls = v['value'] # get the string class value
    cls_int = obj_mapping[cls] # convert to the integer version
    if obj_dict.get(cls_int) is None: # create a new sub dict for the class if it does not exist already
      obj_dict[cls_int] = [] # create an empty list for the new class
    pnts = v.get('line') if v.get('line') is not None else v.get('polygon') # extract all of the points for the object
    
    obj_list = []
    for pnt in pnts: # convert all of the points to float and add it as the object points
      pnt_list = []
      pnt_list.append(float(pnt['x']))
      pnt_list.append(float(pnt['y']))
      obj_list.append(pnt_list)
    obj_dict[cls_int].append(obj_list)
  return json.dumps(obj_dict) # return the mapped object points

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get the Raw Images and Save to Delta

# COMMAND ----------


raw_images = f"{project_location}raw_images/"

raw_df = (
  spark.read.format("binaryfile") # read the images as binary so we can get the file names and the byte content
  .load(raw_images)
  .selectExpr("*","replace(_metadata.file_name,'%20') as file_name") # cleans the file name to remove spaces
)

raw_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.raw_images_bronze") # write out the images to a bronze delta table

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get the Raw Label Data and Save to Delta

# COMMAND ----------

from pyspark.sql.functions import to_json, from_json
# read the csv label file and apply some json structure to the label column
label_df = (
  spark.read.csv(f"{project_location}label_data", header=True)
  .withColumnRenamed("External ID","file_name")
  .withColumn("Label",from_json("Label","objects array<struct<value:string, line:array<struct<x:float,y:float>>,polygon:array<struct<x:float,y:float>>>>"))
  .selectExpr("Label","replace(file_name, ' ') as file_name")
)
label_df.write.mode('overwrite').saveAsTable(f"{CATALOG}.{SCHEMA}.label_data") # write out the labels to a delta table

# COMMAND ----------

# DBTITLE 1,Combine Label and Images to Create Mask Dataset
# read the raw images and join to the labels
mask_df = (
  spark.sql(f"""
          select A.file_name,content, to_json(label) as label from {CATALOG}.{SCHEMA}.raw_images_bronze A
          left join {CATALOG}.{SCHEMA}.label_data B using(file_name)
          """).withColumn("mask_labels",transform_labels('label'))
  .withColumn("mask_binary", create_mask_from_polygons("content", "mask_labels")) # utilize the mask udf to generate object masks for each image
  .drop("content")
)

mask_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.silver_mask") # Save the masks to a silver delta table

# COMMAND ----------

# DBTITLE 1,Create Training Dataset with Original Images and Masks Combined
# Join the masks and raw images together and save to a single gold table.
# This will be helpful to not have to join these multiple times in the future if we try different models
gold_df = (
  spark.sql(f"""
            select mask_binary, content as image_binary, A.file_name
            from {CATALOG}.{SCHEMA}.raw_images_bronze A join {CATALOG}.{SCHEMA}.silver_mask B using(file_name)
            """)
)
gold_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.gold_asset_inventory")

# COMMAND ----------

# DBTITLE 1,Create Test Train Split
# Create a test/train split and save these to a volume so the petastorm can easily pick them up
gold_satellite_image =spark.table(f"{CATALOG}.{SCHEMA}.gold_asset_inventory")
(images_train, images_test) = gold_satellite_image.randomSplit([0.8, 0.2 ], 42)
images_train.write.mode("overwrite").save(f"{project_location}gold_asset_train")
images_test.write.mode("overwrite").save(f"{project_location}gold_asset_test")


# COMMAND ----------


