# Databricks notebook source
IMAGE_RESIZE=640
IMAGE_SIZE=640
NUM_EPOCHS=5
BATCH_SIZE=16 
CATALOG ="mfg_utilities_accelerator"
SCHEMA = "utility_vision_accelerator"
VOLUME = "drone_data"

project_location = f'/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/'

files = [
  # 'Circuit1.zip',
  # 'Circuit2.zip',
  'Circuit3.zip',
  # 'Circuit4.zip',
  # 'Circuit5.zip',
  # 'Circuit6.zip',
  # 'Circuit7.zip',
  # 'Circuit8.zip',
  # 'Circuit9.zip',
  # 'Circuit10.zip',
  # 'Circuit11A.zip',
  # 'Circuit11B.zip',
  # 'Circuit11C.zip',
  # 'Circuit12A.zip',
  # 'Circuit12B.zip',
  # 'Circuit13A.zip',
  # 'Circuit13B.zip',
  # 'Circuit14.zip',
  # 'Circuit15A.zip',
  # 'Circuit15B.zip',
  # 'Circuit16A.zip',
  # 'Circuit16B.zip'
]
