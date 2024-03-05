# Databricks notebook source
# MAGIC %pip install pytorch-lightning==1.5.4 opencv-python==4.8.0.74 segmentation-models-pytorch deltalake

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Type
# MAGIC Segmentation models allow us to draw tighter borders around objects in an image compared to object detection. In the transmission area this might be overkill, but when dealing with distrbution lines there are a lot more items more tightly packed onto structures. Segmentation allows us to get tighter crops on objects in order to do further classification with other fine tuned models.
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/Power_Utilities_AssetID_Semantic_Seg.png' width=###>

# COMMAND ----------

# MAGIC %run ./00-Configuration

# COMMAND ----------

# MAGIC %run ./_resources/03-DL-helpers

# COMMAND ----------

import mlflow
import torch
from deltalake import DeltaTable
from petastorm.spark import SparkDatasetConverter, make_spark_converter
from petastorm import TransformSpec

# COMMAND ----------

# MAGIC %md
# MAGIC # Get Delta Files
# MAGIC We need to get a list of delta files to pass to petastorm for more efficient distributed training.

# COMMAND ----------

# for the train and validation sets we get the parquet file urls for petastorm to use
# petastorm will help us do distributed training
training_dt = DeltaTable(f'{project_location}gold_asset_train')
train_parquet_files = training_dt.file_uris()
train_parquet_files = [
    parquet_file.replace("/Volumes", "file:///Volumes")
    for parquet_file in train_parquet_files
]
train_rows = training_dt.get_add_actions().to_pandas()["num_records"].sum()

val_dt = DeltaTable(f"{project_location}gold_asset_test")
val_parquet_files = val_dt.file_uris()
val_parquet_files = [
    parquet_file.replace("/Volumes", "file:///Volumes") for parquet_file in val_parquet_files
]
val_rows = val_dt.get_add_actions().to_pandas()["num_records"].sum()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Module For Pytorch

# COMMAND ----------

object_id_to_class_mapping={"1":"pole","2":"transformer"}
from petastorm.reader import make_batch_reader
from petastorm.pytorch import DataLoader
import pytorch_lightning as pl
from torchvision import transforms, models

class UtilityDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_parquet_files,
        val_parquet_files,
        device_id: int = 0,
        device_count: int = 1,
        batch_size: int = 16,
        num_epochs: int = 1,
        workers_count: int = 1,
        reader_pool_type: str = "dummy",
        result_queue_size: int = 1,
        feature_column: str = "image_binary",
        label_column: str = "mask_binary",
        object_id_to_class_mapping: dict = object_id_to_class_mapping,
    ):
        super().__init__()
        self.save_hyperparameters()
 
        self.train_dataloader_context = None
        self.val_dataloader_context = None
 
    def create_dataloader_context(self, input_parquet_files):
        petastorm_reader_kwargs = {
            "transform_spec": self._get_transform_spec(),
            "cur_shard": self.hparams.device_id,
            "shard_count": self.hparams.device_count,
            "workers_count": self.hparams.workers_count,
            "reader_pool_type": self.hparams.reader_pool_type,
            "results_queue_size": self.hparams.result_queue_size,
            "num_epochs": None,
        }
        return DataLoader(
            make_batch_reader(input_parquet_files, **petastorm_reader_kwargs),
            self.hparams.batch_size,
        )
 
    def train_dataloader(self):
        if self.train_dataloader_context is not None:
            self.train_dataloader_context.__exit__(None, None, None)
        self.train_dataloader_context = self.create_dataloader_context(
            self.hparams.train_parquet_files
        )
        return self.train_dataloader_context.__enter__()
 
    def val_dataloader(self):
        if self.val_dataloader_context is not None:
            self.val_dataloader_context.__exit__(None, None, None)
        self.val_dataloader_context = self.create_dataloader_context(
            self.hparams.val_parquet_files
        )
        return self.val_dataloader_context.__enter__()
 
    def teardown(self, stage=None):
        # Close all readers (especially important for distributed training to prevent errors)
        self.train_dataloader_context.__exit__(None, None, None)
        self.val_dataloader_context.__exit__(None, None, None)
 
    
 
    def transform_row(self, batch_pd):
      import torchvision
      # transform images
      # -----------------------------------------------------------
      # transform step 1: read incoming content value as an image
      transformersImage = [torchvision.transforms.Lambda(lambda x: (np.moveaxis(np.array(Image.open(io.BytesIO(x)).resize((640,640))), -1, 0)).copy())]
      # transformersMask = [torchvision.transforms.Lambda(lambda x:  np.minimum(np.expand_dims(np.array(Image.open(io.BytesIO(x)).resize((640,640))), 0),1))]
      transformersMask = [torchvision.transforms.Lambda(lambda x:  np.expand_dims(np.array(Image.open(io.BytesIO(x)).resize((640,640))), 0))]

      # assemble transformation steps into a pipeline
      transImg = torchvision.transforms.Compose(transformersImage)
      transMask = torchvision.transforms.Compose(transformersMask)
      
      # apply pipeline to images 
      batch_pd['image_binary'] = batch_pd['image_binary'].map(lambda x: transImg(x))
      
      # -----------------------------------------------------------
      # transform labels (our evaluation metric expects values to be float32)
      # -----------------------------------------------------------
      batch_pd['mask_binary'] = batch_pd['mask_binary'].map(lambda x: transMask(x))
      # -----------------------------------------------------------
      return batch_pd[['image_binary', 'mask_binary']]
 
    def _get_transform_spec(self):
        return TransformSpec(
            self.transform_row, # function to call to retrieve/transform row
            edit_fields=[  # schema of rows returned by function
                ('image_binary', np.uint8, (3, IMAGE_RESIZE, IMAGE_RESIZE), False), 
                ('mask_binary', np.uint8, (1,IMAGE_RESIZE, IMAGE_RESIZE), False)], 
            selected_fields=['image_binary', 'mask_binary'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get Infrastructure Variables Dynamically

# COMMAND ----------

import torch
from torch import optim, nn, utils, Tensor
from torchvision import datasets, transforms
import pytorch_lightning as pl
from pyspark.ml.torch.distributor import TorchDistributor
import mlflow
import os
import numpy as np
 

# NOTE: This assumes the driver node and worker nodes have the same instance type.
NUM_GPUS_PER_WORKER = torch.cuda.device_count() # CHANGE AS NEEDED
USE_GPU = NUM_GPUS_PER_WORKER > 0
 
# get the user dynamically and set the experiment path for MLFlow
username = spark.sql("SELECT current_user()").first()['current_user()']
experiment_path = f'/Users/{username}/pytorch-distributor'
 
# This is needed for later in the notebook when we are doing distrbuted training
db_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().extraContext().apply('api_url')
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
 
# Manually create the experiment so that you can get the ID and can send it to the worker nodes for scaling
experiment = mlflow.set_experiment(experiment_path)
 

# COMMAND ----------

# MAGIC %md
# MAGIC # Main Training Function
# MAGIC This function will be used to scale from a single gpu to multi node GPU training
# MAGIC
# MAGIC ## Commoditity GPU's
# MAGIC Large GPUs may not be readily available all of the time, and smaller GPUs are not always up to the task to train large computer vision models. Being able to distribute GPU training across many smaller nodes allows us to take advantage of the readily available GPU hardware and chain them together.
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/Power_Utilities_AssetID_MultiGPU_Ref.png' width=###>

# COMMAND ----------

BATCH_SIZE = 16
MAX_EPOCHS = 1 #Make larger for better results
READER_POOL_TYPE = "thread"
RESULTS_QUEUE_SIZE = 20
from math import ceil
import os

def main_training_loop(num_tasks, num_proc_per_task, run_id=None):
  import warnings
  warnings.filterwarnings("ignore")
  """
  
  Main train and test loop
  
  """
  # add imports inside pl_train for pickling to work
  from torch import optim, nn, utils, Tensor
  from torchvision import datasets, transforms
  import pytorch_lightning as pl
  import mlflow
  from mlflow.types.schema import Schema, ColSpec, TensorSpec
  from mlflow.models.signature import ModelSignature

  os.mkdir('/tmp/utility_asset_model')
  os.chdir('/tmp/utility_asset_model')
  mlflow.pytorch.autolog()
  input_schema = Schema(
    [
        ColSpec("binary","data_input")
    ]
  )
  output_schema = Schema([
    ColSpec("string","data_output")
  ])
  signature = ModelSignature(inputs=input_schema, outputs=output_schema)
  
  ############################
  ##### Setting up MLflow ####
  # We need to do this so that different processes that will be able to find mlflow
  os.environ['DATABRICKS_HOST'] = db_host
  os.environ['DATABRICKS_TOKEN'] = db_token
  
  os.environ['NCCL_P2P_DISABLE'] = '1'
  WORLD_SIZE = num_tasks * num_proc_per_task
  node_rank = int(os.environ.get("NODE_RANK",0))
  
  train_steps_per_epoch = ceil(train_rows // (BATCH_SIZE * WORLD_SIZE))
  val_steps_per_epoch = ceil(val_rows // (BATCH_SIZE * WORLD_SIZE))
  # epochs = 5
  
  #setup the model using Feature Pyramid Network architecture, resnet34 encoder, RGB (3 channels), and 6 potential classes
  model = UtilityAssetModel("FPN", "resnet34", in_channels=3, out_classes=6)
 
  datamodule = UtilityDataModule(train_parquet_files=train_parquet_files,
                                  val_parquet_files=val_parquet_files, 
                                  batch_size=BATCH_SIZE,
                                  workers_count=1,
                                  reader_pool_type=READER_POOL_TYPE,
                                  device_id=node_rank,
                                  device_count=WORLD_SIZE,
                                  result_queue_size=RESULTS_QUEUE_SIZE)
  
  # train the model
  if num_tasks >1:
    kwargs = {"strategy":"ddp"}
  else:
    kwargs ={}
  trainer = pl.Trainer(accelerator='gpu',
                       devices=num_proc_per_task, 
                       num_nodes=num_tasks,
                       max_epochs=MAX_EPOCHS,
                       limit_train_batches=train_steps_per_epoch,
                       val_check_interval=train_steps_per_epoch,
                       num_sanity_val_steps=0,
                       limit_val_batches=val_steps_per_epoch,
                       reload_dataloaders_every_n_epochs=1,
                       enable_checkpointing=True,
                       **kwargs,
                       )
  
  if run_id is not None:
    mlflow.start_run(run_id=run_id)
  trainer.fit(model=model, datamodule=datamodule)
  delattr(model,"trainer")
  return model, trainer.checkpoint_callback.best_model_path

# COMMAND ----------

# MAGIC %md
# MAGIC ## Uncomment For Single Node Training

# COMMAND ----------

# NUM_TASKS = 1
# NUM_PROC_PER_TASK = 1
# mlflow.pytorch.autolog() 

# from mlflow.types.schema import Schema, ColSpec, TensorSpec
# from mlflow.models.signature import ModelSignature

# mlflow.pytorch.autolog()
# input_schema = Schema(
#   [
#       ColSpec("binary","data_input")
#   ]
# )

# output_schema = Schema([
#   ColSpec("string","data_output")
# ])
# signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# with mlflow.start_run() as run:
#   model, ckpt_path = main_training_loop(NUM_TASKS, NUM_PROC_PER_TASK)
#   mlflow.pyfunc.log_model(artifact_path="model", 
#                           python_model=CVModelWrapper(model),
#                           signature=signature,
#                           )

# COMMAND ----------

# MAGIC %md
# MAGIC # Run Training
# MAGIC This function will work as a distributed framework by using `TorchDistributor`.
# MAGIC
# MAGIC This is especially important in the age of LLM's where there is a shortage of fire breathing GPU behemoth's. Instead of trying to get our hands on A100's or H100's we can setup many T4 machines that are more readily available and cluster them together to work in concert
# MAGIC
# MAGIC Configure how many workers are in the cluster and all other configs should be dynamic

# COMMAND ----------

NUM_WORKERS = 2 # how many executor nodes there are in the cluster
NUM_TASKS = NUM_WORKERS * NUM_GPUS_PER_WORKER
NUM_PROC_PER_TASK = 1 # leave this at 1
NUM_PROC = NUM_TASKS * NUM_PROC_PER_TASK
 
from mlflow.types.schema import Schema, ColSpec, TensorSpec
from mlflow.models.signature import ModelSignature

# set the schema for our model so we can register it in Unity Catalog
input_schema = Schema(
  [
      ColSpec("binary","data_input")
  ]
)

output_schema = Schema([
  ColSpec("string","data_output")
])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

with mlflow.start_run() as run:
  run_id= mlflow.active_run().info.run_id
  # TorchDistributor allows us to easily distribute our taining to multiple nodes in a cluster
  (model, ckpt_path) = TorchDistributor(num_processes=NUM_PROC, local_mode=False, use_gpu=True).run(main_training_loop, NUM_TASKS, NUM_PROC_PER_TASK, run_id) 
  mlflow.pyfunc.log_model(artifact_path="model", 
                          python_model=CVModelWrapper(model),
                          signature=signature,
                          )


# COMMAND ----------

# MAGIC %md
# MAGIC # Register Model to UC
# MAGIC This will register a model in Unity Catalog for inference later

# COMMAND ----------

import mlflow
from mlflow import MlflowClient


model_name = "utility_asset_accelerator"
mlflow.set_registry_uri("databricks-uc")

_register = mlflow.register_model(f"runs:/{run_id}/model", f"{CATALOG}.{SCHEMA}.{model_name}")

client = MlflowClient()
client.set_registered_model_alias(f"{CATALOG}.{SCHEMA}.{model_name}","Production", int(_register.version))


# COMMAND ----------


