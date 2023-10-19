# Databricks notebook source
# MAGIC %pip install pytorch-lightning==1.5.4 opencv-python==4.8.0.74 segmentation-models-pytorch deltalake

# COMMAND ----------

# MAGIC %run ./00-Configuration

# COMMAND ----------

# MAGIC %run ./_resources/03-DL-helpers

# COMMAND ----------

import mlflow
import torch

from petastorm.spark import SparkDatasetConverter, make_spark_converter
from petastorm import TransformSpec

# COMMAND ----------

from deltalake import DeltaTable
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
 
username = spark.sql("SELECT current_user()").first()['current_user()']
 
experiment_path = f'/Users/{username}/pytorch-distributor'
 
# This is needed for later in the notebook
db_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().extraContext().apply('api_url')
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
 
# Manually create the experiment so that you can get the ID and can send it to the worker nodes for scaling
experiment = mlflow.set_experiment(experiment_path)
 

# COMMAND ----------

BATCH_SIZE = 16
MAX_EPOCHS = 1
READER_POOL_TYPE = "thread"
RESULTS_QUEUE_SIZE = 20
from math import ceil
import os

def main_training_loop(num_tasks, num_proc_per_task, run_id=None):
  # from MyDataModule import ImageNetDataModule
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
  # output_schema = Schema([
  #   TensorSpec(np.dtype(np.float32), (1,1,640,640))
  # ])
  output_schema = Schema([
    ColSpec("string","data_output")
  ])
  signature = ModelSignature(inputs=input_schema, outputs=output_schema)
  
  ############################
  ##### Setting up MLflow ####
  # We need to do this so that different processes that will be able to find mlflow
  os.environ['DATABRICKS_HOST'] = db_host
  os.environ['DATABRICKS_TOKEN'] = db_token
  
  # NCCL P2P can cause issues with incorrect peer settings, so let's turn this off to scale for now
  os.environ['NCCL_P2P_DISABLE'] = '1'
  WORLD_SIZE = num_tasks * num_proc_per_task
  node_rank = int(os.environ.get("NODE_RANK",0))
  
  train_steps_per_epoch = ceil(train_rows // (BATCH_SIZE * WORLD_SIZE))
  val_steps_per_epoch = ceil(val_rows // (BATCH_SIZE * WORLD_SIZE))
  # epochs = 5
  
  # mlf_logger = pl.loggers.MLFlowLogger(experiment_name=experiment_path, log_model=True)
  
  # define any number of nn.Modules (or use your current ones)
 
  # init the autoencoder
  model = UtilityAssetModel("FPN", "resnet34", in_channels=3, out_classes=6)
 
  datamodule = UtilityDataModule(train_parquet_files=train_parquet_files,
                                  val_parquet_files=val_parquet_files, batch_size=BATCH_SIZE,
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
                      #  strategy=strategy, 
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

# DBTITLE 1,Single Node Training
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

NUM_WORKERS = 2
NUM_TASKS = NUM_WORKERS * NUM_GPUS_PER_WORKER
NUM_PROC_PER_TASK = 1
NUM_PROC = NUM_TASKS * NUM_PROC_PER_TASK
 
from mlflow.types.schema import Schema, ColSpec, TensorSpec
from mlflow.models.signature import ModelSignature

# mlflow.pytorch.autolog()
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
  (model, ckpt_path) = TorchDistributor(num_processes=NUM_PROC, local_mode=False, use_gpu=True).run(main_training_loop, NUM_TASKS, NUM_PROC_PER_TASK, run_id) 
  mlflow.pyfunc.log_model(artifact_path="model", 
                          python_model=CVModelWrapper(model),
                          signature=signature,
                          )


# COMMAND ----------

import mlflow
from mlflow import MlflowClient


model_name = "utility_asset_accelerator"
mlflow.set_registry_uri("databricks-uc")

_register = mlflow.register_model(f"runs:/{run_id}/model", f"{CATALOG}.{SCHEMA}.{model_name}")

client = MlflowClient()
client.set_registered_model_alias(f"{CATALOG}.{SCHEMA}.{model_name}","Production", int(_register.version))


# COMMAND ----------


