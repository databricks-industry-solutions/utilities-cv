# Databricks notebook source
# MAGIC %md #helpers for DS / DL 
# MAGIC
# MAGIC This notebook contains the functions required to train our model.

# COMMAND ----------

# DBTITLE 1,Lightning Model
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader

class UtilityAssetModel(pl.LightningModule):

    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        self.out_classes = out_classes
        self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        
        image = batch["image_binary"]

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch["mask_binary"].long()  # Explicitly convert to LongTensor
        mask = mask.squeeze(1)
    # Verify that mask values are valid indices for one-hot encoding
        assert mask.min() >= 0
        assert mask.max() < self.out_classes
        mask_shape = mask.shape
        one_hot_mask = torch.zeros(mask.shape[0], self.out_classes, mask.shape[1], mask.shape[2], device=mask.device)
        one_hot_mask.scatter_(1, mask.unsqueeze(1), 1)
        

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= self.out_classes - 1 and mask.min() >= 0

        logits_mask = self.forward(image)
        assert logits_mask.shape == one_hot_mask.shape
        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then 
        # apply thresholding
        prob_mask = logits_mask.softmax(dim=1)
        pred_mask = torch.argmax(prob_mask, dim=1)

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        # tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="multiclass",num_classes=self.out_classes)

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image 
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        
        # aggregate and compute the average loss
        losses = torch.stack([x["loss"] for x in outputs])  # Collect all loss values
        avg_loss = losses.mean()  # Compute the average loss

        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset 
        # with "empty" images (images without target class) a large gap could be observed. 
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
            f"{stage}_avg_loss": avg_loss,
            'current_epoch': self.current_epoch,
        }
        
        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")            

    # def on_train_epoch_end(self, outputs):
    #     return self.shared_epoch_end(outputs, "train")
    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")
    # def on_validation_epoch_end(self, outputs):
    #     return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")  

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Custom model wrapper for MLFlow
# MAGIC
# MAGIC While we could save our model using an MLFlow pytorch flavor, we add a custom wrapper around it to make inference easier.
# MAGIC
# MAGIC The wrapper is flexible: it allows input as base64 or binary, will resize the image to the expected size and run the model inference.
# MAGIC
# MAGIC Base64 will be useful for our interactive real-time demo inference

# COMMAND ----------

import mlflow
import pandas as pd
import base64
from PIL import Image
import torch
from scipy.ndimage import label
import json
import io
import numpy as np

class CVModelWrapper(mlflow.pyfunc.PythonModel):
  def __init__(self, model):
    self.model = model.eval()

  def prep_data(self, data):
    if isinstance(data, str):
      self.output_base64 = True
      data = base64.b64decode(data)
    return torch.tensor(np.moveaxis(np.array(Image.open(io.BytesIO(data)).resize((640,640)).convert("RGB")), -1, 0))
  
  def compute_bounding_box(self, binary_mask):
    """
    Compute bounding box from a binary mask.
    Returns top-left and bottom-right coordinates.
    """
    rows = torch.any(binary_mask, dim=1)
    cols = torch.any(binary_mask, dim=0)
    ymin, ymax = torch.where(rows)[0][[0, -1]]
    xmin, xmax = torch.where(cols)[0][[0, -1]]
    return xmin.item(), ymin.item(), xmax.item(), ymax.item()
  
  
    
  def predict(self, context, model_input):
    if isinstance(model_input, pd.DataFrame):
      model_input = model_input.iloc[:, 0]
    features = model_input.map(lambda x: self.prep_data(x))

    outputs = []
    for i in torch.utils.data.DataLoader(features):
      with torch.no_grad():
        o = self.model(i)
      pr_masks = o.softmax(dim=1)
      mask = torch.argmax(pr_masks[0], dim=0)
      unique_values = torch.unique(mask)
      labels = {}
      for val in unique_values:
          if val == 0:  # Assuming 0 is background
              continue
          if labels.get(val.item()) is None:
            labels[val.item()] = []
          # Convert the mask for a specific class to a binary mask
          binary_mask = mask == val
          # Label connected components
          labeled_array, num_features = label(binary_mask.numpy())

          # Extract bounding boxes for each labeled component
          for i in range(1, num_features + 1):
              instance_mask = torch.tensor(labeled_array == i)
              bbox = self.compute_bounding_box(instance_mask)
              labels[val.item()].append(bbox)
      outputs.append(json.dumps(labels))
    return pd.Series(outputs)


