# Databricks notebook source
# MAGIC %md #helpers for DS / DL 
# MAGIC
# MAGIC This notebook contains the functions required to train our model.

# COMMAND ----------

# DBTITLE 1,Image transformation for pytorch
# from functools import partial
# import torchvision
# import cv2

# # def decode_image(image_binary):
# #   return cv2.imdecode()
# # Transform our images to be ready for the ML model
# def transform_row(is_train, batch_pd):
  
#   # transform images
#   # -----------------------------------------------------------
#   # transform step 1: read incoming content value as an image
#   transformersImage = [torchvision.transforms.Lambda(lambda x: (np.moveaxis(np.array(Image.open(io.BytesIO(x)).resize((640,640))), -1, 0)).copy())]
#   # transformersMask = [torchvision.transforms.Lambda(lambda x:  np.minimum(np.expand_dims(np.array(Image.open(io.BytesIO(x)).resize((640,640))), 0),1))]
#   transformersMask = [torchvision.transforms.Lambda(lambda x:  (np.array(Image.open(io.BytesIO(x)).resize((640,640)))/51).astype(np.uint8))]

#   # transformersImage = [torchvision.transforms.Lambda(lambda x: np.array(cv2.resize(cv2.imdecode(np.frombuffer(x, dtype=np.uint8), cv2.IMREAD_COLOR),(640,640)).transpose(2,0,1),order='C'))]
#   # transformersMask = [torchvision.transforms.Lambda(lambda x: np.array(cv2.resize(cv2.imdecode(np.frombuffer(x, dtype=np.uint8), cv2.IMREAD_GRAYSCALE),(640,640)),order='C'))]
#   # assemble transformation steps into a pipeline
#   transImg = torchvision.transforms.Compose(transformersImage)
#   transMask = torchvision.transforms.Compose(transformersMask)
  
#   # print(np.minimum(np.expand_dims(np.array(Image.open(io.BytesIO(batch_pd['mask_binary'].iloc[0])).resize((640,640))), 0),1).shape)
#   # apply pipeline to images 
#   batch_pd['image_binary'] = batch_pd['image_binary'].map(lambda x: transImg(x))
  
#   # -----------------------------------------------------------
#   # transform labels (our evaluation metric expects values to be float32)
#   # -----------------------------------------------------------
#   batch_pd['mask_binary'] = batch_pd['mask_binary'].map(lambda x: transMask(x))
#   # -----------------------------------------------------------
#   return batch_pd[['image_binary', 'mask_binary']]
 
# # define function to retrieve transformation spec
# def get_transform_spec(is_train=True):
#   return TransformSpec(
#             partial(transform_row, is_train), # function to call to retrieve/transform row
#             edit_fields=[  # schema of rows returned by function
#                 ('image_binary', np.uint8, (3, IMAGE_RESIZE, IMAGE_RESIZE), False), 
#                 ('mask_binary', np.uint8, (1,IMAGE_RESIZE, IMAGE_RESIZE), False)], 
#             selected_fields=['image_binary', 'mask_binary'])

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
        # for image segmentation dice loss could be the best first choice
        # self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
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

        # Check that image dimensions are divisible by 32, 
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of 
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have 
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch["mask_binary"].long()  # Explicitly convert to LongTensor
        mask = mask.squeeze(1)
        # print("Image shape:", image.shape)  # Debug line
        # print("Mask shape:", mask.shape)    # Debug line
    # Verify that mask values are valid indices for one-hot encoding
        assert mask.min() >= 0
        assert mask.max() < self.out_classes
        mask_shape = mask.shape
        one_hot_mask = torch.zeros(mask.shape[0], self.out_classes, mask.shape[1], mask.shape[2], device=mask.device)
        one_hot_mask.scatter_(1, mask.unsqueeze(1), 1)
        # print("One-hot Mask shape:", one_hot_mask.shape)
        
        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        # assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= self.out_classes - 1 and mask.min() >= 0

        logits_mask = self.forward(image)
        # print("Logits Mask shape:", logits_mask.shape)
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
# from shapely.geometry import MultiPoint, Point
# from shapely import affinity

class CVModelWrapper(mlflow.pyfunc.PythonModel):
  # from DataLoader import SimpleOxfordPetDataset
  def __init__(self, model):
    # self.x = SimpleOxfordPetDataset(None, None)
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
  
  # def compute_bounding_polygon(self, binary_mask):
  #   """
  #   Compute bounding polygon from a binary mask.
  #   Returns coordinates of topmost, bottommost, leftmost, and rightmost points.
  #   """
  #   # Find the indices of non-zero elements (i.e., boundary or object pixels)
  #   ys, xs = torch.where(binary_mask)

  #   # Get coordinates of the topmost point
  #   top_y = torch.min(ys).item()
  #   top_x = xs[torch.argmin(ys)].item()

  #   # Get coordinates of the bottommost point
  #   bottom_y = torch.max(ys).item()
  #   bottom_x = xs[torch.argmax(ys)].item()

  #   # Get coordinates of the leftmost point
  #   left_x = torch.min(xs).item()
  #   left_y = ys[torch.argmin(xs)].item()

  #   # Get coordinates of the rightmost point
  #   right_x = torch.max(xs).item()
  #   right_y = ys[torch.argmax(xs)].item()

  #   return top_x, top_y, bottom_x, bottom_y, left_x, left_y, right_x, right_y
  
#   def compute_bounding_polygon(self, binary_mask):
#     ys, xs = torch.where(binary_mask)
#     points = list(zip(xs, ys))

#     points = MultiPoint(points)
#     # Get the convex hull around the points
#     convex_hull = points.convex_hull
#     # Get the rotated angles
#     edges = list(zip(convex_hull.boundary.coords[:-1], convex_hull.boundary.coords[1:]))
#     angles = [np.arctan2(y2 - y1, x2 - x1) for ((x1, y1), (x2, y2)) in edges]
    
#     # Iterate through each angle to find the bounding rectangle
#     min_area = float("inf")
#     best_rectangle = None
#     for angle in angles:
#         # Rotate the points to make the edge (almost) horizontal
#         rotated_points = affinity.rotate(convex_hull, angle, origin=(0, 0), use_radians=True)
#         minx, miny, maxx, maxy = rotated_points.bounds
#         width = maxx - minx
#         height = maxy - miny
#         area = width * height
#         if area < min_area:
#             min_area = area
#             # Get the 4 corners of the rectangle
#             corner1 = (minx, miny)
#             corner2 = (maxx, miny)
#             corner3 = (maxx, maxy)
#             corner4 = (minx, maxy)
#             rectangle = [corner1, corner2, corner3, corner4]
#             # Rotate the rectangle back to the original angle
#             best_rectangle = [affinity.rotate(Point([p]), -angle, origin=(0, 0), use_radians=True).coords[0] for p in rectangle]
    
#     return best_rectangle
  
#   def compute_centroid(self, points):
#     x_coords = [points[i] for i in range(0, len(points), 2)]
#     y_coords = [points[i] for i in range(1, len(points), 2)]
#     centroid_x = sum(x_coords) / len(x_coords)
#     centroid_y = sum(y_coords) / len(y_coords)
#     return centroid_x, centroid_y

# # Function to compute angle relative to centroid
#   def angle_from_centroid(self, point, centroid):
#       from math import atan2
#       return atan2(point[1] - centroid[1], point[0] - centroid[0])
    
  def predict(self, context, model_input):
    if isinstance(model_input, pd.DataFrame):
      model_input = model_input.iloc[:, 0]
    features = model_input.map(lambda x: self.prep_data(x))

    outputs = []
    for i in torch.utils.data.DataLoader(features):
      # print('looping')
      with torch.no_grad():
        o = self.model(i)
      pr_masks = o.softmax(dim=1)
      # print(pr_masks.size())
      mask = torch.argmax(pr_masks[0], dim=0)
      # print(mask.size())
      unique_values = torch.unique(mask)
      # print(unique_values)
      labels = {}
      for val in unique_values:
          # print('unique vale')
          if val == 0:  # Assuming 0 is background
              continue
          if labels.get(val.item()) is None:
            labels[val.item()] = []
          # Convert the mask for a specific class to a binary mask
          binary_mask = mask == val
          # print(binary_mask.size())
          # Label connected components
          labeled_array, num_features = label(binary_mask.numpy())
          # print(num_features)

          # Extract bounding boxes for each labeled component
          # for i in range(1, num_features + 1):
          for i in range(1, num_features + 1):
              instance_mask = torch.tensor(labeled_array == i)
              # bbox = self.compute_bounding_polygon(instance_mask)
              bbox = self.compute_bounding_box(instance_mask)
              # print(i)
              # print(bbox)
              # pairs = [(bbox[i], bbox[i + 1]) for i in range(0, len(bbox), 2)]
              # centroid = self.compute_centroid(bbox)
              # sorted_pairs = sorted(pairs, key=lambda p: self.angle_from_centroid(p, centroid))
              # sorted_points = [coord for pair in sorted_pairs for coord in pair]
              labels[val.item()].append(bbox)
              # labels[val.item()].append(sorted_points) # I just took out an extra layer of list it used to look like labels[val.item()].append([bbox])
      # print(labels)
      outputs.append(json.dumps(labels))
    return pd.Series(outputs)


