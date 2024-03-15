![image](https://raw.githubusercontent.com/databricks-industry-solutions/.github/main/profile/solacc_logo_wide.png)

[![CLOUD](https://img.shields.io/badge/CLOUD-ALL-blue?logo=googlecloud&style=for-the-badge)](https://cloud.google.com/databricks)
[![POC](https://img.shields.io/badge/POC-10_days-green?style=for-the-badge)](https://databricks.com/try-databricks)

%md
# Analyzing Electrical Grid Assets Using Computer Vision

Utilities can have millions of individual distribution assets in circulation to help keep their grid operation smoothly. Many times these assets are out of sync with their digital versions on systems like GIS, or they deteriorate over time and become damaged. Drones have become the defacto way to take imagery of these assets, but it is a daunting task to manually review this imagery to correct GIS or identify assets that need repaired or replaced. These activities need to be automated through a subfield of machine learning called computer vision. This involved teaching a machine how to look at images and be able to identify items in that image.

In this accelerator we will explore how to use Databricks for computer vision use cases that involve drone imagery and power distribution assets. This framework can easily be substitued with transmission assets, or solar generation assets. Addtionally we can extend this accelerator by cropping identified assets in order to apply other fine tuned models for things like damage detection.

## Notebook Outline
**01_DownloadData** will create some [UnityCatalog](https://www.databricks.com/product/unity-catalog) assets for us to download and store the images and labels needed to train our computer vision model

**02_CreateTables** will manipulate our downloaded data so that labels will be applied to images for teaching the computer how to identify these assets

**03_MultiGPU_ModelTraining** will utilize our table data to train our model through multiple rounds of training and testing. This will be done on more than one machine to make it faster

**04_MLInference** will take the model trained in the previous notebook and look at a never before seen drone image to see if it can identify the assets properly
%md

&copy; 2024 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.
</p>

| library                                | description             | license    | source                                              |
|----------------------------------------|-------------------------|------------|-----------------------------------------------------|
| pytorch-lightning  | lightweight PyTorch wrapper for ML researchers | Apache Software License (Apache-2.0)   | https://pypi.org/project/pytorch-lightning/   |
| opencv-python  | Wrapper package for OpenCV python bindings| Apache Software License (Apache 2.0)   | https://pypi.org/project/opencv-python/   |
| segmentation-models-pytorch  | Image segmentation models with pre-trained backbones. PyTorch. | MIT License (MIT)    | https://pypi.org/project/segmentation-models-pytorch/  |
| shapely  |Manipulation and analysis of geometric objects |BSD License (BSD 3-Clause)    | https://pypi.org/project/shapely/   |

## Getting started

Although specific solutions can be downloaded as .dbc archives from our websites, we recommend cloning these repositories onto your databricks environment. Not only will you get access to latest code, but you will be part of a community of experts driving industry best practices and re-usable solutions, influencing our respective industries. 

<img width="500" alt="add_repo" src="https://user-images.githubusercontent.com/4445837/177207338-65135b10-8ccc-4d17-be21-09416c861a76.png">

To start using a solution accelerator in Databricks simply follow these steps: 

1. Clone solution accelerator repository in Databricks using [Databricks Repos](https://www.databricks.com/product/repos)
2. Attach the `RUNME` notebook to any cluster and execute the notebook via Run-All. A multi-step-job describing the accelerator pipeline will be created, and the link will be provided. The job configuration is written in the RUNME notebook in json format. 
3. Execute the multi-step-job to see how the pipeline runs. 
4. You might want to modify the samples in the solution accelerator to your need, collaborate with other users and run the code samples against your own data. To do so start by changing the Git remote of your repository  to your organization’s repository vs using our samples repository (learn more). You can now commit and push code, collaborate with other user’s via Git and follow your organization’s processes for code development.

The cost associated with running the accelerator is the user's responsibility.


## Project support 

Please note the code in this project is provided for your exploration only, and are not formally supported by Databricks with Service Level Agreements (SLAs). They are provided AS-IS and we do not make any guarantees of any kind. Please do not submit a support ticket relating to any issues arising from the use of these projects. The source in this project is provided subject to the Databricks [License](./LICENSE). All included or referenced third party libraries are subject to the licenses set forth below.

Any issues discovered through the use of this project should be filed as GitHub Issues on the Repo. They will be reviewed as time permits, but there are no formal SLAs for support. 
