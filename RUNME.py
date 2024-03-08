# Databricks notebook source
# MAGIC %md This notebook sets up the companion cluster(s) to run the solution accelerator. It also creates the Workflow to illustrate the order of execution. Happy exploring! 
# MAGIC 🎉
# MAGIC
# MAGIC **Steps**
# MAGIC 1. Simply attach this notebook to a cluster and hit Run-All for this notebook. A multi-step job and the clusters used in the job will be created for you and hyperlinks are printed on the last block of the notebook. 
# MAGIC
# MAGIC 2. Run the accelerator notebooks: Feel free to explore the multi-step job page and **run the Workflow**, or **run the notebooks interactively** with the cluster to see how this solution accelerator executes. 
# MAGIC
# MAGIC     2a. **Run the Workflow**: Navigate to the Workflow link and hit the `Run Now` 💥. 
# MAGIC   
# MAGIC     2b. **Run the notebooks interactively**: Attach the notebook with the cluster(s) created and execute as described in the `job_json['tasks']` below.
# MAGIC
# MAGIC **Prerequisites** 
# MAGIC 1. You need to have cluster creation permissions in this workspace.
# MAGIC
# MAGIC 2. In case the environment has cluster-policies that interfere with automated deployment, you may need to manually create the cluster in accordance with the workspace cluster policy. The `job_json` definition below still provides valuable information about the configuration these series of notebooks should run with. 
# MAGIC
# MAGIC **Notes**
# MAGIC 1. The pipelines, workflows and clusters created in this script are not user-specific. Keep in mind that rerunning this script again after modification resets them for other users too.
# MAGIC
# MAGIC 2. If the job execution fails, please confirm that you have set up other environment dependencies as specified in the accelerator notebooks. Accelerators may require the user to set up additional cloud infra or secrets to manage credentials. 

# COMMAND ----------

# DBTITLE 0,Install util packages
# MAGIC %pip install git+https://github.com/databricks-academy/dbacademy@v1.0.13 git+https://github.com/databricks-industry-solutions/notebook-solution-companion@safe-print-html --quiet --disable-pip-version-check
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from solacc.companion import NotebookSolutionCompanion
nsc = NotebookSolutionCompanion()

# COMMAND ----------

# MAGIC %md
# MAGIC # Download instructions
# MAGIC Before proceding below, fill out this form to get the root url for the download link in order to download the images.
# MAGIC
# MAGIC Go [here](https://forms.office.com/pages/responsepage.aspx?id=AGTMMj0V90K8VXflpQ8bz0Fhowp3sHlPsHuc9DMJSEBURFFQTkNWVFhBVUxZRVU1NVJFU0wxR0taUS4u) and fill out the form to get the root url for downloading the zip files and place it in the root_location

# COMMAND ----------

# root_location = "https://some_dns/some_path/"
root_location = "https://publicstorageaccnt.blob.core.windows.net/drone-distribution-inspection-imagery/"

# COMMAND ----------

job_json = {
        "timeout_seconds": 28800,
        "max_concurrent_runs": 1,
        "tags": {
            "usage": "solacc_testing",
            "group": "MFG"
        },
        "tasks": [
            {
                "job_cluster_key": "mfg_data_loader",
                "notebook_task": {
                    "notebook_path": f"01_DownloadData",
                    "base_parameters": {
                        "root_location": root_location,
                        "reset_all": "True"
                    },
                },
                "task_key": "01_DownloadData"
            },
            {
                "job_cluster_key": "mfg_data_loader",
                "notebook_task": {
                    "notebook_path": f"02_CreateTables"
                },
                "task_key": "02_CreateTables",
                "depends_on": [
                    {
                        "task_key": "01_DownloadData"
                    }
                ]
            },
            {
                "job_cluster_key": "mfg_model_training",
                "notebook_task": {
                    "notebook_path": f"03_MultiGPU_ModelTraining"
                },
                "task_key": "03_MultiGPU_ModelTraining",
                "depends_on": [
                    {
                        "task_key": "02_CreateTables"
                    }
                ]
            },
            {
                "job_cluster_key": "mfg_model_training",
                "notebook_task": {
                    "notebook_path": f"04_MLInference"
                },
                "task_key": "04_MLInference",
                "depends_on": [
                    {
                        "task_key": "03_MultiGPU_ModelTraining"
                    }
                ]
            }
        ],
        "job_clusters": [
            {
                "job_cluster_key": "mfg_data_loader",
                "new_cluster": {
                    "spark_version": "13.3.x-cpu-ml-scala2.12",
                "spark_conf": {
                    "spark.master": "local[*, 4]",
                    "spark.databricks.delta.preview.enabled": "true"
                    },
                    "data_security_mode": "SINGLE_USER",
                    "num_workers": 0,
                    "node_type_id": {"AWS": "i3.xlarge", "MSA": "Standard_DS3_v2","GCP":"c2-standard-4"}, # this accelerator does not support GCP
                    "custom_tags": {
                        "usage": "solacc_testing"
                    },
                }
            },
            {
                "job_cluster_key": "mfg_model_training",
                "new_cluster": {
                    "spark_version": "13.3.x-gpu-ml-scala2.12",
                    "data_security_mode": "SINGLE_USER",
                    "num_workers": 2,
                    "node_type_id": {"AWS": "g4dn.xlarge", "MSA": "Standard_NC4as_T4_v3","GCP":"g2-standard-4"}, # this accelerator does not support GCP
                    "custom_tags": {
                        "usage": "solacc_testing"
                    },
                }
            }
        ]
    }

# COMMAND ----------

# DBTITLE 1,Deploy job and cluster
dbutils.widgets.dropdown("run_job", "False", ["True", "False"])
run_job = dbutils.widgets.get("run_job") == "True"
nsc.deploy_compute(job_json, run_job=run_job)

# COMMAND ----------


