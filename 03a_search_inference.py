# Databricks notebook source
# MAGIC %md
# MAGIC # Inference
# MAGIC
# MAGIC This module bring everything together. We take our vector store and implementation code and deploy a serving endpoint that can be used for multimodal retrieval.
# MAGIC
# MAGIC This subsection (03a) uses the retriever directly to query the vector store.

# COMMAND ----------

# MAGIC %run ./00_setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config
# MAGIC Parse our config using pydantic types and validation to standardize productionized workflow

# COMMAND ----------

# load config
from pathlib import Path
import mlflow
from src.config import parse_config
import os

root_dir = Path(os.getcwd())
implementation_path = root_dir / "agents" / "search"
mlflow_config = mlflow.models.ModelConfig(
    development_config=implementation_path / "config.yaml"
)
sls_config = parse_config(mlflow_config)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chain
# MAGIC Setup our components and nodes

# COMMAND ----------

# API Interfaces
from src.retrievers import get_vector_retriever
from databricks_langchain import ChatDatabricks

retriever = get_vector_retriever(sls_config)
