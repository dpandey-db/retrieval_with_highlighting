# Databricks notebook source
# MAGIC %md
# MAGIC # Featurize
# MAGIC
# MAGIC This notebook featurizes the vector search index

# COMMAND ----------

# MAGIC %run ./00_setup

# COMMAND ----------

import pyspark.sql.functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract Metadata and Chunks
# MAGIC Here we extract the metadata and chunks so that we have a single vector index

# COMMAND ----------

(
    combined.write.mode("overwrite")
    .options(mergeSchema=True)
    .saveAsTable("shm.multimodal.combined_chunks")
)
display(combined)

# COMMAND ----------

display(spark.table("shm.multimodal.combined_chunks"))

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE shm.multimodal.combined_chunks
# MAGIC SET TBLPROPERTIES (delta.enableChangeDataFeed = true)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Vector Search Index
# MAGIC Now that we have a single table for vector search, let's load it into Databricks Vector Search. There is more we can do here, but for now we simply want to

# COMMAND ----------

from mlflow.models import ModelConfig
from databricks.vector_search.client import VectorSearchClient
from agent.retrievers import index_exists

config = ModelConfig(development_config="agents/langgraph/config.yaml")
vs_config = config.get("vector_search")
vs_endpoint = vs_config.get("endpoint_name")
vs_index_name = vs_config.get("index_name")
vs_source_table = vs_config.get("combined_chunks_table")
client = VectorSearchClient()

# COMMAND ----------

# MAGIC %md
# MAGIC If the index exists already, we will run a sync on the table. If not, we will use the SDK to create the index

# COMMAND ----------

if index_exists(client, vs_endpoint, vs_index_name):
    index = client.get_index(vs_endpoint, vs_index_name)
    index.sync()
else:
    index = client.create_delta_sync_index(
        endpoint_name=vs_endpoint,
        source_table_name=vs_source_table,
        index_name=vs_index_name,
        pipeline_type="TRIGGERED",
        primary_key="id",
        embedding_source_column="text",
        embedding_model_endpoint_name="databricks-gte-large-en",
        columns_to_sync=["filename", "pages", "type", "ref", "img_path"],
    )
