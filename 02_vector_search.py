# Databricks notebook source
# MAGIC %md
# MAGIC # Featurize
# MAGIC
# MAGIC This notebook featurizes the vector search index. We have manually downloaded sample data from our legislation.

# COMMAND ----------

# MAGIC %run ./00_setup

# COMMAND ----------

import pyspark.sql.functions as F

# COMMAND ----------

display(spark.table("devanshu_pandey.retriever_agent_demo.elaws_sample_data_chunked"))

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE devanshu_pandey.retriever_agent_demo.elaws_sample_data_chunked
# MAGIC SET TBLPROPERTIES (delta.enableChangeDataFeed = true)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Vector Search Index
# MAGIC Now that we have a single table for vector search, let's load it into Databricks Vector Search. There is more we can do here, but for now we simply want to

# COMMAND ----------

from mlflow.models import ModelConfig
from databricks.vector_search.client import VectorSearchClient
from src.retrievers import index_exists

config = ModelConfig(development_config="agents/search/config.yaml")
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
