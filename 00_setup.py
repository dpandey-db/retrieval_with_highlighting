# Databricks notebook source
# MAGIC %md
# MAGIC Installing dev requirements, initializing globals, adding path

# COMMAND ----------

# MAGIC %pip install -r requirements.txt --quiet
# MAGIC %restart_python

# COMMAND ----------

import sys

sys.path.append("./src")

# COMMAND ----------

# UNITY CATALOG
CATALOG = "devanshu_pandey"
SCHEMA = "retriever_agent_demo"

USERNAME = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
)

ROOT_PATH = "/".join(
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .notebookPath()
    .get()
    .split("/")[:-1]
)

# COMMAND ----------

import logging

# Configure logger
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

# COMMAND ----------

# Ensure config and schemas
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

try:
    w.catalogs.create(name=CATALOG)
except:
    log.info(f"{CATALOG} catalog exists")

try:
    w.schemas.create(catalog_name=CATALOG, name=SCHEMA)
except:
    log.info(f"{SCHEMA} catalog exists")
