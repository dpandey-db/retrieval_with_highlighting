# Databricks notebook source
# MAGIC %md
# MAGIC # Inference
# MAGIC
# MAGIC This module bring everything together. We take our vector store and implementation code and deploy a serving endpoint that can be used for multimodal retrieval.
# MAGIC
# MAGIC This subsection (04a) implements the inference flow using LangGraph

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
implementation_path = root_dir / "implementations" / "agents" / "langgraph"
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
model = ChatDatabricks(endpoint=sls_config.model.endpoint_name)

# Nodes
from src.states import get_state
from src.nodes import (
    make_query_vector_database_node,
    make_context_generation_node,
)

state = get_state(sls_config)
retriever_node = make_query_vector_database_node(retriever, sls_config)
context_generation_node = make_context_generation_node(model, sls_config)

# COMMAND ----------

# MAGIC %md
# MAGIC Setup the Graph

# COMMAND ----------


# Graph
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableLambda
from src.utils import graph_state_to_chat_type

workflow = StateGraph(state)
workflow.add_node("retrieve", retriever_node)
workflow.add_node("generate_w_context", context_generation_node)
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate_w_context")
workflow.add_edge("generate_w_context", END)
app = workflow.compile()

chain = app | RunnableLambda(graph_state_to_chat_type)

# COMMAND ----------

# MAGIC %md
# MAGIC Test our LangGraph Agent

# COMMAND ----------

input_example = {
    "messages": [
        {"role": "human", "content": "What is the factor of safety for fabric straps?"}
    ]
}

with mlflow.start_run():
    mlflow.langchain.autolog()
    result = chain.invoke(input_example)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log
# MAGIC Log the agent using mlflow

# COMMAND ----------

# Setup tracking and registry
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# Setup experiment
mlflow.set_experiment(f"/Users/{USERNAME}/multimodal-langgraph")

# Setup retriever schema
mlflow.models.set_retriever_schema(
    primary_key=sls_config.retriever.mapping.primary_key,
    text_column=sls_config.retriever.mapping.chunk_text,
    doc_uri=sls_config.retriever.mapping.document_uri,
)

# Signature
from mlflow.models import ModelSignature
from mlflow.types.llm import CHAT_MODEL_INPUT_SCHEMA, CHAT_MODEL_OUTPUT_SCHEMA

signature = ModelSignature(
    inputs=CHAT_MODEL_INPUT_SCHEMA, outputs=CHAT_MODEL_OUTPUT_SCHEMA
)

# Setup passthrough resources
from mlflow.models.resources import (
    DatabricksVectorSearchIndex,
    DatabricksServingEndpoint,
)

databricks_resources = [
    DatabricksServingEndpoint(endpoint_name=sls_config.model.endpoint_name),
    DatabricksVectorSearchIndex(index_name=sls_config.retriever.index_name),
]

# Get packages from requirements to set standard environments
with open("requirements.txt", "r") as file:
    packages = file.readlines()
    package_list = [pkg.strip() for pkg in packages]

# COMMAND ----------

# Log the model
with mlflow.start_run():
    logged_agent_info = mlflow.langchain.log_model(
        lc_model=str(implementation_path / "agent.py"),
        model_config=str(implementation_path / "config.yaml"),
        pip_requirements=packages,
        artifact_path="agent",
        code_paths=["maud"],
        registered_model_name=sls_config.agent.uc_model_name,
        input_example=input_example,
        signature=signature,
        resources=databricks_resources,
    )

    print(f"Model logged and registered with URI: {logged_agent_info.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC Test the reloaded model before deploying

# COMMAND ----------

reloaded = mlflow.langchain.load_model(
    f"models:/{sls_config.agent.uc_model_name}/{logged_agent_info.registered_model_version}"
)
result = reloaded.invoke(input_example)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy
# MAGIC Now we deploy the model

# COMMAND ----------

from mlflow.deployments import get_deploy_client
from databricks import agents

client = get_deploy_client("databricks")

deployment_info = agents.deploy(
    sls_config.agent.uc_model_name,
    logged_agent_info.registered_model_version,
    scale_to_zero=True,
)
