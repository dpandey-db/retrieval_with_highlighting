# Config
import mlflow
from src.config import parse_config

mlflow_config = mlflow.models.ModelConfig(development_config="./config.yaml")
sls_config = parse_config(mlflow_config)

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

mlflow.models.set_model(chain)
