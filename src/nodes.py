from typing import Dict, Any, Union
from .config import SLSConfig
from langchain_core.runnables import RunnableLambda
from .states import GraphState, StreamState
from .prompts import chat_template, context_template, rephrase_template
from .retrievers import format_documents
from .utils import (
    format_generation_user,
    format_generation_assistant,
    get_last_user_message,
)
from langchain_openai import ChatOpenAI
from databricks_langchain import ChatDatabricks
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.runnables import RunnableLambda


def make_simple_generation_node(
    model: Union[ChatOpenAI, ChatDatabricks], config: SLSConfig
):
    def simple_generation_node(state: Union[GraphState, StreamState]) -> Dict[str, Any]:
        """
        Simple generation node that does not use the prompt or context
        """
        last_msg = get_last_user_message(state)
        chain = chat_template | model | RunnableLambda(format_generation_assistant)
        response = chain.invoke(last_msg)

        if config.agent.streaming:
            response = state["messages"] + response

        return {"messages": response}

    return simple_generation_node


def make_query_vector_database_node(retriever: VectorStoreRetriever, config: SLSConfig):
    def query_vector_database_node(
        state: Union[GraphState, StreamState]
    ) -> Dict[str, Any]:
        """
        Retrieve and format the documents from the vector index.
        """
        last_msg = get_last_user_message(state)
        documents = retriever.invoke(last_msg[0]["content"])
        formatted_documents = format_documents(config, documents)
        return {"context": formatted_documents, "documents": documents}

    return query_vector_database_node


def make_context_generation_node(
    model: Union[ChatOpenAI, ChatDatabricks], config: SLSConfig
):
    def context_generation_node(
        state: Union[GraphState, StreamState]
    ) -> Dict[str, Any]:
        """
        Inject the users's question, which may have been previously rephrased
        by the model if there was message history, and the context retrieved
        from the vector index into the prompt.
        """
        chain = context_template | model | RunnableLambda(format_generation_assistant)

        last_msg = get_last_user_message(state)

        if "context" in state:
            context = state["context"]
        else:
            context = ""

        response = chain.invoke({"context": context, "question": last_msg})

        if config.agent.streaming:
            response = state["messages"] + response

        return {"messages": response}

    return context_generation_node
