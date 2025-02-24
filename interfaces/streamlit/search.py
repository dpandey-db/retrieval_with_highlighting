import streamlit as st
from pathlib import Path
import mlflow
from src.config import parse_config
from src.retrievers import get_vector_retriever
from databricks_langchain import ChatDatabricks
from langgraph.graph import StateGraph, START, END
from src.states import get_state
from src.nodes import make_query_vector_database_node, make_context_generation_node
import re


def highlight_text(text, query_terms):
    """Highlight query terms in text using HTML mark tags."""
    highlighted_text = text
    for term in query_terms:
        if term.strip():
            # Make the pattern match word boundaries to avoid partial word matches
            pattern = re.compile(f"\\b({re.escape(term)})\\b", re.IGNORECASE)
            # Use <mark> for highlighting
            highlighted_text = pattern.sub(r"<mark>\1</mark>", highlighted_text)
    return highlighted_text


def setup_agent():
    """Setup the agent and workflow"""
    root_dir = Path.cwd()
    config_path = root_dir / "agents" / "langgraph" / "config.yaml"

    mlflow_config = mlflow.models.ModelConfig(development_config=config_path)
    sls_config = parse_config(mlflow_config)

    retriever = get_vector_retriever(sls_config)
    model = ChatDatabricks(endpoint=sls_config.model.endpoint_name)

    state = get_state(sls_config)
    retriever_node = make_query_vector_database_node(retriever, sls_config)
    context_generation_node = make_context_generation_node(model, sls_config)

    workflow = StateGraph(state)
    workflow.add_node("retrieve", retriever_node)
    workflow.add_node("generate_w_context", context_generation_node)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate_w_context")
    workflow.add_edge("generate_w_context", END)

    return workflow.compile()


def main():
    st.title("Document Query Assistant")

    # Initialize session state
    if "agent" not in st.session_state:
        st.session_state.agent = setup_agent()

    # Query input
    query = st.text_input("Enter your query:")

    if query:
        # Prepare input state
        input_state = {"messages": [{"type": "user", "content": query}]}

        # Get response
        with st.spinner("Processing query..."):
            result = st.session_state.agent.invoke(input_state)

        # Display AI response
        if "messages" in result and len(result["messages"]) > 1:
            st.write("### AI Response")
            ai_response = result["messages"][-1]["content"]
            query_terms = query.lower().split()
            highlighted_response = highlight_text(ai_response, query_terms)
            # Enable HTML for mark tags
            st.markdown(highlighted_response, unsafe_allow_html=True)

        # Display retrieved documents
        if "context" in result:
            st.write("### Retrieved Documents")

            # Split query into terms for highlighting
            query_terms = query.lower().split()

            # Process each document
            for doc in result.get("documents", []):
                with st.expander(f"Document {doc.metadata.get('doc_id', 'Unknown')}"):
                    # Highlight query terms in the content
                    highlighted_content = highlight_text(doc.page_content, query_terms)

                    # Display the highlighted content
                    st.markdown(highlighted_content, unsafe_allow_html=True)

                    # Display metadata and link
                    st.markdown("**Metadata:**")
                    st.json(doc.metadata)

                    # Add a link to the original document if URI exists
                    if "doc_id" in doc.metadata:
                        st.markdown(
                            f"[View Original Document](document/{doc.metadata['doc_id']})"
                        )


if __name__ == "__main__":
    main()
