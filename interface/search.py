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

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()


def simple_tokenize(text):
    """Simple word tokenizer that splits on whitespace and preserves punctuation"""
    return re.findall(r"\b\w+\b|[^\w\s]|\s+", text)


def stem_words(text):
    """Stem words using provided stemmer"""
    words = simple_tokenize(text.lower())
    return [stemmer.stem(word) for word in words if re.match(r"\b\w+\b", word)]


def highlight_stemmed_text(text, query_terms):
    """Highlight query terms in text using HTML mark tags, accounting for word stems."""
    # Get stems of the query terms
    stemmed_query_terms = set(stem_words(" ".join(query_terms)))

    # Split the text into words while preserving spaces and punctuation
    words = simple_tokenize(text)

    # Process each word and rebuild the text
    result = []
    for word in words:
        if re.match(r"\b\w+\b", word):  # If it's a word (not space/punctuation)
            stemmed_word = stemmer.stem(word.lower())
            if stemmed_word in stemmed_query_terms:
                result.append(f"<mark>{word}</mark>")
            else:
                result.append(word)
        else:
            result.append(word)  # Preserve spaces and punctuation

    return "".join(result)


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
            highlighted_response = highlight_stemmed_text(ai_response, query_terms)
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
                    highlighted_content = highlight_stemmed_text(
                        doc.page_content, query_terms
                    )

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
