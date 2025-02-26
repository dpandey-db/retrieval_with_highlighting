import re
import requests
from pathlib import Path
import streamlit as st
from databricks.sdk import WorkspaceClient
from nltk.stem import PorterStemmer

from src.interface import load_interface_config

stemmer = PorterStemmer()

workspace_client = WorkspaceClient()
workspace_url = workspace_client.config.host
token = workspace_client.config.token

config = load_interface_config(str(Path(__file__).parent / "config.yaml"))


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


def main():
    st.title(config.title)
    st.write(config.description)

    # Query input
    query = st.text_input("Enter your query:")

    endpoint_url = (
        f"{workspace_url}/serving-endpoints/{config.serving_endpoint}/invocations"
    )

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    payload = {
        "messages": [
            {"role": "user", "content": query},
        ]
    }

    if query:
        # Make direct request to the endpoint
        with st.spinner("Processing query..."):
            try:
                response = requests.post(endpoint_url, headers=headers, json=payload)
                response.raise_for_status()  # Raise an exception for bad status codes
                result = response.json()

                st.write("### Assistant")
                ai_response = result["choices"][0]["message"]["content"]
                query_terms = query.lower().split()
                highlighted_response = highlight_stemmed_text(ai_response, query_terms)
                st.markdown(highlighted_response, unsafe_allow_html=True)

                # Display retrieved documents
                if "documents" in result["custom_outputs"]:
                    st.write("### Retrieved Documents")
                    query_terms = query.lower().split()

                    for doc in result["custom_outputs"].get("documents", []):
                        metadata = doc.get("metadata", {})
                        with st.expander(f"Document {metadata.get('doc_id', '')}"):
                            highlighted_content = highlight_stemmed_text(
                                doc["page_content"], query_terms
                            )
                            st.markdown(highlighted_content, unsafe_allow_html=True)

                            # Display metadata
                            st.markdown("**Metadata:**")
                            st.json(doc.get("metadata", {}))

                            # Add document link if doc_id exists
                            if "doc_id" in doc:
                                st.markdown(
                                    f"[View Original Document](document/{metadata.get("url","")})"
                                )

            except requests.exceptions.RequestException as e:
                st.error(f"Error making request: {str(e)}")


if __name__ == "__main__":
    main()
