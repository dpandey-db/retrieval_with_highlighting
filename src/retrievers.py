from src.config import SLSConfig
from databricks_langchain.vectorstores import DatabricksVectorSearch
from langchain_core.vectorstores import VectorStoreRetriever


def get_vector_retriever(config: SLSConfig) -> VectorStoreRetriever:
    vector_search = DatabricksVectorSearch(
        endpoint=config.retriever.endpoint_name,
        index_name=config.retriever.index_name,
        columns=config.retriever.mapping.all_columns,
    )

    retriever = vector_search.as_retriever(
        search_kwargs={
            "k": config.retriever.parameters.k,
            "score_threshold": config.retriever.score_threshold,
            "query_type": config.retriever.parameters.query_type,
        }
    )

    return retriever


def make_text_chunk(chunk, doc_uri):
    pages = [[x.page_no for x in x.prov] for x in chunk.meta.doc_items]
    unique_pages = list(set([page for sublist in pages for page in sublist]))
    unique_pages.sort()
    doc_refs = [x.self_ref for x in chunk.meta.doc_items]
    headings = chunk.meta.headings
    captions = chunk.meta.captions

    return {
        "doc_uri": doc_uri,
        "pages": unique_pages,
        "doc_refs": doc_refs,
        "headings": headings,
        "captions": captions,
        "text": chunk.text,
        "enriched_text": f"""
            Headings:{", ".join(headings) if headings else ""}\n
            Captions:{", ".join(captions) if captions else ""}\n
            Text:{chunk.text}
        """,
    }


def format_documents(config: SLSConfig, docs):
    chunk_template = config.retriever.chunk_template
    chunk_contents = [
        chunk_template.format(
            chunk_text=d.page_content,
            document_uri=d.metadata[config.retriever.mapping.document_uri],
        )
        for d in docs
    ]
    return "".join(chunk_contents)


def index_exists(client, vs_endpoint, index_name):
    try:
        client.get_index(vs_endpoint, index_name)
        return True
    except Exception as e:
        if "IndexNotFoundException" in str(e):
            return False
        else:
            raise e
