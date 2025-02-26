# COMMAND ----------

import pandas as pd

df = pd.read_csv("../assets/elaws_links.csv")
df.head(3)

# COMMAND ----------

# Setup a non-OCR document converter
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.chunking import HybridChunker
from src.retrievers import make_text_chunk

# We don't need OCR for these PDFs
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False

converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)

# Setup a hybrid chunker that respects headings and hierarchy
chunker = HybridChunker(max_tokens=1000)


# COMMAND ----------

# Process each PDF
from pathlib import Path

all_chunks = []
for idx, row in df.iterrows():
    try:
        print(f"Processing {row.title}")
        result = converter.convert(Path("../") / row.asset_path)
        chunk_iter = chunker.chunk(result.document)
        doc_chunks = [make_text_chunk(x, doc_uri=row.link_to_page) for x in chunk_iter]
        all_chunks.extend(doc_chunks)
    except Exception as e:
        print(f"Error processing {row.asset_path}: {str(e)}")


# COMMAND ----------

pd.DataFrame(all_chunks)
