from typing import Callable, List, Literal, Set
from pydantic import BaseModel

class InitEngine (BaseModel):
    ingest_pdf: bool
    kv_store_init: bool
    vector_index_init: bool
    text_search_index_init: bool
    vector_index_path: str
    text_index_path: str
    kv_store_path: str

class singlePDFArgs (BaseModel):
    pdf_path: str
    scanned: bool

class EmbeddingArgs (BaseModel):
    embed_model: Literal[
        "text-embedding-3-small",
        "text-embedding-3-large", 
        "text-embedding-ada-002"
    ]
    dimensions: Literal[
        256, 
        768, 
        1536
    ]
    encoding_format: Literal["float", "base64"]

class ProcessedQuery (BaseModel):
    raw: List[str]
    autocorrected: str
    expanded_terms: Set[str]
    primary_keywords: List[str]

