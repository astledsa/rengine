from typing import List, Literal, Set
from pydantic import BaseModel

class InitEngine (BaseModel):
    ingest_pdf: bool
    kv_store_init: bool
    vector_index_init: bool
    text_search_index_init: bool

class PDFExtractArgs (BaseModel):
    pdf_path: str = ""
    scanned: bool = False
    out_dir: str = "output"
    lang: str = "eng"
    dpi: int = 200
    min_blob_area: int = 5000

class VectorIndexArgs (BaseModel):
    path_to_index: str = "./storage/index/vector"
    dimensions: int = 768

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
    