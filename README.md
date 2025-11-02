# Hybrid Retrieval Augmented Generation Engine

This repository is a simple, scalable and open source RAG engine, which deals with pdfs and images. The vector index construction is handled by [coreNN](https://github.com/wilsonzlin/CoreNN/tree/master) library, full-text search by [tanvity-py](https://github.com/quickwit-oss/tantivy-py) and [sqlite](https://sqlite.org/) as a simple key-value store. The reason for using these specific libraries, apart from them being open source, is:

For a complete solution search, despite multiple options like [elasticsearch](https://github.com/elastic/elasticsearch), [chromadb](https://github.com/chroma-core/chroma), [mindsdb](https://github.com/mindsdb/mindsdb/tree/main) etc ... I wanted to have a more modular solution, where I had specific control over which full-text search and vector search libraries I can utilize. 

For not choosing full text search oriented libraries: [postgres](https://www.postgresql.org/) was relational, and I felt I needed a dedicated text search solution. [Apache solr](https://solr.apache.org/) and [pylucene](https://lucene.apache.org/pylucene/) where great solutions, but added non-trivial complexity while setting up, and forced me to deal with other languages (Java), which added an overhead. [neo4j](https://github.com/neo4j/neo4j) was a good solution, but is a more graph oriented db. As for [mongo](https://github.com/mongodb/mongo) it's lucene-backed solution (for now), was only available for it's enterprise edition. This left [pyserini](https://github.com/castorini/pyserini), which was a near perfect match, but the only problem was it offered the entire semantic search experience, including the vector indexing as well, which I specifically wanted to manage by myself. It also seems heavy, as it requires huggingface models (including `sentence-transformers`), which makes it more or less a post-MVP solution. My last choices where [xapian](https://xapian.org/) or tanvity, both of which did not support sharding. Since I would have to shard and manage them myself anyways, I chose the more powerful/expressive one.

For not choosing semantic search oriented libraries: the biggest factor was as simple as, every library utilized the HNSW index, while what I wanted was to try to use more scalable and disk backed solutions. Almost every library stuck to the trusted HNSW solution, including [faiss](https://github.com/facebookresearch/faiss), [pinecone](https://www.pinecone.io/), [weaviate](https://weaviate.io/), [postgres](https://github.com/pgvector/pgvector) etc ... Read more [here](https://blog.wilsonl.in/corenn/) for how I came upon disk-backed indexes, and chose coreNN. (_There are some [implementations/services which do offer disk based vector indexes](https://chatgpt.com/share/69053d65-1da4-8006-88f8-83f11b90dea4), but each had a caveat, while coreNN was an extremely easy to set up solution._)

As for a key value store, there no contention when it comes to light weight and easy to use, it's always SQLite.

## Usage

For now, I have mostly focused on PDFs. One running assumption I have is that as the models grow better and more powerful, hence complicated heuristics for sorting out PDFs _may_ not be necessary (or, just feed the page to an LLM or image embeddings). Here's how to initiliase the engine:

```python
rengine = RetrievalEngine(
            initargs=InitEngine(
                ingest_pdf=False,
                kv_store_init=False,
                vector_index_init=False,
                text_search_index_init=False
            ),
            pdfargs=PDFExtractArgs(
                pdf_path="./storage/data/sample.pdf",
                scanned=True
            ),
            vargs=VectorIndexArgs(
                path_to_index="./storage/index/vector",
                dimensions=768
            )
    )
```

One only has to initiliase the full-text, vector engines and the KV store once, at the start. Though the paths must always be provided for every subsequent uses.
Mentioning the paths to store vector and full-text search indexes may get complicated with sharding, where in the future, I might have to make a router of sorts. The sharding itself depends on the use-case.

**Ingestion** for PDFs

```python
rengine.ingest(
            embed=True,
                eargs=EmbeddingArgs(
                embed_model="text-embedding-3-small",
                dimensions=768,
                encoding_format="float"
            )
        )
```

**Ingestion in bulk** for PDFs

```python
 pdfs: List[singlePDFArgs] = [
    singlePDFArgs(pdf_path="./storage/data/sample_1.pdf", scanned=True),
    singlePDFArgs(pdf_path="./storage/data/sample_2.pdf", scanned=False),
    singlePDFArgs(pdf_path="./storage/data/sample_3.pdf", scanned=True),
]

rengine.ingest_bulk (pdfs)
```

**Chat**

```python
rengine.chat(
    query="A specific query, related to the docs"
)
```