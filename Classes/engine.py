import io
import os
import fitz
import tantivy
import pytesseract
import numpy as np
from PIL import Image
from openai import OpenAI
from utils.models import *
from corenn_py import CoreNN
from Classes.kvstore import KV
from pytesseract import Output
from dotenv import load_dotenv
from utils.query_processor import *
from typing import List, Optional, Tuple
from semantic_text_splitter import TextSplitter as SemSplitter

try:
    import cv2
except Exception:
    cv2 = None

load_dotenv()

_defaultembedargs = EmbeddingArgs(
    embed_model="text-embedding-3-small",
    dimensions=768,
    encoding_format="float"
)

def is_valid_paragraph(text: str, min_words: int = 40, min_chars: int = 200) -> bool:
    if not text:
        return False
    clean = text.strip()
    if len(clean) < min_chars:
        return False
    if len(clean.split()) < min_words:
        return False
    return True

class RetrievalEngine:
    
    def __init__(
        self, 
        initargs: InitEngine, 
    ):
        self.kv = KV (path=initargs.kv_store_path)
        self.openai = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        
        if initargs.text_search_index_init:
            schema_builder = tantivy.SchemaBuilder()
            schema_builder.add_text_field("content", stored=True, tokenizer_name="en_stem")
            schema = schema_builder.build()
            self.tanv = tantivy.Index(schema, path=initargs.text_index_path)
            print("Initialsed text index")
        else:
            self.tanv = tantivy.Index.open(path=initargs.text_index_path)
            
        if initargs.ingest_pdf :
            self.lang: str = "eng"
            self.dpi: int = 200
            self.min_blob_area: int = 5000
        
        if initargs.vector_index_init:
            self.corenn = CoreNN.create(
                initargs.vector_index_path, 
                {
                    "dim": 768
                }
            )
            print("Initialsed vector index")
        else :
            self.corenn = CoreNN.open(initargs.vector_index_path)
        
        if initargs.kv_store_init:
            self.kv.create_table()
            print("Initialsed kv store")
        
    def _render_page(self, page: fitz.Page) -> Image.Image:
        pix = page.get_pixmap(dpi=self.dpi, alpha=False)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    def _ocr_image(self, img: Image.Image):
        return pytesseract.image_to_data(img, lang=self.lang, output_type=Output.DICT)

    def _data_to_text(self, data: dict) -> str:
        parts = []
        for i in range(len(data["text"])):
            t = (data["text"][i] or "").strip()
            if t:
                parts.append(t)
        return " ".join(parts).strip()

    def _segment_non_text(self, img: Image.Image, data: dict) -> list:
        if cv2 is None:
            return [{"name": "full", "image": img}]
        
        h, w = img.height, img.width
        mask = np.zeros((h, w), dtype=np.uint8)
        for i in range(len(data["text"])):
            t = (data["text"][i] or "").strip()
            if not t:
                continue
            x, y, ww, hh = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            if ww > 0 and hh > 0:
                cv2.rectangle(mask, (x, y), (x + ww, y + hh), 255, -1)
        inv = cv2.bitwise_not(mask)
        contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cvimg = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        images = [{"name": "full", "image": Image.fromarray(cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB))}]
        k = 1
        for c in contours:
            x, y, ww, hh = cv2.boundingRect(c)
            if ww * hh < self.min_blob_area:
                continue
            crop = cvimg[y:y + hh, x:x + ww]
            crop_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            images.append({"name": f"fig_{k:02d}", "image": crop_img})
            k += 1
        return images

    def _extract_embedded_images(self, doc: fitz.Document, page: fitz.Page) -> list:
        paths = []
        imgs = page.get_images(full=True)
        k = 1
        for info in imgs:
            xref = info[0]
            base = doc.extract_image(xref)
            img_bytes = base["image"]
            img = Image.open(io.BytesIO(img_bytes))
            paths.append({"name": f"img_{k:02d}", "image": img})
            k += 1
        return paths

    def ingest(
            self, 
            pargs: singlePDFArgs, 
            embed: bool = False, 
            eargs: EmbeddingArgs = _defaultembedargs
        ) -> None:
            
            keys, texts = [], []
            writer = self.tanv.writer()
            sem_splitter = SemSplitter (400)

            with fitz.open(pargs.pdf_path) as doc:
                for i in range(doc.page_count):
                    try:
                        try:
                            page = doc.load_page(i)
                        except Exception as e:
                            print(f"Failed to load page {i+1}: {e}")
                            continue
                        
                        if pargs.scanned:
                            img = self._render_page(page)
                            data = self._ocr_image(img)
                            text = self._data_to_text(data)
                        else:
                            try:
                                text = page.get_text("text") or ""
                            except Exception:
                                text = ""
                            
                            if not text.strip():
                                img = self._render_page(page)
                                data = self._ocr_image(img)
                                text = self._data_to_text(data)

                        page_key = f"{pargs.pdf_path.rstrip('.pdf')}_page_{i+1:04d}"
                        
                        clean_text = text.strip()
                        if clean_text and is_valid_paragraph(clean_text):
                            writer.add_document(tantivy.Document(content=clean_text))

                        if embed and clean_text:
                            if len(clean_text) < 1000:
                                keys.append(page_key)
                                texts.append(clean_text)
                                self.kv.set(page_key, clean_text)
                            else:
                                for j, f in enumerate(sem_splitter.chunks(clean_text)):
                                    keys.append(f"{page_key}_part_{j}")
                                    texts.append(f)
                                    self.kv.set(f"{page_key}_part_{j}", f)

                    except Exception as e:
                        print(f"Error on page {i+1}:", e)
            
            writer.commit()
            writer.wait_merging_threads()
            
            print(f"Indexed {len(texts)} chunks: text")
            
            if embed and texts:
                def est_tokens(s): return max(1, len(s) // 4)
                
                token_limit, batch_size = 30_000, 2000
                batch, batch_keys, batch_tokens = [], [], 0
                all_keys = list()
                for k, t in zip(keys, texts):
                    t_tokens = est_tokens(t)
                    if batch and (batch_tokens + t_tokens > token_limit or len(batch) >= batch_size):
                        response = self.openai.embeddings.create(
                            model=eargs.embed_model, 
                            dimensions=eargs.dimensions, 
                            encoding_format=eargs.encoding_format, 
                            input=batch
                        )
                        vectors = np.array([d.embedding for d in response.data], dtype=np.float32)
                        self.corenn.insert_f32(batch_keys, vectors)
                        all_keys.extend(batch_keys)
                        batch, batch_keys, batch_tokens = [], [], 0

                    batch.append(t)
                    batch_keys.append(k)
                    batch_tokens += t_tokens
                if batch:
                    response = self.openai.embeddings.create(
                        model=eargs.embed_model, 
                        dimensions=eargs.dimensions, 
                        encoding_format=eargs.encoding_format,
                        input=batch
                    )
                    vectors = np.array([d.embedding for d in response.data], dtype=np.float32)
                    self.corenn.insert_f32(batch_keys, vectors)
                    all_keys.extend(batch_keys)
                    
                print(f"Indexed {len(keys)} chunks: semantic")

    def ingest_bulk (self, data: List[singlePDFArgs]) -> None:
        
        for arg in data:
            self.ingest(pargs=arg, embed=True, eargs=EmbeddingArgs(
                embed_model="text-embedding-3-small",
                dimensions=768,
                encoding_format="float"
            ))
            print("Done with a PDF")
    
    def FullTextRetrieve (self, query: str) -> List[Tuple[str, float]]:
        
        searcher = self.tanv.searcher()
        squery: tantivy.Query = build_query(self.tanv, structure_query(query))
        best_hits = searcher.search(squery, 25).hits
        
        results: List[Tuple[str, float]] = []
        for (score, best_doc_address) in best_hits:
            best_doc = searcher.doc(best_doc_address)
            results.append((best_doc["content"][0], score))
        
        return results
    
    def SemanticRetrieve (self, query: str, eargs: Optional[EmbeddingArgs] = None) -> List[Tuple[str, float]]:
        
        res = self.openai.embeddings.create(
                model=eargs.embed_model if eargs else "text-embedding-3-small",
                dimensions=eargs.dimensions if eargs else 768,
                encoding_format=eargs.encoding_format if eargs else "float",
                input=query
            )
        
        return self.corenn.query_f32(
            np.array(
                [d.embedding for d in res.data],
                dtype=np.float32
            ),
            25
        )[0]
    
    def chat (self, query: str, stream: bool = False) -> str:
        
        results_keyword: Set[str] = set([t[0] for t in self.FullTextRetrieve(query)])
        results_semantic: Set[str] = set([self.kv.get(t[0]) for t in self.SemanticRetrieve(query)])
        
        common = results_keyword & results_semantic
        keyword_match = results_keyword - results_semantic
        semantic_match = results_semantic - results_keyword
        
        context: str = f"""
            common: {common}\n
            semantic match: {semantic_match}\n
            keyword match: {keyword_match}
        """
        
        system: str = """
            You are an expert naval engineering AI agent. Use only the information explicitly provided 
            in the context to answer the following user query.
            Your response must be detailed, technically accurate, and fully grounded in the context.
            Do not include any information, examples, or explanations that are not directly supported by 
            the given context.
            If the context does not contain sufficient information to answer fully, acknowledge the gap 
            clearly instead of speculating. DO NOT MENTION the context, simply answer.
            
            context: {c}\n
            query: {q}
        """
        
        if not stream:
            response = self.openai.responses.create(
                model="gpt-5-mini-2025-08-07",
                input=system.format(
                    c=context,
                    q=query
                )
            )
            return response.output_text
        else:
            with self.openai.responses.stream(
                model="gpt-5-mini-2025-08-07",
                input=system.format(
                    c=context,
                    q=query
                )
            ) as stream:
                output = []
                for event in stream:
                    if event.type == "response.output_text.delta":
                        print(event.delta, end="", flush=True)
                        output.append(event.delta)
                    elif event.type == "response.completed":
                        print("\n---\n[Done]")
                return "".join(output)