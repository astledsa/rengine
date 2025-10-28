import io
import os
import fitz
import pytesseract
import numpy as np
from Classes.kvstore import KV
from models import *
from PIL import Image
from openai import OpenAI
from typing import List, Optional, Tuple
from corenn_py import CoreNN
from pytesseract import Output
from dotenv import load_dotenv

try:
    import cv2
except Exception:
    cv2 = None

load_dotenv()

class RetrievalEngine:
    
    def __init__(
        self, 
        initargs: InitEngine, 
        pdfargs: Optional[PDFExtractArgs] = None, 
        vargs: Optional[VectorIndexArgs] = None,
    ):
        if initargs.ingest_pdf and pdfargs:
            self.pdf_path: str = pdfargs.pdf_path or "./storage/data/sample.pdf"
            self.scanned: bool = pdfargs.scanned or False
            self.lang: str = pdfargs.lang or "eng"
            self.dpi: int = pdfargs.dpi or 200
            self.min_blob_area: int = pdfargs.min_blob_area or 5000
        
        if initargs.vector_index_init and vargs:
            self.corenn = CoreNN.create(
                vargs.path_to_index, 
                {
                    "dim": vargs.dimensions
                }
            )
        elif vargs:
            self.corenn = CoreNN.open(vargs.path_to_index)
        
        self.kv = KV ()
        self.openai = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        
        if initargs.kv_store_init:
            self.kv.create_table()
        
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

    def ingest(self, embed: bool = False, eargs: Optional[EmbeddingArgs] = None) -> None:
        
        keys, texts = [], []

        with fitz.open(self.pdf_path) as doc:
            for i in range(doc.page_count):
                page = doc.load_page(i)
                if self.scanned:
                    img = self._render_page(page)
                    data = self._ocr_image(img)
                    text = self._data_to_text(data)
                    # images = self._segment_non_text(img, data)
                else:
                    text = page.get_text("text")
                    # images = self._extract_embedded_images(doc, page)
                
                page_key = f"{self.pdf_path.strip('.')}_page_{i+1:04d}"
                self.kv.set(page_key, text.strip())
                if embed:
                    keys.append(page_key)
                    texts.append(text.strip())

        if embed and texts:
            response = self.openai.embeddings.create(
                model=eargs.embed_model or "text-embedding-3-small",
                dimensions=eargs.dimensions or 768,
                encoding_format=eargs.encoding_format or "float",
                input=texts
            )
            vectors = np.array([d.embedding for d in response.data], dtype=np.float32)
            self.corenn.insert_f32(keys, vectors)
        
        print(f"Indexed and stored {len(texts)} arrays")
    
    def retrieve (self, query: str, eargs: Optional[EmbeddingArgs] = None) -> List[Tuple[str, float]]:
        
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
    
    def chat (self, query: str) -> str:
        
        context: str = " ".join([t[0] for t in self.retrieve(query)])
        system: str = """
            You are an expert naval engineering AI agent. Given the context, answer the following user query.
            Do not mention anything from the context which isn't relevant, and keep the answers detailed and precise.
            
            context: {c}
            query: {q}
        """
        
        response = self.openai.responses.create(
            model="gpt-4o",
            input=system.format(
                c=context,
                q=query
            )
        )
        
        return response.output_text
        
        