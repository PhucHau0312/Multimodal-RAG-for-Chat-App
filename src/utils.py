import os 
import io
import base64

import fitz  
import hashlib
import numpy as np

from typing import Callable, List, Optional, Tuple, Union
from pathlib import Path
from PIL import Image

from dataclasses import dataclass
from pdf2image import convert_from_path
from rapidocr_onnxruntime import RapidOCR

from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class ImagePage:
    doc_id: str
    page_num: int
    image: Image.Image


def _norm(s: str) -> str:
        return " ".join((s or "").lower().split())


def make_rapidocr(min_score: float = 0.5):
    engine = RapidOCR()

    def ocr(image_bytes: bytes) -> str:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.array(img)
        result, _ = engine(arr)
        if not result:
            return ""
        lines = []
        for item in result:
            text = (item[1] or "").strip()
            score = float(item[2]) if len(item) > 2 else 1.0
            if text and score >= min_score:
                lines.append(text)
        return "\n".join(lines).strip()

    return ocr

    
def get_pdf_images(file_path):
    page_images = convert_from_path(file_path)

    image_pages = []
    for page_num, image in enumerate(page_images):
        print("Process page:", page_num)

        page = ImagePage(
            doc_id=Path(file_path).stem,
            page_num=page_num,
            image=image,
        )
        image_pages.append(page)

    return image_pages


def get_text_chunks(file_path): 
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    for doc in documents:
        doc.metadata["doc_id"] = Path(file_path).stem

    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=180, length_function=len
        )
    chunks = text_splitter.split_documents(documents)

    return chunks


def get_pdf_page_as_base64(pdf_path, page_number):
    doc = fitz.open(pdf_path)
    
    try:
        page = doc.load_page(page_number)
        
        zoom = 2.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        img_bytes = pix.tobytes("png")   
        base64_string = base64.b64encode(img_bytes).decode('utf-8')
        
        return base64_string
        
    except IndexError:
        return f"Error: Page {page_number} not found."
    finally:
        doc.close()


def complex_get_text_chunks(
    file_path: str,
    ocr_fn: Callable[[bytes], str] = make_rapidocr(),
    ocr_only_if_raw_is_shorter_than: int = 100,
    min_image_area_ratio: float = 0.03,
    dedupe_ocr_if_in_raw: bool = True,
    max_images_per_page: Optional[int] = None,
    raw_chunk_size: int = 800,
    raw_chunk_overlap: int = 180,
    ocr_chunk_size: int = 300,
    ocr_chunk_overlap: int = 80,
    ):
    print("Process file path:", file_path)

    page_docs = PyMuPDFLoader(file_path, mode="page").load()
    for d in page_docs:
        d.metadata["doc_id"] = Path(file_path).stem
        d.metadata["text_source"] = "raw"

    raw_by_page = {int(d.metadata.get("page", i)): (d.page_content or "") for i, d in enumerate(page_docs)}

    ocr_docs = []
    ocr_cache = {}

    doc = fitz.open(file_path)
    try:
        for page_idx in range(doc.page_count):
            raw_text = (raw_by_page.get(page_idx, "") or "").strip()

            if len(raw_text) >= ocr_only_if_raw_is_shorter_than:
                print("Raw texts!")
                continue

            print("Using OCR!")

            page_ocr_texts = []
            page_images_meta = []

            page = doc.load_page(page_idx)
            page_area = float(page.rect.width * page.rect.height) or 1.0

            dct = page.get_text("dict")
            img_blocks = [b for b in dct.get("blocks", []) if b.get("type") == 1]

            if max_images_per_page is not None:
                img_blocks = img_blocks[:max_images_per_page]

            fallback_imgs = page.get_images(full=True)

            for img_i, b in enumerate(img_blocks):
                bbox = b.get("bbox")
                if not bbox:
                    continue

                x0, y0, x1, y1 = bbox
                img_area = max(0.0, (x1 - x0)) * max(0.0, (y1 - y0))
                if (img_area / page_area) < min_image_area_ratio:
                    continue

                xref = b.get("xref")
                if not xref:
                    if img_i < len(fallback_imgs):
                        xref = fallback_imgs[img_i][0]
                    else:
                        continue

                extracted = doc.extract_image(int(xref))
                image_bytes = extracted.get("image")
                if not image_bytes:
                    continue

                h = hashlib.sha1(image_bytes).hexdigest()
                if h in ocr_cache:
                    ocr_text = ocr_cache[h]
                else:
                    ocr_text = (ocr_fn(image_bytes) or "").strip()
                    ocr_cache[h] = ocr_text

                if not ocr_text:
                    continue

                if dedupe_ocr_if_in_raw:
                    if _norm(ocr_text) and _norm(ocr_text) in _norm(raw_text):
                        continue

                page_ocr_texts.append(ocr_text)
                page_images_meta.append(
                    {
                        "image_xref": int(xref),
                        "image_index_on_page": img_i,
                        "image_bbox": bbox
                    }
                )
            merged_ocr_page = "\n\n".join(page_ocr_texts).strip()

            ocr_docs.append(
                Document(
                    page_content=merged_ocr_page,
                    metadata={
                        "source": Path(file_path).name,
                        "doc_id": Path(file_path).stem,
                        "page": page_idx,
                        "text_source": "ocr",
                        "images": page_images_meta
                    },
                )
            )

    finally:
        doc.close()

    raw_splitter = RecursiveCharacterTextSplitter(
        chunk_size=raw_chunk_size,
        chunk_overlap=raw_chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    ocr_splitter = RecursiveCharacterTextSplitter(
        chunk_size=ocr_chunk_size,
        chunk_overlap=ocr_chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )

    raw_chunks = raw_splitter.split_documents([d for d in page_docs if (d.page_content or "").strip()])
    ocr_chunks = ocr_splitter.split_documents([d for d in ocr_docs if (d.page_content or "").strip()])

    return raw_chunks + ocr_chunks