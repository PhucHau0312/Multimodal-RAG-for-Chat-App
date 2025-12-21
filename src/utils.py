import fitz  
import base64

from PIL import Image
import os 
import base64

from dataclasses import dataclass
from pdf2image import convert_from_path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class ImagePage:
    doc_id: str
    page_num: int
    image: Image.Image

    
def get_pdf_images(file_path):
    page_images = convert_from_path(file_path)

    image_pages = []
    for page_num, image in enumerate(page_images):
        print("Process page:", page_num)

        page = ImagePage(
            doc_id=file_path.rsplit('.', 1)[0],
            page_num=page_num,
            image=image,
        )
        image_pages.append(page)

    return image_pages


def get_text_chunks(file_path): 
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    for doc in documents:
        doc.metadata["doc_id"] = file_path.rsplit('.', 1)[0]

    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=200, length_function=len
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
