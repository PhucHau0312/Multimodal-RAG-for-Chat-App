import torch
import os
import weaviate

from torch.utils.data import DataLoader
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.utils import get_pdf_images, get_text_chunks


VISUAL_CLIENT = os.getenv("VISUAL_CLIENT")
TEXTUAL_CLIENT = os.getenv("TEXTUAL_CLIENT")
VECTOR_DIMENSION = 128


def create_visual_vectordb(embedding, collection_name, data_dir="docs"):

    image_pages = []
    for entry in os.listdir(data_dir):
        full_path = os.path.join(data_dir, entry)
        image_pages.extend(get_pdf_images(full_path))
    
    batch_size = 32
    total_batches = (len(image_pages) + batch_size - 1) // batch_size
    dataloader = DataLoader(image_pages, batch_size=batch_size, shuffle=False,
                            collate_fn=lambda x: embedding.vision_processor.process_images([page.image for page in x]))

    all_embeddings = []
    for batch_doc in tqdm(dataloader, desc="Building embeddings", total=total_batches):
        with torch.no_grad():
            batch_doc = {
                k: v.to(embedding.vision_model.device)
                for k, v in batch_doc.items()
            }

            with torch.cuda.device(embedding.vision_model.device):
                torch.cuda.empty_cache()

            embeddings_doc = embedding.vision_model(**batch_doc)
            embeddings_doc = embeddings_doc.cpu().float().numpy().tolist()
            all_embeddings.extend(embeddings_doc)

    print(f"Num embeddings: {len(all_embeddings)}")

    def point_generator():
        for i, (doc_page, embedding) in enumerate(zip(image_pages, all_embeddings)):
            yield models.PointStruct(
                id=i,
                vector={
                    "embedding": embedding  # Named vector for multi-vector config
                },
                payload={"file_name": f"{doc_page.doc_id}.pdf",
                          "doc_id": doc_page.doc_id,
                          "page_num": doc_page.page_num}
            )
    try:
        client = QdrantClient(url=VISUAL_CLIENT)

        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "embedding": models.VectorParams(
                        size=VECTOR_DIMENSION,
                        distance=models.Distance.COSINE,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM
                        ),
                    )
                }
            )
        client.upload_points(
            collection_name=collection_name,
            points=point_generator(),
            batch_size=32,
            parallel=4,
            wait=True
        )
        print("✅ Creating process complete!")
    except Exception as e:
        print(f"❌ Error creating index: {e}")


def create_textual_vectordb(embedding, data_dir="docs"):

    all_chunks = []
    for entry in os.listdir(data_dir):
        full_path = os.path.join(data_dir, entry)
        all_chunks.extend(get_text_chunks(full_path))
    
    embeddings = embedding.encode([chunk.page_content for chunk in all_chunks])

    try:
        client = weaviate.Client(TEXTUAL_CLIENT)

        def init_weaviate_schema():
            schema = {
                "classes": [{
                    "class": "Document",
                    "vectorizer": "none",  
                    "properties": [
                        {
                        "name": "content",
                        "dataType": ["text"],
                        }, 
                        {
                        "name": "doc_id",
                        "dataType": ["text"],
                        }, 
                        {
                        "name": "page_num",
                        "dataType": ["int"],
                        },
                    ]
                }]
            }

            client.schema.delete_all()
            client.schema.create(schema)

        init_weaviate_schema()

        for i, chunk in enumerate(all_chunks):
            client.data_object.create(
                data_object={
                            "content": chunk.page_content,
                            "doc_id": chunk.metadata["doc_id"],
                            "page_num": chunk.metadata["page"]
                            },
                class_name='Document',
                vector=embeddings[i]
            )

        print("✅ Creating process complete!")
    except Exception as e:
        print(f"❌ Error creating index: {e}")


def update_visual_vectordb(client, embedding, collection_name, file_path):

    image_pages = get_pdf_images(file_path)

    batch_size = min(len(image_pages), 2)
    total_batches = (len(image_pages) + batch_size - 1) // batch_size
    dataloader = DataLoader(image_pages, batch_size=batch_size, shuffle=False,
                            collate_fn=lambda x: embedding.vision_processor.process_images([page.image for page in x]))

    all_embeddings = []
    for batch_doc in tqdm(dataloader, desc="Building embeddings", total=total_batches):
        with torch.no_grad():
            batch_doc = {
                k: v.to(embedding.vision_model.device)
                for k, v in batch_doc.items()
            }

            with torch.cuda.device(embedding.vision_model.device):
                torch.cuda.empty_cache()

            embeddings_doc = embedding.vision_model(**batch_doc)
            embeddings_doc = embeddings_doc.cpu().float().numpy().tolist()
            all_embeddings.extend(embeddings_doc)

    def point_generator():
        for i, (doc_page, embedding) in enumerate(zip(image_pages, all_embeddings)):
            yield models.PointStruct(
                id=i,
                vector={
                    "embedding": embedding  # Named vector for multi-vector config
                },
                payload={"file_name": f"{doc_page.doc_id}.pdf",
                          "doc_id": doc_page.doc_id,
                          "page_num": doc_page.page_num}
            )
    try:
        client.upload_points(
            collection_name=collection_name,
            points=point_generator(),
            batch_size=batch_size,
            parallel=1,
            wait=True
        )
        print("✅ Update complete!")
    except Exception as e:
        print(f"❌ Error uploading: {e}")


def update_textual_vectordb(client, embedding, file_path):

    chunks = get_text_chunks(file_path)
    embeddings = embedding.encode([chunk.page_content for chunk in chunks])

    try:
        for i, chunk in enumerate(chunks):
            client.data_object.create(
                data_object={
                            "content": chunk.page_content,
                            "doc_id": chunk.metadata["doc_id"],
                            "page_num": chunk.metadata["page"]
                            },
                class_name='Document',
                vector=embeddings[i]
            )

        print("✅ Update complete!")
    except Exception as e:
        print(f"❌ Error uploading: {e}")