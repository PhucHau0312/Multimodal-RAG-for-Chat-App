import os
import gc
import uuid

import torch
import hashlib
import numpy as np

import weaviate
from weaviate.util import generate_uuid5

from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.utils import get_pdf_images, get_text_chunks, complex_get_text_chunks


VISUAL_CLIENT = os.getenv("VISUAL_CLIENT")
TEXTUAL_CLIENT = os.getenv("TEXTUAL_CLIENT")
VECTOR_DIMENSION = 128


def create_visual_vectordb(embedding,
                           collection_name,
                           data_dir="docs",
                           batch_size=8,
                           parallel_upload=2):

    pdf_files = []
    for entry in os.listdir(data_dir):
        full_path = os.path.join(data_dir, entry)
        if os.path.isfile(full_path) and entry.lower().endswith(".pdf"):
            pdf_files.append(full_path)

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
    device = embedding.vision_model.device

    def collate_fn(pages):
        images = [p.image for p in pages]
        metas = [(p.doc_id, p.page_num) for p in pages]

        inputs = embedding.vision_processor.process_images(images)

        for im in images:
            try:
                im.close()
            except Exception:
                pass

        return inputs, metas

    for pdf_path in pdf_files:
        pages = get_pdf_images(pdf_path)  # list of page objects; keep only for this PDF

        dataloader = DataLoader(
            pages,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,     # Colab: safer for memory than >0
            pin_memory=True,
            persistent_workers=False,
        )
        pbar = tqdm(dataloader, desc=f"Embedding {Path(pdf_path).name}", leave=False)

        for batch_inputs, metas in pbar:
            try:
                # Move to GPU
                batch_inputs = {
                    k: v.to(device, non_blocking=True)
                    for k, v in batch_inputs.items()
                }

                with torch.no_grad():
                    out = embedding.vision_model(**batch_inputs)
                emb = out.detach().float().cpu().numpy()

                points = []
                for (doc_id, page_num), vec in zip(metas, emb):
                    # Stable id avoids needing a global counter
                    pid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{page_num}"))
                    points.append(
                        models.PointStruct(
                            id=pid,
                            vector={"embedding": vec.tolist()},
                            payload={
                                "file_name": f"{doc_id}.pdf",
                                "doc_id": doc_id,
                                "page_num": int(page_num),
                            },
                        )
                    )

                client.upload_points(
                    collection_name=collection_name,
                    points=points,
                    batch_size=len(points),
                    parallel=parallel_upload,
                    wait=True,
                )

            except torch.cuda.OutOfMemoryError:
                # Emergency fallback: clear and continue with a smaller batch size
                torch.cuda.empty_cache()
                gc.collect()
                raise RuntimeError(
                    "CUDA OOM. Reduce batch_size (e.g., 4 or 2), or downscale images in the processor."
                )
            finally:
                # Free references ASAP
                del batch_inputs, out, emb, points
                torch.cuda.empty_cache()
                gc.collect()

        del pages
        gc.collect()

    print("✅ Creating process complete!")


def create_textual_vectordb(
    embedding,
    data_dir="docs",
    embed_batch_size=64,
    weaviate_batch_size=128,
):
    client = weaviate.Client(TEXTUAL_CLIENT)

    def ensure_schema():
        schema_class = {
            "class": "Document",
            "vectorizer": "none",
            "properties": [
                {"name": "content", "dataType": ["text"]},
                {"name": "file_name", "dataType": ["text"]},
                {"name": "doc_id", "dataType": ["text"]},
                {"name": "page_num", "dataType": ["int"]},
                {"name": "text_source", "dataType": ["text"]},
            ],
        }

        existing = client.schema.get()
        exists = any(c.get("class") == schema_class["class"] for c in existing.get("classes", []))

        if not exists:
            client.schema.create_class(schema_class)

    ensure_schema()

    try:
        client.batch.configure(
            batch_size=weaviate_batch_size,
            dynamic=True,
            num_workers=2,
        )

        def stable_text_id(text: str) -> str:
            # stable across runs
            return hashlib.sha1(text.encode("utf-8")).hexdigest()

        def flush(batch, chunk_buffer):
            if not chunk_buffer:
                return

            texts = [c.page_content for c in chunk_buffer]
            vecs = embedding.encode(texts)
            vecs = np.asarray(vecs, dtype=np.float32)

            for chunk, v in zip(chunk_buffer, vecs):
                doc_id = chunk.metadata.get("doc_id")
                file_name = f"{doc_id}.pdf"
                page_num = int(chunk.metadata.get("page"))
                text_source = chunk.metadata.get("text_source")

                key = f"{doc_id}:{page_num}:sha1:{stable_text_id(chunk.page_content)}"
                uid = generate_uuid5(key)

                batch.add_data_object(
                    data_object={
                        "content": chunk.page_content,
                        "file_name": file_name,
                        "doc_id": doc_id,
                        "page_num": page_num,
                        "text_source": text_source,
                    },
                    class_name="Document",
                    uuid=uid,
                    vector=v.tolist(),
                )

            chunk_buffer.clear()
            del vecs, texts
            gc.collect()

            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        buffer = []
        with client.batch as batch:
            for entry in os.listdir(data_dir):
                full_path = os.path.join(data_dir, entry)
                if not (os.path.isfile(full_path) and entry.lower().endswith(".pdf")):
                    continue

                chunks = complex_get_text_chunks(full_path)
                for chunk in chunks:
                    buffer.append(chunk)
                    if len(buffer) >= embed_batch_size:
                        flush(batch, buffer)

                del chunks
                gc.collect()

            flush(batch, buffer)

        print("✅ Creating process complete!")

    except Exception as e:
        print(f"❌ Error creating index: {e}")