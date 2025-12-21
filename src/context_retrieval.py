import os
from qdrant_client.http import models

from src.utils import get_pdf_page_as_base64


class VisualRetriever:
    def __init__(self, client, collection_name, embedding, data_dir, top_k=3):
        self.client = client
        self.collection_name = collection_name
        self.embedding = embedding 
        self.data_dir = data_dir
        self.top_k = top_k

    def retrieve(self, query): 
        query_embedding = self.embedding.encode(query, input_type="text")[0]

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=models.NamedVector(
                name="embedding",
                vector=query_embedding
            ),
            limit=self.top_k
        )
        
        retrieved_pages = []
        for res in results:
            doc_id, page_num  = res.payload["doc_id"], res.payload["page_num"]
            file_path = os.path.join(self.data_dir, doc_id + ".pdf")

            base64_code = get_pdf_page_as_base64(file_path, page_num)

            retrieved_pages.append(
                {
                    "image": {
                                "doc_id": doc_id,
                                "page_num": page_num,
                                "base64": base64_code
                            },
                    "score": float(res.score)
                }
            )
        return retrieved_pages


class TextualRetriever:
    def __init__(self, client, embedding, top_k=5, alpha=0.5):
        self.client = client
        self.embedding = embedding 
        self.top_k = top_k
        self.alpha = alpha

    def retrieve(self, query):
        query_embedding = self.embedding.encode([query])[0]

        results = ( self.client.query.get("Document", ["content", "doc_id", "page_num"])
                    .with_hybrid(query=query, alpha=self.alpha, vector=query_embedding)
                    .with_additional(["score"])
                    .with_limit(self.top_k)
                    .do()
        )

        contexts = [
            {
                    "text": {
                                "doc_id": res["doc_id"],
                                "page_num": res["page_num"],
                                "content": res["content"]
                            },
                    "score": float(res["_additional"]["score"])
                }
            for res in results["data"]["Get"]["Document"]
        ]

        return contexts