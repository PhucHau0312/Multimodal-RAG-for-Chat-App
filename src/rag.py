import os 
import shutil

from weaviate import Client
from qdrant_client import QdrantClient

from src.embeddings import VisionEmbedding, TextEmbedding
from src.context_retrieval import VisualRetriever, TextualRetriever
from src.indexing import update_visual_vectordb, update_textual_vectordb
from src.responses import openai_chat_complete, route_and_rewrite, generate_visual_response, generate_textual_response, combine_responses, combine_responses_for_single_query
from src.parse import extract_sections, parse_combined_output


VISUAL_COLLECTION = os.getenv("VISUAL_COLLECTION")
DATA_DIR = os.getenv("DATA_DIR")

VISUAL_CLIENT = os.getenv("VISUAL_CLIENT")
TEXTUAL_CLIENT = os.getenv("TEXTUAL_CLIENT")

VISUAL_REPO = os.getenv("VISUAL_REPO")
TEXTUAL_REPO = os.getenv("TEXTUAL_REPO")

visual_client = QdrantClient(url=VISUAL_CLIENT)
textutal_client = Client(TEXTUAL_CLIENT)

visual_embedding = VisionEmbedding(repo=VISUAL_REPO)
textual_embedding = TextEmbedding(repo=TEXTUAL_REPO)

visual_retriever = VisualRetriever(client=visual_client,
                                   collection_name=VISUAL_COLLECTION,
                                   embedding=visual_embedding,
                                   data_dir=DATA_DIR)
textual_retriever = TextualRetriever(client=textutal_client,
                                     embedding=textual_embedding)

class RAGChain:
    def __init__(self):
        pass

    def run(self, query, chat_history):  
        decision = route_and_rewrite(query, chat_history)

        if decision.route == "rag":
            rewritten_query = decision.rewritten_query

            visual_contexts = visual_retriever.retrieve(rewritten_query)
            visual_response = generate_visual_response(rewritten_query, visual_contexts)
            visual_response_dict = extract_sections(visual_response)

            textual_contexts = textual_retriever.retrieve(rewritten_query)
            textual_response = generate_textual_response(rewritten_query, textual_contexts)
            textual_response_dict = extract_sections(textual_response)

            combined_responses = combine_responses(query, chat_history, visual_response_dict, textual_response_dict)
            final_response = parse_combined_output(combined_responses)

            return {
                    "question": query,
                    "answer": final_response.get("Final Answer", ""),
                    "analysis": final_response.get("Analysis", ""),
                    "conclusion": final_response.get("Conclusion", ""),
                    "response1": visual_response_dict,
                    "response2": textual_response_dict
                    }
        else:
            messages = [
            {"role": "system", "content": "As an intelligent assistant, please answer these questions based on your own knowledge."},
            {"role": "user", "content": query}
            ]
            response = openai_chat_complete(messages=messages)
            
            return {
                    "question": query,
                    "answer": response
                    }

    def update_knowledge(file_path):
        _, file_name = os.path.split(file_path)

        if os.path.exists(os.path.join(DATA_DIR, file_name)): 
            print("Already been in knowledge!")
        else:
            update_visual_vectordb(client=visual_client,
                                embedding=visual_embedding,
                                collection_name=VISUAL_COLLECTION,
                                file_path=file_path)
            
            update_textual_vectordb(client=textutal_client, 
                                    embedding=textual_embedding,
                                    file_path=file_path)
            
            print("Update knowledge successfully!")

            shutil.move(file_path, os.path.join(DATA_DIR, file_name))


def rag_for_single_query(query):

    visual_contexts = visual_retriever.retrieve(query)
    visual_response = generate_visual_response(query, visual_contexts)
    visual_response_dict = extract_sections(visual_response)

    textual_contexts = textual_retriever.retrieve(query)
    textual_response = generate_textual_response(query, textual_contexts)
    textual_response_dict = extract_sections(textual_response)

    combined_responses = combine_responses_for_single_query(query, visual_response_dict, textual_response_dict)
    final_response = parse_combined_output(combined_responses)

    return {
            "question": query,
            "answer": final_response.get("Final Answer", ""),
            "analysis": final_response.get("Analysis", ""),
            "conclusion": final_response.get("Conclusion", ""),
            "visual_context": visual_contexts,
            "textual_context": textual_contexts
            }