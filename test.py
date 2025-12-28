import json
import random
import pandas as pd

from src.rag import rag_for_single_query


def filter_dataset(df):
    unique_doc_ids = df.doc_id.unique()
    unique_doc_ids = random.sample(unique_doc_ids, 50)

    filtered_df = df[df.doc_id.isin(unique_doc_ids)]
    filtered_df = filtered_df[["q_id", "doc_id", "doc_path", "question", "answer", "evidence_pages"]]

    return filtered_df


if __name__ == "__main__":
    
    test_data_path = "data.csv"
    output_file = "results.json"

    df = pd.read_csv(test_data_path)
    filtered_df = filter_dataset(df)
    del df
    
    results = []
    for row in filtered_df.iterrows():
        q_id, doc_id = row["q_id"], row["doc_id"]
        gt_ids = [f"{doc_id}_{page}" for page in eval(row["evidence_pages"], list)]

        try:
            response = rag_for_single_query(eval(row["question"], str))

            visual_retrieved_ids = [f"{con.get("image").get("doc_id")}_{con.get("image").get("page_num")}" 
                            for con in response["visual_context"]]
            visual_scores = [con.get("score") for con in response["visual_context"]]

            textual_retrieved_ids = [f"{con.get("text").get("doc_id")}_{con.get("text").get("page_num")}" 
                            for con in response["textual_context"]]
            textual_scores = [con.get("score") for con in response["textual_context"]]

            results.append({
                "q_id": q_id,
                "question": row["question"], 
                "gt_answer": row["answer"],
                "gt_ids": gt_ids, 
                "final_answer": response["answer"],
                "visual_response_dict": response["visual_response_dict"],
                "textual_response_dict": response["textual_response_dict"],
                "visual_retrieved_ids": visual_retrieved_ids,
                "textual_retrieved_ids": textual_retrieved_ids,
                "visual_scores": visual_scores,
                "textual_scores": textual_scores

            })
        except Exception as e:
            print(f"Error answering for query {q_id}: {str(e)}")
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)