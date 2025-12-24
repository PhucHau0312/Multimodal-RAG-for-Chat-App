import os
import json

import re
import string

from typing import List
from collections import Counter

from vidore_benchmark.evaluation.metrics import mrr, recall_at_k, ndcg_at_k

from deepeval.metrics.g_eval import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval import evaluate


def normalize_answer(s):
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    Lower text and remove punctuation, articles and extra whitespace.
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def word_tokenize(text: str) -> List[str]:
    """Tokenize text into words after normalization."""
    normalized = normalize_answer(text)
    return normalized.split()


def calculate_f1(prediction: List[str], ground_truth: List[str]) -> float:
    """Calculate F1 score between prediction and ground truth tokens."""
    prediction_counter = Counter(prediction)
    ground_truth_counter = Counter(ground_truth)
    
    true_positives = sum((prediction_counter & ground_truth_counter).values())
    false_positives = sum(prediction_counter.values()) - true_positives
    false_negatives = sum(ground_truth_counter.values()) - true_positives
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1


def calculate_retrieval_scores(result_data):
    results = {}
    formatted_qrels = {}

    for data in result_data:
        results[str(data["q_id"])] = {str(id): float(score) for id, score in zip(data["retrieved_ids"], data["scores"])}
        formatted_qrels[str(data["q_id"])] = {id: 1.0 for id in data["gt_ids"]}

    return {
        "ndcg@5", ndcg_at_k(formatted_qrels, results, k=5),
        "recall@5", recall_at_k(formatted_qrels, results, k=5),
        "mrr@10", mrr(formatted_qrels, results, k=10)
    }


def calculate_generation_scores(result_data):
    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine if the 'actual output' is factually correct based on the 'expected output' (ground truth).",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.8
    )

    test_cases = []
    f1_scores = []
    for i, data in enumerate(result_data):
        answer, gt_answer = data["answer"], data["gt_answer"]

        if answer and gt_answer:
            test_case = LLMTestCase(
                input=data['question'],          
                actual_output=answer,    
                expected_output=gt_answer   
            )
            test_cases.append(test_case)

            predicted_tokens = word_tokenize(answer)
            ground_truth_tokens = word_tokenize(gt_answer)
            f1_scores.append(calculate_f1(predicted_tokens, ground_truth_tokens))

    results = evaluate(test_cases, [correctness_metric], print_results=False)

    return {
        "correctness": {
            "score": sum([res.metrics_metadata[0].score for res in results]) / len(results),
            "success_rate": sum(1 for res in results if res.success) / len(results) * 100
        },
        "f1": sum(f1_scores) / len(f1_scores)
    }


def print_results(retrieval_results, generation_results):
    print("Retrieval Evaluation:")
    print("NDCG@5:", retrieval_results["ndcg@5"])
    print("Recall@5:", retrieval_results["recall@5"])
    print("MRR@10:", retrieval_results["mrr@10"])
    
    print("Generation Evaluation:")
    print("Average Correctness Score:", generation_results["correctness"]["score"])
    print("Success Rate:", generation_results["correctness"]["success_rate"])
    print("F1 Score:", generation_results["f1"])


if __name__ == "__main__":

    file_path = ""
    with open(file_path, "r") as f:
        data = json.load(f)

    retrieval_result = calculate_retrieval_scores(data)
    generation_result = calculate_generation_scores(data)
    
    print_results(retrieval_result, generation_result)

    
