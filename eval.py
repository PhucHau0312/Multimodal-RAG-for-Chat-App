import os
import json

import re
import string

from typing import List
from collections import Counter

from vidore_benchmark.evaluation.metrics import mrr, recall_at_k, ndcg_at_k

from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from deepeval.metrics.g_eval import GEval
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric


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
    formatted_qrels  = {}
    visual_results, textual_results  = {}, {}

    for data in result_data:
        formatted_qrels[str(data["q_id"])] = {id: 1.0 for id in data["gt_ids"]}

        visual_results[str(data["q_id"])] = {str(id): float(score) for id, score in zip(data["visual_retrieved_ids"], data["visual_scores"])}
        textual_results[str(data["q_id"])] = {str(id): float(score) for id, score in zip(data["textual_retrieved_ids"], data["textual_scores"])}

    visual_eval_results = {
        "ndcg@5": ndcg_at_k(formatted_qrels, visual_results, k=5),
        "recall@5": recall_at_k(formatted_qrels, visual_results, k=5),
        "mrr@10": mrr(formatted_qrels, visual_results, k=10)
    }
    textual_eval_results = {
        "ndcg@5": ndcg_at_k(formatted_qrels, textual_results, k=5),
        "recall@5": recall_at_k(formatted_qrels, textual_results, k=5),
        "mrr@10": mrr(formatted_qrels, textual_results, k=10)
    }

    return {"visual": visual_eval_results,
            "textual": textual_eval_results}


def calculate_answering_scores(result_data):
    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine if the 'actual output' is factually correct based on the 'expected output' (ground truth).",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.8
    )
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.7)

    visual_test_cases, textual_test_cases, final_test_cases = [], [], []
    visual_f1_scores, textual_f1_scores, final_f1_scores = [], [], []
    for i, data in enumerate(result_data):
        visual_answer = data["visual_response_dict"].get("Answer")
        textual_answer = data["textual_response_dict"].get("Answer")
        answer, gt_answer = data["answer"], data["gt_answer"]

        if visual_answer:
            test_case = LLMTestCase(
                input=data['question'],          
                actual_output=answer,    
                expected_output=gt_answer   
            )
            visual_test_cases.append(test_case)
            visual_f1_scores.append(calculate_f1(word_tokenize(answer), word_tokenize(gt_answer)))

        if textual_answer:
            test_case = LLMTestCase(
                input=data['question'],          
                actual_output=answer,    
                expected_output=gt_answer   
            )
            textual_test_cases.append(test_case)
            textual_f1_scores.append(calculate_f1(word_tokenize(answer), word_tokenize(gt_answer)))

        if answer:
            test_case = LLMTestCase(
                input=data['question'],          
                actual_output=answer,    
                expected_output=gt_answer   
            )
            final_test_cases.append(test_case)
            final_f1_scores.append(calculate_f1(word_tokenize(answer), word_tokenize(gt_answer)))

    visual_eval = evaluate(visual_test_cases, [correctness_metric, answer_relevancy_metric], print_results=False)
    textual_eval = evaluate(textual_test_cases, [correctness_metric, answer_relevancy_metric], print_results=False)
    final_eval = evaluate(final_test_cases, [correctness_metric, answer_relevancy_metric], print_results=False)
        
    visual_results =  {
        "correctness": {
            "score": sum([res["metrics_results"][0]["score"] for res in visual_eval]) / len(visual_eval),
            "success_rate": sum(1 for res in visual_eval if res["metrics_results"][0]["success"]) / len(visual_eval) * 100
        },
        "answer_relevancy": {
            "score": sum([res["metrics_results"][1]["score"] for res in visual_eval]) / len(visual_eval),
            "success_rate": sum(1 for res in visual_eval if res["metrics_results"][1]["success"]) / len(visual_eval) * 100
        },
        "f1": sum(visual_f1_scores) / len(visual_f1_scores)
    }
    textual_results =  {
        "correctness": {
            "score": sum([res["metrics_results"][0]["score"] for res in textual_eval]) / len(textual_eval),
            "success_rate": sum(1 for res in textual_eval if res["metrics_results"][0]["success"]) / len(textual_eval) * 100
        },
        "answer_relevancy": {
            "score": sum([res["metrics_results"][1]["score"] for res in textual_eval]) / len(textual_eval),
            "success_rate": sum(1 for res in textual_eval if res["metrics_results"][1]["success"]) / len(textual_eval) * 100
        },
        "f1": sum(textual_f1_scores) / len(textual_f1_scores)
    }
    final_results =  {
        "correctness": {
            "score": sum([res["metrics_results"][0]["score"] for res in final_eval]) / len(final_eval),
            "success_rate": sum(1 for res in final_eval if res["metrics_results"][0]["success"]) / len(final_eval) * 100
        },
        "answer_relevancy": {
            "score": sum([res["metrics_results"][1]["score"] for res in final_eval]) / len(final_eval),
            "success_rate": sum(1 for res in final_eval if res["metrics_results"][1]["success"]) / len(final_eval) * 100
        },
        "f1": sum(final_f1_scores) / len(final_f1_scores)
    }

    return {"visual": visual_results,
            "textual": textual_results, 
            "final": final_results}


def print_results(retrieval_results, answering_results):
    print("Retrieval Evaluation:")

    print("Visual Retrieval:")
    print("NDCG@5:", retrieval_results["visual"]["ndcg@5"])
    print("Recall@5:", retrieval_results["visual"]["recall@5"])
    print("MRR@10:", retrieval_results["visual"]["mrr@10"])
    print("---------------------")
    print("Textual Retrieval:")
    print("NDCG@5:", retrieval_results["textual"]["ndcg@5"])
    print("Recall@5:", retrieval_results["textual"]["recall@5"])
    print("MRR@10:", retrieval_results["textual"]["mrr@10"])
    
    print("---------------------\n---------------------")

    print("Generation Evaluation:")

    print("Visual Answering:")
    print("Correctness:")
    print("- Average Score:", answering_results["visual"]["correctness"]["score"])
    print("- Success Rate:", answering_results["visual"]["correctness"]["success_rate"])
    print("Answer Relevancy:")
    print("- Average Score:", answering_results["visual"]["answer_relevancy"]["score"])
    print("- Success Rate:", answering_results["visual"]["answer_relevancy"]["success_rate"])
    print("F1 Score:", answering_results["visual"]["f1"])
    print("---------------------")
    print("Textual Answering:")
    print("Correctness:")
    print("- Average Score:", answering_results["textual"]["correctness"]["score"])
    print("- Success Rate:", answering_results["textual"]["correctness"]["success_rate"])
    print("Answer Relevancy:")
    print("- Average Score:", answering_results["textual"]["answer_relevancy"]["score"])
    print("- Success Rate:", answering_results["textual"]["answer_relevancy"]["success_rate"])
    print("F1 Score:", answering_results["textual"]["f1"])
    print("---------------------")
    print("Final Answering:")
    print("Correctness:")
    print("- Average Score:", answering_results["final"]["correctness"]["score"])
    print("- Success Rate:", answering_results["final"]["correctness"]["success_rate"])
    print("Answer Relevancy:")
    print("- Average Score:", answering_results["final"]["answer_relevancy"]["score"])
    print("- Success Rate:", answering_results["final"]["answer_relevancy"]["success_rate"])
    print("F1 Score:", answering_results["final"]["f1"])


if __name__ == "__main__":

    file_path = ""
    with open(file_path, "r") as f:
        data = json.load(f)

    retrieval_result = calculate_retrieval_scores(data)
    answering_result = calculate_answering_scores(data)
    
    print_results(retrieval_result, answering_result)

    