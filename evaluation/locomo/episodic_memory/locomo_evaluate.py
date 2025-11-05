# This is adapted from Mem0 (https://github.com/mem0ai/mem0/blob/main/evaluation/evals.py).
# It is modified to only report LLM judge scores and to be simpler.
# Enhanced with parallelization, retry logic, progress bar, and checkpointing.

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any

from dotenv import load_dotenv
from tqdm import tqdm
from llm_judge import evaluate_llm_judge


def load_existing_results(target_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load existing results if they exist for resume capability."""
    if os.path.exists(target_path):
        try:
            with open(target_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            print(f"Warning: Could not load existing results from {target_path}")
    return {}


def save_results(results: Dict[str, List[Dict[str, Any]]], target_path: str):
    """Save results to file."""
    with open(target_path, "w") as f:
        json.dump(results, f, indent=4)


def evaluate_single_item(item: Dict[str, Any], category: str) -> Dict[str, Any]:
    """Evaluate a single question-answer pair."""
    question = item["question"]
    locomo_answer = f"{item['locomo_answer']}"
    response = f"{item['model_answer']}"

    try:
        llm_score = evaluate_llm_judge(question, locomo_answer, response)
    except Exception as e:
        print(f"\nError evaluating question: {e}")
        print(f"Question: {question[:100]}...")
        llm_score = 0  # Default to WRONG on error

    return {
        "question": question,
        "answer": locomo_answer,
        "response": response,
        "category": category,
        "llm_score": llm_score,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LoCoMo results with LLM judge (parallel + progress bar)"
    )
    parser.add_argument(
        "--data-path", required=True, help="Path to the source data file"
    )
    parser.add_argument(
        "--target-path", required=True, help="Path to the target data file"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Number of parallel workers (default: 10)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=50,
        help="Save results every N questions (default: 50)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing results file",
    )
    args = parser.parse_args()

    data_path = args.data_path
    target_path = args.target_path
    max_workers = args.max_workers
    checkpoint_interval = args.checkpoint_interval

    # Load environment variables
    load_dotenv()

    # Load test data
    with open(data_path, "r") as f:
        test_data = json.load(f)

    # Load existing results if resume is enabled
    results = load_existing_results(target_path) if args.resume else {}

    # Build list of all items to evaluate
    all_items = []
    for key, value in test_data.items():
        if key == "5":
            continue
        for item in value:
            # Create a unique identifier for this item
            item_id = f"{key}_{item['question'][:50]}"
            # Check if already evaluated (for resume)
            already_done = False
            if args.resume and key in results:
                for existing in results[key]:
                    if existing["question"] == item["question"]:
                        already_done = True
                        break
            if not already_done:
                all_items.append((key, item))

    total_items = len(all_items)
    print(f"Total questions to evaluate: {total_items}")
    print(f"Using {max_workers} parallel workers")
    print(f"Checkpointing every {checkpoint_interval} questions\n")

    # Initialize results structure for all categories
    for key in test_data.keys():
        if key != "5" and key not in results:
            results[key] = []

    # Process items in parallel with progress bar
    completed_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_item = {
            executor.submit(evaluate_single_item, item, category): (category, item)
            for category, item in all_items
        }

        # Process completed tasks with progress bar
        with tqdm(total=total_items, desc="Evaluating", unit="question") as pbar:
            for future in as_completed(future_to_item):
                category, original_item = future_to_item[future]
                try:
                    result = future.result()
                    results[category].append(result)
                    completed_count += 1

                    # Checkpoint: save periodically
                    if completed_count % checkpoint_interval == 0:
                        save_results(results, target_path)
                        pbar.set_postfix_str(f"Saved checkpoint at {completed_count}")

                except Exception as e:
                    print(f"\nFailed to process item: {e}")
                    # Add a failed result
                    results[category].append(
                        {
                            "question": original_item.get("question", "UNKNOWN"),
                            "answer": original_item.get("locomo_answer", ""),
                            "response": original_item.get("model_answer", ""),
                            "category": category,
                            "llm_score": 0,
                        }
                    )
                finally:
                    pbar.update(1)

    # Final save
    save_results(results, target_path)
    print(f"\nEvaluation complete! Results saved to {target_path}")

    # Print summary statistics
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    for category in sorted(results.keys()):
        if results[category]:
            scores = [item["llm_score"] for item in results[category]]
            avg_score = sum(scores) / len(scores)
            correct = sum(scores)
            total = len(scores)
            print(f"Category {category}: {avg_score:.4f} ({correct}/{total} correct)")
    print("=" * 50)


if __name__ == "__main__":
    main()
