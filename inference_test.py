from mlx_lm import load, generate
import json
import argparse
from tqdm import tqdm
from fuzzywuzzy import fuzz
import re

def load_model(model_path, adapter_path):
    """Load the model and tokenizer"""
    print(f"Loading model from {model_path}...")
    model, tokenizer = load(model_path, adapter_path=adapter_path, 
                            tokenizer_config=
                            {
                                "eos_token": "<|im_end|>",
                                "pad_token": "<|im_end|>",
                                "temp": "0.6",
                             })

        
    return model, tokenizer

def get_prediction(model, tokenizer, symptoms):
    """Get model prediction for given symptoms"""
    prompt = f"Patient reported these symptoms: {symptoms}"
    messages = [
        {"role": "system",
         "content": "You are a medical diagnosis classifier. Given a description of symptoms, provide ONLY the name of the most likely diagnosis. Do not include explanations, reasoning, or additional text."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    response = generate(model, tokenizer, prompt=text, verbose=False, max_tokens=512) 
    # Clean the response to extract just the diagnosis
    diagnosis = response.strip()
    # Remove any common prefixes in responses
    prefixes_to_remove = ["The diagnosis is ", "Diagnosis: ", "Likely diagnosis: "]
    for prefix in prefixes_to_remove:
        if diagnosis.startswith(prefix):
            diagnosis = diagnosis[len(prefix):]
    
    # Better cleaning: Extract just the first part of the diagnosis 
    # (before any non-English characters or special tokens)
    # Match only valid English diagnosis characters and stop at first non-English/special character
    match = re.search(r'^[a-zA-Z0-9\s\(\)\-\'\.]+', diagnosis)
    if match:
        diagnosis = match.group(0).strip()
    else:
        # Fallback cleaning if regex doesn't match
        diagnosis = re.sub(r'[^\w\s\(\)\-\'\.]+', '', diagnosis)
    
    # Remove 'useralative', token identifiers and other common artifacts
    artifacts = ["useralative", "tokennameidentifier", "<|im_end|>"]
    for artifact in artifacts:
        diagnosis = diagnosis.replace(artifact, "")
    
    return diagnosis.strip()

def normalize_diagnosis(diagnosis):
    """Normalize diagnosis names for consistent comparison"""
    # Convert to lowercase
    diagnosis = diagnosis.lower()
    
    # Replace common variations
    replacements = {
        "copd": "chronic obstructive pulmonary disease",
        "bph": "benign prostatic hyperplasia",
        # Add more common variations as needed
    }
    
    for key, value in replacements.items():
        if key in diagnosis and value not in diagnosis:
            diagnosis = diagnosis.replace(key, value)
    
    # Remove special characters and extra whitespace
    import re
    diagnosis = re.sub(r'[^\w\s\(\)\-\.]+', '', diagnosis)
    diagnosis = re.sub(r'\s+', ' ', diagnosis).strip()
    
    return diagnosis

def is_correct_prediction(predicted, true, threshold=80):
    """More robust fuzzy matching for comparing diagnoses"""
    # Extract just the first word or first few words for initial comparison
    def extract_main_term(text):
        text = text.lower().strip()
        # Get the first word or hyphenated term
        main_term = re.split(r'[\s,]', text)[0]
        return main_term
    
    # First try exact matching of normalized strings
    normalized_pred = normalize_diagnosis(predicted)
    normalized_true = normalize_diagnosis(true)
    
    # If the main terms match (e.g., "sepsis" in "sepsis" and "sepsis with complications")
    # or the fuzzy ratio is high enough, consider it correct
    main_pred = extract_main_term(normalized_pred)
    main_true = extract_main_term(normalized_true)
    
    exact_match = normalized_pred == normalized_true
    main_term_match = main_pred == main_true and len(main_pred) > 3  # Avoid matching on short terms
    fuzzy_match = fuzz.ratio(normalized_pred, normalized_true) > threshold
    
    return exact_match or main_term_match or fuzzy_match

def calculate_error_rate(jsonl_path, model_path, adapter_path):
    model, tokenizer = load_model(model_path, adapter_path)

    total_cases = 0
    errors = 0
    results = []

    print(f"Processing cases from {jsonl_path}...")

    # Read and process JSONL file
    with open(jsonl_path, 'r') as file:
        for line in tqdm(file):
            try:
                data = json.loads(line)

                # Extract the conversation
                messages = data.get("messages", [])
                if len(messages) < 3:  # Need system, user, and assistant messages
                    continue

                # Extract symptoms from user message
                user_message = next((m["content"] for m in messages if m["role"] == "user"), "")
                symptoms = user_message.replace("Patient reported these symptoms:", "").strip()

                # Extract ground truth diagnosis from assistant message
                true_diagnosis = next((m["content"] for m in messages if m["role"] == "assistant"), "").lower()

                if not symptoms or not true_diagnosis:
                    continue

                # Get model prediction
                predicted_diagnosis = get_prediction(model, tokenizer, symptoms).lower()

                # Compare prediction with ground truth
                is_correct = is_correct_prediction(predicted_diagnosis, true_diagnosis)
                if not is_correct:
                    errors += 1

                # Store result
                results.append({
                    "symptoms": symptoms,
                    "true_diagnosis": true_diagnosis,
                    "predicted_diagnosis": predicted_diagnosis,
                    "is_correct": is_correct
                })

                total_cases += 1
            except json.JSONDecodeError:
                print(f"Warning: Skipped invalid JSON line")
                continue

    # Calculate error rate
    error_rate = errors / total_cases if total_cases > 0 else 0
    accuracy = 1 - error_rate

    print(f"\nResults:")
    print(f"Total cases evaluated: {total_cases}")
    print(f"Correct predictions: {total_cases - errors}")
    print(f"Incorrect predictions: {errors}")
    print(f"Error rate: {error_rate:.4f} ({error_rate * 100:.2f}%)")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # Save detailed results to file
    output_path = "diagnosis_evaluation_results.json"
    with open(output_path, 'w') as f:
        json.dump({
            "summary": {
                "total_cases": total_cases,
                "errors": errors,
                "error_rate": error_rate,
                "accuracy": accuracy
            },
            "detailed_results": results
        }, f, indent=2)

    print(f"Detailed results saved to {output_path}")
    return error_rate


def calculate_confusion_matrix(jsonl_path, model_path, adapter_path):
    """Calculate confusion matrix and detailed metrics for each diagnosis"""
    model, tokenizer = load_model(model_path, adapter_path)

    # Dictionary to store diagnosis counts
    diagnoses_stats = {}
    all_diagnoses = set()
    confusion_matrix = {}

    print(f"Analyzing diagnosis patterns from {jsonl_path}...")

    # Read and process JSONL file
    with open(jsonl_path, 'r') as file:
        for line in tqdm(file):
            try:
                data = json.loads(line)

                # Extract the conversation
                messages = data.get("messages", [])
                if len(messages) < 3:
                    continue

                # Extract symptoms from user message
                user_message = next((m["content"] for m in messages if m["role"] == "user"), "")
                symptoms = user_message.replace("Patient reported these symptoms:", "").strip()

                # Extract ground truth diagnosis from assistant message
                true_diagnosis = next((m["content"] for m in messages if m["role"] == "assistant"), "").lower()

                if not symptoms or not true_diagnosis:
                    continue

                # Get model prediction
                predicted_diagnosis = get_prediction(model, tokenizer, symptoms).lower()

                # Add diagnoses to tracking sets
                all_diagnoses.add(true_diagnosis)
                all_diagnoses.add(predicted_diagnosis)

                # Update confusion matrix
                if true_diagnosis not in confusion_matrix:
                    confusion_matrix[true_diagnosis] = {}

                if predicted_diagnosis not in confusion_matrix[true_diagnosis]:
                    confusion_matrix[true_diagnosis][predicted_diagnosis] = 0

                confusion_matrix[true_diagnosis][predicted_diagnosis] += 1

                # Track diagnosis statistics
                if true_diagnosis not in diagnoses_stats:
                    diagnoses_stats[true_diagnosis] = {"total": 0, "correct": 0, "incorrect": 0}

                diagnoses_stats[true_diagnosis]["total"] += 1
                if predicted_diagnosis == true_diagnosis:
                    diagnoses_stats[true_diagnosis]["correct"] += 1
                else:
                    diagnoses_stats[true_diagnosis]["incorrect"] += 1

            except json.JSONDecodeError:
                print(f"Warning: Skipped invalid JSON line")
                continue

    # Calculate accuracy for each diagnosis
    for diagnosis, stats in diagnoses_stats.items():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0

    # Sort diagnoses by their frequency
    sorted_diagnoses = sorted(diagnoses_stats.items(), key=lambda x: x[1]["total"], reverse=True)

    print("\nDiagnosis Performance:")
    print("-" * 80)
    print(f"{'Diagnosis':<30} {'Total':<10} {'Correct':<10} {'Accuracy':<10}")
    print("-" * 80)

    for diagnosis, stats in sorted_diagnoses[:20]:  # Show top 20 diagnoses
        print(f"{diagnosis:<30} {stats['total']:<10} {stats['correct']:<10} {stats['accuracy']:.4f}")

    # Save detailed analysis to file
    output_path = "diagnosis_detailed_analysis.json"
    with open(output_path, 'w') as f:
        json.dump({
            "diagnoses_stats": diagnoses_stats,
            "confusion_matrix": confusion_matrix
        }, f, indent=2)

    print(f"\nDetailed diagnosis analysis saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate medical diagnosis model on JSONL dataset")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B", help="Path to the model")
    parser.add_argument("--data", type=str, default="./data/test.jsonl", help="Path to JSONL file with test cases")
    parser.add_argument("--adapter", type=str, help="Adapter file path")
    parser.add_argument("--analysis", type=bool, default=True, help="Perform detailed diagnosis analysis")
    args = parser.parse_args()

    error_rate = calculate_error_rate(args.data, args.model, args.adapter)

    if args.analysis:
        calculate_confusion_matrix(args.data, args.model, args.adapter)