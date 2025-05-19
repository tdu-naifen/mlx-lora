from datasets import load_dataset
import json

# Folder to save JSONL files
folder = "data/"

# Load dataset
#  See the data distribution and adjust over/under representation.
dataset = load_dataset("Specify data set")

# Create train/validation split from 'train'
train_test_split = dataset["train"].train_test_split(test_size=0.2, seed=42)

# Create test/valid split from 'test'
test_valid_splits = dataset["test"].train_test_split(test_size=0.5, seed=42)

# Assign datasets
train_dataset = train_test_split["train"]
valid_dataset = train_test_split["test"]
test_dataset = test_valid_splits["train"]
final_valid_dataset = test_valid_splits["test"]

# System message used in prompt format
system_message = (
    ...
)

# Save HuggingFace dataset to JSONL in OpenAI fine-tune format
def save_as_jsonl(dataset_split, filename):
    with open(filename, 'w') as file:
        for sample in dataset_split:
            conversation = {
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": sample["input_text"]},
                    {"role": "assistant", "content": sample["output_text"]}
                ]
            }
            file.write(json.dumps(conversation) + '\n')

# Save to disk
save_as_jsonl(train_dataset, folder + "train.jsonl")
save_as_jsonl(valid_dataset, folder + "valid.jsonl")
save_as_jsonl(test_dataset, folder + "test.jsonl")