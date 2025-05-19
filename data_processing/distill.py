import json
import os
import time
from tqdm import tqdm

from openai import AzureOpenAI

endpoint = os.environ["OPENAI_ENDPOINT"]
model_name = "gpt-4o-mini"
deployment = "gpt-4o-mini"

subscription_key = os.environ["OPENAI_KEY"]
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

failed_to_fetch = []

# Generate a summary using OpenAI API
def get_summary(condition):
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    ...
                },
                {
                    "role": "user",
                    "content": f"{condition}",
                }
            ],
            max_tokens=4096,
            temperature=0.7,
            top_p=1.0,
            model=deployment
        )

        return response.choices[0].message.content
    except Exception as e:
        print(f"Error fetching summary for '{condition}': {e}")
        return ""


output_directory = "distilled_data"
# Load the JSON file
filename = 'output.json'
with open(filename, 'r') as file:
    conditions = json.load(file)

list_of_conditions = list(conditions.keys())

# Loop through conditions and update with summaries
for condition in tqdm(list_of_conditions):

    output_file = f'{output_directory}/{condition}.md'
    # Write to a new Markdown file
    if os.path.exists(output_file):
        continue

    summary = get_summary(condition)
    if summary == "" or len(summary) <1:
        failed_to_fetch.append(condition)
        continue

    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(summary)
        time.sleep(1)  # Respect API rate limits


with open("failed.txt", 'w', encoding='utf-8') as file:
    file.write(','.join(failed_to_fetch))

