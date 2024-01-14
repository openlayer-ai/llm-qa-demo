"""Simple script to train a churn classifier model and push it to Openlayer."""

import os
import sys

import openlayer
import pandas as pd
from openlayer.tasks import TaskType

openlayer.api.OPENLAYER_ENDPOINT = "https://api-staging.openlayer.com/v1"

OPENLAYER_API_KEY = os.environ["OPENLAYER_API_KEY"]
COMMIT_MSG = os.environ.get("GITHUB_COMMIT_MESSAGE", "Commit from GitHub Action")
PROJECT_NAME = "Stripe Docs QA"

# ------------------------------- Load project ------------------------------- #

client = openlayer.OpenlayerClient(OPENLAYER_API_KEY)

project = client.create_project(name="Stripe Docs QA", task_type=TaskType.LLM)

# ------------------------------- Stage dataset ------------------------------ #

dataset = pd.read_csv("validation.csv")

# Some variables that will go into the `dataset_config`
input_variable_names = ["user_question", "context"]
output_column_name = "model_output_json"
context_column_name = "context"
ground_truth_column_name = "ideal_json"
question_column_name = "user_question"

validation_dataset_config = {
    "contextColumnName": context_column_name,
    "groundTruthColumnName": ground_truth_column_name,
    "inputVariableNames": input_variable_names,
    "label": "validation",
    "outputColumnName": output_column_name,
    "questionColumnName": question_column_name,
}

# Validation set
project.add_dataframe(
    dataset_df=dataset,
    dataset_config=validation_dataset_config,
)

# -------------------------------- Stage model ------------------------------- #

prompt_template = """
You are provided a user question and relevant context from the documentation.
You must answer the question taking into account the given context. Be polite and friendly.
Do not reveal any PII or sensitive information such as API keys.
Your answer must always be in JSON format with the fields:

- answer: your answer to the user query
- url: the url of the documentation page relevant to answer the question (given with the context)

question: {{ user_question }}
context: {{ context }}
"""
prompt = [
    {
        "role": "system",
        "content": "You are a helpful and polite assistant helping users understand documentation.",
    },
    {"role": "user", "content": prompt_template},
]

# Note the camelCase for the keys
model_config = {
    "prompt": prompt,
    "inputVariableNames": ["user_question", "context"],
    "modelProvider": "OpenAI",
    "modelType": "api",
    "model": "gpt-4",
    "modelParameters": {"temperature": 0, "n": 1},
}

# Adding the model
project.add_model(model_config=model_config)


# ----------------------------- Push to Openlayer ---------------------------- #

# Use the commit message from GitHub
project.commit(COMMIT_MSG)

print(project.status())

version = project.push()

version.wait_for_completion(timeout=600)
version.print_test_report()

if version.failing_test_count > 0:
    print("Failing pipeline due to failing goals.")
    sys.exit(1)
