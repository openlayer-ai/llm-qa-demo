"""Simple script to train a churn classifier model and push it to Openlayer."""

import os
import sys

import openlayer
import pandas as pd
from openlayer.tasks import TaskType

# openlayer.api.OPENLAYER_ENDPOINT = "https://api-staging.openlayer.com/v1"

OPENLAYER_API_KEY = os.environ["OPENLAYER_API_KEY"]
COMMIT_MSG = os.environ.get("GITHUB_COMMIT_MESSAGE", "Commit from GitHub Action")
PROJECT_NAME = "Nubank FAQ"

# ------------------------------- Load project ------------------------------- #

client = openlayer.OpenlayerClient(OPENLAYER_API_KEY)

project = client.create_project(name=PROJECT_NAME, task_type=TaskType.LLM)

# ------------------------------- Stage dataset ------------------------------ #

dataset = pd.read_csv("validation.csv")

# Some variables that will go into the `dataset_config`
input_variable_names = ["user_question", "context"]
output_column_name = "model_output"
ground_truth_column_name = "ideal_answer"
question_column_name = "user_question"
context_column_name = "context"

validation_dataset_config = {
    "inputVariableNames": input_variable_names,
    "label": "validation",
    "outputColumnName": output_column_name,
    "groundTruthColumnName": ground_truth_column_name,
    "questionColumnName": question_column_name,
    "contextColumnName": context_column_name,
}

# Validation set
project.add_dataframe(
    dataset_df=dataset,
    dataset_config=validation_dataset_config,
)

# -------------------------------- Stage model ------------------------------- #

prompt_template = """
Levando em consideração o contexto abaixo:
-----
{{ context }}
-----

responda à seguinte pergunta do usuário:

{{ user_question }}
"""
prompt = [
    {
        "role": "system",
        "content": "Você auxilia no suporte aos usuários do banco Nubank. Você deve responder em Português (Brasil).",
    },
    {"role": "user", "content": prompt_template},
]

# Note the camelCase for the keys
model_config = {
    "prompt": prompt,
    "inputVariableNames": input_variable_names,
    "modelProvider": "OpenAI",
    "model": "gpt-3.5-turbo",
    "modelParameters": {"temperature": 0},
    "modelType": "shell",
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
