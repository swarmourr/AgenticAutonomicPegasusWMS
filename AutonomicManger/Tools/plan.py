import yaml
import os
import requests
import subprocess
from ruamel.yaml import YAML
from collections.abc import Mapping
import json
import pprint as p
from enum import Enum
from tinydb import TinyDB, Query
import time
import logging


class TerminalColor(Enum):
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    LIGHT_GRAY = '\033[37m'
    DARK_GRAY = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    WHITE = '\033[97m'
    RESET = '\033[0m'

    def apply(self, text):
        return f"{self.value}{text}{TerminalColor.RESET.value}"


class PegasusWorkflowManagerPlanner:
    def __init__(self, workflow_dir, api_url, api_key, workflow_id, corrected_yaml_path="corrected_workflow.yml"):
        self.workflow_dir = workflow_dir
        self.workflow_id = workflow_id
        self.api_url = api_url
        self.api_key = api_key
        self.corrected_yaml_path = corrected_yaml_path
        self.models = [
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "deepseek-ai/DeepSeek-R1",
            "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            "google/gemma-2-27b-it",
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "Qwen/Qwen2.5-72B-Instruct-Turbo",
            "Qwen/Qwen2.5-7B-Instruct-Turbo"
        ]

    def run_pegasus_analyzer(self):
        """Run Pegasus Analyzer to generate workflow logs."""
        try:
            result = subprocess.run(
                ['pegasus-analyzer', self.workflow_dir],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            return result.stdout + "\n" + result.stderr
        except subprocess.CalledProcessError as e:
            print(f"Error running pegasus-analyzer: {e.stderr}")
            return e.stdout + "\n" + e.stderr

    def find_yaml_file(self):
        """Find the workflow YAML file in the given directory."""
        wf_name = self.workflow_dir.split("/")[-2]
        for file_name in os.listdir(self.workflow_dir):
            if (file_name.endswith(wf_name + '.yml') or file_name.endswith(wf_name + '.yaml')) and file_name != 'braindump.yml':
                return os.path.join(self.workflow_dir, file_name)
        raise FileNotFoundError(f"No YAML file found in directory: {self.workflow_dir}")

    def clean_data(self, data):
        """Recursively clean ruamel.yaml objects to plain Python types."""
        if isinstance(data, Mapping):
            return {key: self.clean_data(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.clean_data(item) for item in data]
        else:
            return data

    def load_workflow_yaml(self, file_path):
        """Load the workflow YAML file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Workflow file not found: {file_path}")
        yaml = YAML()
        yaml.preserve_quotes = True
        with open(file_path, 'r') as yaml_file:
            raw_data = yaml.load(yaml_file)
        return self.clean_data(raw_data)

    def save_corrected_workflow(self, corrected_workflow):
        """Save the corrected workflow YAML to a file."""
        with open(self.corrected_yaml_path, 'w') as file:
            yaml.safe_dump(corrected_workflow, file, default_flow_style=False)

    def send_logs_and_workflow_to_llm(self, logs, workflow, general_analysis):
        """Send workflow logs and YAML to multiple LLMs for correction."""
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self.api_key}"
        }

        responses = {}
        log_dir = f"logs/{self.workflow_id}/responses"
        os.makedirs(log_dir, exist_ok=True)

        for model in self.models:
            stored_response = {}
            print(f"Analyzing using the model: {model}")
            payload = {
    "model": model,
    "messages": [
        {
            "role": "user",
            "content": (
                "Given the identified issues and their general descriptions, along with the Pegasus-WMS workflow failure logs and the original workflow YAML file, "
                "provide detailed corrections to resolve the issues and optimize the workflow. "
                "Use the provided general analysis to ensure your corrections align with the identified problems. \n\n"
                
                "For each issue, ensure your response includes the following structure:\n"
                "{\n"
                "  \"problems_and_solutions\": [\n"
                "    {\n"
                "      \"problem\": \"Description of the issue.\",\n"
                "      \"solution\": \"Specific, detailed steps to resolve the issue.\",\n"
                "      \"explanation\": \"Why the issue occurred and how the solution fixes it.\",\n"
                "      \"error_level\": \"site/replica/transformation/workflow/other\",\n"
                "      \"priority\": \"high/medium/low\",\n"
                "      \"level\": \"user/system\",\n"
                "      \"file_path\": \"Absolute path to the corrected file.\"\n"
                "    },\n"
                "    ...\n"
                "  ],\n"
                "  \"corrected_workflow\": \"The corrected workflow in YAML format as a string.\",\n"
                "  \"execution_pipeline\": {\n"
                f"    \"bash_script\": \"A Bash script containing step-by-step commands to execute the corrected workflow, this workflow already existe and stored by default in logs/{self.workflow_id}/Corrected_workflow.json not workflow dir, including submission and validation.\"\n"
                "  },\n"
                "  \"confidence_score\": {\n"
                "    \"score\": 0.9,\n"
                "    \"explanation\": \"This score is based on the model's confidence in resolving the identified problems and optimizing the workflow.\"\n"
                "  }\n"
                "}\n\n"
                
                "Base your corrections and optimizations on the following inputs:\n"
                "- **General Analysis:**\n"
                f"{general_analysis}\n\n"
                "- **Logs:**\n"
                f"{logs}\n\n"
                "- **Workflow:**\n"
                f"{workflow}\n"
            )
        }
    ]
}

            try:
                start = time.time()
                response = requests.post(self.api_url, json=payload, headers=headers)
                end = time.time()

                if response.status_code == 200:
                    responses[model] = response.json()
                    stored_response[model] = response.json()
                    stored_response["processing_time"] = end - start
                    print(TerminalColor.GREEN.apply(f"Response received from {model}"))

                    with open(f"{log_dir}/{model.replace('/', '_')}.json", "w") as f:
                        json.dump(stored_response, f, indent=2)
                else:
                    print(TerminalColor.RED.apply(f"Error {response.status_code} from {model}: {response.text}"))
                    responses[model] = None
            except requests.exceptions.RequestException as e:
                print(TerminalColor.RED.apply(f"Request failed for {model}: {str(e)}"))
                responses[model] = None

        return responses

    def extract_workflow_info(self, json_payload):
        """Extract structured information from LLM response."""
        assistant_message = json_payload.get("choices", [])[0].get("message", {}).get("content", "")
        cleaned_json = assistant_message.strip().replace("```json\n", "").replace("```", "")
        cleaned_json = json.loads(cleaned_json)

        if isinstance(cleaned_json, dict):
            json_file = f"logs/{self.workflow_id}/Corrected_workflow.json"
            with open(json_file, "w") as f:
                json.dump(cleaned_json, f, indent=2)
            print(TerminalColor.GREEN.apply(f"Saved Corrected workflow {json_file}"))
            return cleaned_json
        else:
            return {"error": "Input data is not a valid dictionary"}

    def process_workflow(self, general_analysis):
        """Perform the complete workflow analysis and correction process."""
        try:
            logs = self.run_pegasus_analyzer()
            if logs:
                print("Pegasus Analyzer Logs:")
                original_yaml_path = self.find_yaml_file()
                original_workflow = self.load_workflow_yaml(original_yaml_path)

                llm_responses = self.send_logs_and_workflow_to_llm(logs, original_workflow, general_analysis)

                print(TerminalColor.BLUE.apply("\nSummary of Model Responses:"))
                for model, response in llm_responses.items():
                    if response:
                        print(TerminalColor.GREEN.apply(f"- {model}: SUCCESS"))
                        p.pprint(self.extract_workflow_info(response))
                    else:
                        print(TerminalColor.RED.apply(f"- {model}: FAILED"))
            else:
                print("Failed to retrieve logs from Pegasus Analyzer.")
        except Exception as e:
            print(f"Error: {e}")


# Example Usage
if __name__ == "__main__":
    workflow_manager = PegasusWorkflowManagerPlanner(
        workflow_dir="/home/hsafri/LLM-Fine-Tune/hsafri/pegasus/falcon-7b/run0001",
        api_url="https://api.together.xyz/v1/chat/completions",
        api_key="your_api_key_here",
        workflow_id="run0001"
    )
    workflow_manager.process_workflow("General analysis of issues here")
