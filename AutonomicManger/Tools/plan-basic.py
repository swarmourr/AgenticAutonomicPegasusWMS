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
import subprocess
import json
import time
import os
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
    def __init__(self, workflow_dir, api_url, api_key, workflow_id ,corrected_yaml_path="corrected_workflow.yml"):
        """
        Initialize the PegasusWorkflowManager.
        :param workflow_dir: Path to the workflow directory.
        :param api_url: API URL for the LLM service.
        :param api_key: API key for authentication.
        :param corrected_yaml_path: Path to save the corrected workflow YAML file.
        """
        self.workflow_dir = workflow_dir
        self.workflow_id = workflow_id
        self.api_url = api_url
        self.api_key = api_key
        self.corrected_yaml_path = corrected_yaml_path

    def run_pegasus_analyzer(self):
        """
        Run the Pegasus Analyzer to generate workflow logs.
        :return: Combined standard output and error output from the analyzer.
        """
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
        """
        Find the workflow YAML file in the given directory.
        :return: Path to the workflow YAML file.
        :raises FileNotFoundError: If no YAML file is found in the directory.
        """
        wf_name = self.workflow_dir.split("/")[-2]
        for file_name in os.listdir(self.workflow_dir):
            if (file_name.endswith(wf_name + '.yml') or file_name.endswith(wf_name + '.yaml')) and file_name != 'braindump.yml':      
                   return os.path.join(self.workflow_dir, file_name)
        raise FileNotFoundError(f"No YAML file found in directory: {self.workflow_dir}")

    def clean_data(self,data):
        """
        Recursively clean ruamel.yaml objects to plain Python types.
        :param data: Data loaded by ruamel.yaml
        :return: Cleaned data with plain Python types
        """
        if isinstance(data, Mapping):  # Handle dictionaries (CommentedMap in ruamel)
            return {key: self.clean_data(value) for key, value in data.items()}
        elif isinstance(data, list):  # Handle lists
            return [self.clean_data(item) for item in data]
        else:  # Handle scalar values
            return data
        
    def load_workflow_yaml(self, file_path):
        """
        Load the workflow YAML file.
        :param yaml_file_path: Path to the workflow YAML file.
        :return: Dictionary containing the workflow data.
        :raises FileNotFoundError: If the YAML file does not exist.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Workflow file not found: {file_path}")
        yaml = YAML()
        yaml.preserve_quotes = True  # Optional: Preserve quotes from the YAML file
        with open(file_path, 'r') as yaml_file:
            raw_data = yaml.load(yaml_file)
        return self.clean_data(raw_data)

    def save_corrected_workflow(self, corrected_workflow):
        """
        Save the corrected workflow YAML to a file.
        :param corrected_workflow: Corrected workflow data.
        """
        with open(self.corrected_yaml_path, 'w') as file:
            yaml.safe_dump(corrected_workflow, file, default_flow_style=False)

    def send_logs_and_workflow_to_llm(self, logs, workflow,general_analysis):
        """
        Send workflow logs and YAML to the LLM for correction.
        :param logs: Logs from the Pegasus Analyzer.
        :param workflow: Original workflow YAML data.
        :return: Response from the LLM.
        """
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self.api_key}"
        }

        payload = {
    "model": "Qwen/Qwen2.5-72B-Instruct-Turbo",
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



        response = requests.post(self.api_url, json=payload, headers=headers)
        if response.status_code == 200:
            #print(response.json())
            return response.json()
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None

    def extract_yaml_from_response(self, llm_response):
        """
        Extract the corrected YAML from the LLM response.
        :param llm_response: The JSON response from the LLM.
        :return: Corrected workflow data as a Python dictionary.
        :raises ValueError: If the response is malformed or the YAML parsing fails.
        """
        assistant_message = llm_response.get("choices", [])[0].get("message", {}).get("content", "")
        print(assistant_message)
        if not assistant_message:
            raise ValueError("No content found in LLM response.")

        # Clean the YAML content by removing unwanted Markdown code block markers
        cleaned_yaml = assistant_message.strip()
        cleaned_yaml = cleaned_yaml.replace("```yaml", "").replace("```", "")

        # Attempt to fix common YAML syntax issues
        cleaned_yaml = self.fix_yaml_formatting(cleaned_yaml)

        try:
            corrected_workflow = yaml.safe_load(cleaned_yaml)
            return corrected_workflow
        except yaml.YAMLError as e:
            print("Failed to parse YAML content.")
            raise ValueError(f"Failed to parse YAML content: {e}")
            
    def extract_workflow_info(self,json_payload):
        # Load the payload (assuming the payload is in the correct format as a string)
        assistant_message = json_payload.get("choices", [])[0].get("message", {}).get("content", "")
        cleaned_json = assistant_message.strip()
        cleaned_json = cleaned_json.replace("```json\n", "").replace("```", "")
        cleaned_json = json.loads(cleaned_json)
        if isinstance(cleaned_json, dict):
            
            # Extract 'problems_and_solutions' and 'corrected_workflow'
            problems_and_solutions = cleaned_json.get("problems_and_solutions", [])
            confidence_score = cleaned_json.get("confidence_score", "")
            corrected_workflow=cleaned_json.get("corrected_workflow", "")
            execution_pipeline=cleaned_json.get("execution_pipeline", "")
            # Prepare the structured response
            extracted_info = {
                "problems_and_solutions": [],
                "corrected_workflow": cleaned_json.get("corrected_workflow", ""),
                "execution_pipeline" :execution_pipeline,
                "confidence_score": confidence_score
            }

            for problem_solution in problems_and_solutions:
                problem_info = {
                    "problem": problem_solution.get("problem", ""),
                    "solution": problem_solution.get("solution", ""),
                    "explanation": problem_solution.get("explanation", ""),
                    "priority": problem_solution.get("priority", ""),
                    "error_level": problem_solution.get("error_level", ""),
                    "file_path": problem_solution.get("file_path", ""),
                    "level": problem_solution.get("level", "")
                }
                extracted_info["problems_and_solutions"].append(problem_info)
            json_file = f"logs/{self.workflow_id}/Corrected_workflow.json"
            with open(json_file, "w") as f:
                 json.dump(corrected_workflow, f, indent=2)
            print(TerminalColor.GREEN.apply(f"Saved Corrected workflow {json_file}"))
            return extracted_info
        else:
            return {"error": "Input data is not a valid dictionary"}


    def fix_yaml_formatting(self, yaml_content):
        """
        Fix common YAML syntax issues, such as missing colons or incorrect indentation.
        :param yaml_content: Raw YAML content as a string.
        :return: Corrected YAML content.
        """
        # Replace invalid lines or adjust formatting
        fixed_content = yaml_content

        # Example of replacing invalid lines like ' - - - x-pegasus' to valid YAML format
        #fixed_content = fixed_content.replace("- - - x-pegasus", "- x-pegasus")

        # You can add more rules here to fix specific issues you encounter.

        return fixed_content

    def process_workflow(self,general_analysis):
        """
        Perform the complete workflow analysis and correction process.
        """
        try:
            logs = self.run_pegasus_analyzer()
            if logs:
                print("Pegasus Analyzer Logs:")
                #print(logs)
            
                original_yaml_path = self.find_yaml_file()
                original_workflow = self.load_workflow_yaml(original_yaml_path)

                llm_response = self.send_logs_and_workflow_to_llm(logs, original_workflow,general_analysis)
                if llm_response:
                        #try:
                        #corrected_workflow = self.extract_yaml_from_response(llm_response)
                        #self.save_corrected_workflow(corrected_workflow)
                        # Write JSON file
                        print(f"Response Parsed PLanner")
                        p.pprint(llm_response)
                        p.pprint(self.extract_workflow_info(llm_response))
                        #except ValueError as e:
                        #print(f"Error processing LLM response: {e}")
                else:
                    print("No response from LLM or correction process failed.")
            else:
                print("Failed to retrieve logs from Pegasus Analyzer.")
        except Exception as e:
            print(f"Error: {e}")


# Example Usage
"""if __name__ == "__main__":
    workflow_manager = PegasusWorkflowManager(
        workflow_dir="//home//hsafri//LLM-Fine-Tune//hsafri//pegasus//falcon-7b//run0001",
        #workflow_dir="//home//hsafri//LLM-Fine-Tune//generated_workflows//hsafri/pegasus//falcon-7b//run0013",
        api_url="https://api.together.xyz/v1/chat/completions",
        api_key="9330fcf33a0d19c088c90a6799e1208f30c405627bc12cf27c4d4eaba36dcb96"
    )
    workflow_manager.process_workflow()
"""