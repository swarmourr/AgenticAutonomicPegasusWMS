from llama_index.llms.together import TogetherLLM
from llama_index.core.agent.workflow import FunctionAgent
from typing import Dict, Any, List, Optional
import yaml
import json
import re
import logging
from datetime import datetime
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import os

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MAPE-K")

# System prompt existant
mape_k_system_prompt = """
You are a MAPE-K (Monitor, Analyze, Plan, Execute, Knowledge) agent specialized in managing scientific and computational workflows.
You will receive notifications about workflow failures and are responsible for analyzing the logs, planning a solution, and executing the fix.
- You are activated when a workflow failure event occurs.
- Each event contains workflow logs, error messages, workflow descriptors, and generated execution code.
- You do not actively poll for issues; instead, you respond to external notification events.
- The monitoring phase is handled by external sensors that feed data to you.

Example of a workflow failure event:
{
  "timestamp": "2025-04-25 14:50:19",
  "workflow_id": "123e4567-e89b-12d3-a456-426614174000",
  "directory": "/path/to/workflow/run_directory",
  "failed_tasks": [
    {
      "task_id": "Task_ID0001",
      "status": "Failed",
      "failure_reason": "Task terminated due to [REASON]. Details: [ADDITIONAL_INFORMATION].",
      "site": "compute_node_01",
      "cmd": "/path/to/script_or_command.sh",
      "platform_info": "x86_64_Linux",
      "software_version": "App v2.3.1",
      "task_priority": 20
    }
  ],
  "analyzer_output": "Execution summary: total tasks executed, number succeeded, number failed, number pending, with reasons listed for failures.",
  "original_yaml": "/path/to/workflow/descriptor.yml",
  "generated_code": "# Auto-generated code snippet for workflow orchestration\n..."
}

Goal:

1. ANALYZE:
- When triggered, identify the root causes of workflow failures by examining log patterns, event data, and execution code.
- Distinguish between **user implementation errors** (e.g., coding/configuration issues) and **system errors** (e.g., resource limits, environment instability).
- For **user errors**: Identify the problems, suggest corrections, and propose fixes.
- For **system errors**: Explain the system limitations encountered and recommend adjustments to resources or execution settings.
- Prioritize solutions based on workflow requirements and available resources.

2. PLAN:
- Develop a solution strategy based on the analysis.
- Generate specific, actionable plans with concrete steps.
- Update resource allocations when necessary.
- Correct workflow code if implementation issues are found.
- Provide clear explanations for each proposed change.
- Ensure the plan is executable and can be followed step-by-step.

3. EXECUTE:
- Apply the fixes and resubmit workflows.
Privde a json object with the following structure:
{ "steps": [ "list of steps can also consider the tools " ], "list_of_command": [ "list of command to run" ] }
- All corrections must be made **directly inside the provided Python code** (`generated_code`) included with the event.
- You must not create new code files unless explicitly instructed.
- In the execution output, you must include the **full corrected Python code** after applying the necessary changes.
- Submit corrected workflows through a workflow management agent.
- Monitor execution status via event notifications.
- Document all changes for user reference.
- Maintain an execution history to build knowledge.

4. KNOWLEDGE:
- Build and maintain understanding of common workflow issues.
- Continuously learn from past workflow failures and resolutions to improve future handling.

MANDATORY OUTPUT FORMAT:

You must **always** return your output strictly as a valid JSON object with the following structure and the code corrections must be returned as full code, not as a diff or fragment:

{
  "analysis": {
    "root_cause": "string",
    "error_type": "user_error | system_error",
    "summary": "string"
  },
  "plan": {
    "proposed_changes": "string",
    "steps": [ "list of steps" ],
    "rationale": "string"
  },
  "execution": {
    "fix_status": "applied | not_applied",
    "steps": [ "list of steps" ],
    "commands": [ "list of commands" ],
    "corrected_generated_code": "string containing the new full Python code (never use ...rest of the generated code... or ellipsis, always provide the full executable code)",
    "resubmission_status": "pending | submitted | failed"
  },
  "knowledge_update": {
    "insights": "string",
    "actionable_lessons": "string"
  }
}

⚠️ In the field "corrected_generated_code", you must always return the full, complete, and executable Python code after applying the corrections. Never use "...rest of the generated code...", "rest of the code", "remaining code", "the rest of the code remains the same", "the rest of the generated code remains the same", or any ellipsis or comment implying incomplete code. The code must be ready to run as-is, from the first import to the last line. If you modify a function or class, return the entire file content, not just the diff or a code fragment.

⚠️ No additional text, no markdown, no explanations outside of the JSON object. Only the JSON object must be returned. 
Failure to comply can cause system malfunction.

Tools: 

- You have access also to the this Tools:
    1. PegasusWorkflowGenerator: A tool to generate Pegasus workflows from python file.You can use this tool to generate workflows based on the provided Python code. PegasusworkflowGenerator.py <python_file_path>.
    2. PegasusPlanSubmission: A tool to submit Pegasus workflows. You can use this tool to submit the generated workflows for execution. PegasusPlanSubmission.py <Genrated_yaml_workflow_file_path>.
    """

class WorkflowAnalyzer:
    """Classe pour analyser les workflows Pegasus"""
    
    @staticmethod
    def parse_analyzer_output(output: str) -> Dict[str, Any]:
        """Parse la sortie de l'analyseur Pegasus"""
        try:
            stats = {}
            current_section = None
            
            for line in output.split('\n'):
                line = line.strip()
                
                # Ignorer les lignes vides et les séparateurs
                if not line or line.startswith('*'):
                    continue
                
                # Gérer les sections
                if line.endswith('Summary'):
                    current_section = 'summary'
                    continue
                elif "Held jobs' details" in line:
                    current_section = 'held_jobs'
                    continue
                
                # Parser les lignes contenant des informations
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Nettoyer et convertir les valeurs numériques
                    if '(' in value:
                        value = value.split('(')[0].strip()
                    
                    try:
                        # Tenter de convertir en nombre si possible
                        if value.replace('.', '').isdigit():
                            value = float(value) if '.' in value else int(value)
                    except ValueError:
                        pass
                    
                    stats[key] = value
            
            return stats
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse de la sortie: {str(e)}")
            return {}

class PegasusAgent:
    """Agent principal pour la gestion des workflows Pegasus"""
    
    def __init__(self):
        """Initialisation de l'agent avec la configuration"""
        load_dotenv()  # Charger les variables d'environnement
        
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError("La clé API Together n'est pas définie dans le fichier .env")
        
        self.llm = TogetherLLM(
            model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            api_key=api_key,
            temperature=0.7,
            max_tokens=10000
        )
        
        self.workflow_analyzer = WorkflowAnalyzer()

    async def analyze_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse un événement de workflow"""
        try:
            if "analyzer_output" in event:
                event["parsed_analysis"] = self.workflow_analyzer.parse_analyzer_output(
                    event["analyzer_output"]
                )
            event_str = json.dumps(event, indent=2)
            
            # Appel direct du LLM (prompt classique)
            response = await self.llm.acomplete(
                prompt=f"{mape_k_system_prompt}\n\nEvent:\n{event_str}"
            )
            return self._validate_response(response.text)
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse de l'événement: {str(e)}")
            return {
                "error": str(e),
                "status": "failed"
            }

    def _validate_response(self, response: str) -> Dict[str, Any]:
        """Valide et nettoie la réponse de l'agent"""
        try:
            # Extraire le JSON si la réponse est une chaîne contenant du JSON
            if isinstance(response, str):
                # Rechercher un objet JSON valide dans la réponse
                json_match = re.search(r'(\{.*\})', response, re.DOTALL)
                if json_match:
                    response = json.loads(json_match.group(1))
                else:
                    try:
                        response = json.loads(response)
                    except json.JSONDecodeError:
                        raise ValueError("La réponse ne contient pas de JSON valide")
            
            required_keys = ["analysis", "plan", "execution", "knowledge_update"]
            for key in required_keys:
                if key not in response:
                    raise ValueError(f"Clé manquante dans la réponse: {key}")
            
            corrected_code = response["execution"]["corrected_generated_code"]
            print("Code corrigé:", corrected_code)
            forbidden_patterns = [
                "rest of the generated code",
                "rest of the code",
                "remaining code",
                "the rest of the code remains the same",
                "...",
                "…"
            ]
            if any(pattern in corrected_code for pattern in forbidden_patterns):
                print("⚠️ Le code retourné n'est pas complet. Merci de reformuler la demande ou d'ajuster le prompt système.")
            
            return response
        except Exception as e:
            logger.error(f"Erreur de validation de la réponse: {str(e)}")
            return {
                "error": "Invalid response format",
                "details": str(e)
            }

    def generate_corrected_code(self, event: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """Génère du code corrigé basé sur l'analyse"""
        # Méthode à implémenter pour générer le code corrigé
        # Pour l'instant, renvoie simplement le code original
        return event.get("generated_code", "# Code non disponible")

async def merge_codes_with_llm(llm, original_code: str, partial_code: str) -> str:
    merge_prompt = (
        "PLease correct and merge the following two pieces of code  by modifying the original code and return a full, complete, and executable code:\n\n"
        "Your task is to merge the corrections from the partial code into the original code, and keep the original code as is. "
        "you must not change the original code, but you can add or modify the partial code. "
        "and return the full, complete, and executable Python code. "
        "Do not use any ellipsis or comments like 'rest of the code'. "
        "Return only the full code, ready to run from the first import to the last line.\n\n"
        "Original code:\n"
        f"{original_code}\n\n"
        "Partial corrected code:\n"
        f"{partial_code}\n\n"
        "Merged, full, executable code:"
    )
    response = await llm.acomplete(prompt=merge_prompt , temperature=0.1, max_tokens=10000)
    return response.text.strip()

async def main():
    """Fonction principale pour tester l'agent"""
    
    # Exemple d'événement de workflow échoué plus détaillé
    example_event = {
  "timestamp": "2025-04-25 14:50:19",
  "workflow_id": "ac9b9fff-3d93-437f-be89-005469f944e7",
  "directory": "/home/hsafri/LLM-Fine-Tune/generated_workflows/hsafri/pegasus/falcon-7b/run0055",
  "held_jobs": [
    {
      "job_id": "FineTuneLLM_ID0000001",
      "status": "Held",
      "hold_reason": "Error from slot1_1@testpool-gpu-20250425144144.novalocal: Job has gone over cgroup memory limit of 10624 megabytes. Last measured usage: 404 megabytes.  Consider resubmitting with a higher request_memory.",
      "site": "condorpool",
      "cmd": "/home/hsafri/LLM-Fine-Tune/generated_workflows/hsafri/pegasus/falcon-7b/run0055/00/00/FineTuneLLM_ID0000001.sh",
      "condor_platform": "$CondorPlatform: x86_64_AlmaLinux8 $",
      "condor_version": "$CondorVersion: 24.6.0 2025-03-05 BuildID: 790852 PackageID: 24.6.0-1 $",
      "job_priority": 20
    }
  ],
  "analyzer_output": "Database version: '5.1.0dev' (sqlite:////home/hsafri/LLM-Fine-Tune/generated_workflows/hsafri/pegasus/falcon-7b/run0055/falcon-7b-0.stampede.db)\n\n************************************Summary*************************************\n\n Submit Directory   : /home/hsafri/LLM-Fine-Tune/generated_workflows/hsafri/pegasus/falcon-7b/run0055\n Workflow Status    : running\n Total jobs         :      5 (100.00%)\n # jobs succeeded   :      2 (40.00%)\n # jobs failed      :      0 (0.00%)\n # jobs held        :      1 (20.00%)\n # jobs unsubmitted :      3 (60.00%)\n\n*******************************Held jobs' details*******************************\n\n=============================FineTuneLLM_ID0000001==============================\n\nsubmit file            : FineTuneLLM_ID0000001.sub\nlast_job_instance_id   : 7\nreason                 :  Error from slot1_1@testpool-gpu-20250425144144.novalocal: Job has gone over cgroup memory limit of 10624 megabytes. Last measured usage: 404 megabytes.  Consider resubmitting with a higher request_memory.\n\n**************************************Done**************************************\n\npegasus-analyzer: end of status report\n\n\n",
  "original_yaml": "/home/hsafri/LLM-Fine-Tune/generated_workflows/hsafri/pegasus/falcon-7b/run0055/falcon-7b.yml",
  "generated_code": "from Pegasus.api import *\nimport os\n\nclass Falcon_7bWorkflow:\n    \"\"\"\n    A Pegasus workflow for falcon-7b.\n    \"\"\"\n    def __init__(self, base_dir=\".\"):\n        \"\"\"\n        Initialize the workflow, sites, replicas, transformations, and job containers.\n        \n        :param base_dir: Base directory for the workflow (default: current directory)\n        \"\"\"\n        self.base_dir = base_dir\n        \n        # Change to the workflow directory for all operations\n        # This ensures catalog files are written to the correct location\n        os.chdir(self.base_dir)\n        \n        self.wf = Workflow(name=\"falcon-7b\")\n        self.sites = SiteCatalog()\n        self.replicas = ReplicaCatalog()\n        self.transformations = TransformationCatalog()\n        self.files = {}\n        self.jobs = {}\n\n    def build_sites(self):\n        local = Site(\"local\")\n        local.add_directories(Directory(Directory.SHARED_SCRATCH, \"/home/hsafri/LLM-Fine-Tune/scratch\").add_file_servers(FileServer(\"file:///home/hsafri/LLM-Fine-Tune/scratch\", Operation.ALL)))\n        local.add_directories(Directory(Directory.LOCAL_STORAGE, \"/home/hsafri/LLM-Fine-Tune/output\").add_file_servers(FileServer(\"file:///home/hsafri/LLM-Fine-Tune/output\", Operation.ALL)))\n        self.sites.add_sites(local)\n\n        condorpool = Site(\"condorpool\")\n        condorpool.add_profiles(Namespace.CONDOR, key=\"universe\", value=\"vanilla\")\n        condorpool.add_profiles(Namespace.PEGASUS, key=\"style\", value=\"condor\")\n        self.sites.add_sites(condorpool)\n\n    def build_replicas(self):\n        self.replicas.add_replica(\"local\", \"pegasus_data\", \"/home/hsafri/LLM-Fine-Tune/data/data.json\")\n\n    def build_transformations(self):\n        container = Container(\"FineTuneLLM\", Container.SINGULARITY, \"docker://swarmourr/finetune-pegasus:amd64\", image_site=\"docker_hub\")\n        self.transformations.add_containers(container)\n\n        transformation = Transformation(\"FineTuneLLM\", site=\"condorpool\", pfn=\"/home/hsafri/LLM-Fine-Tune/bin/finetune.py\", is_stageable=True)\n        transformation.add_profiles(Namespace.PEGASUS, key=\"cores\", value=\"4\")\n        transformation.add_profiles(Namespace.PEGASUS, key=\"memory\", value=\"10600\")\n        transformation.add_profiles(Namespace.PEGASUS, key=\"gpus\", value=\"1\")\n        self.transformations.add_transformations(transformation)\n\n    def build_jobs(self):\n        job = Job(\"FineTuneLLM\", _id=\"ID0000001\")\n        job.add_args(\"--data_path\", \"pegasus_data\", \"--model_name\", \"tiiuae/falcon-7b\", \"--output_dir\", \"tiiuae/falcon-7b\", \"--num_train_epochs\", \"3\", \"--batch_size\", \"4\", \"--save_steps\", \"5000\", \"--learning_rate\", \"3e-05\", \"--gpu\", \"1\", \"--auth_token\", \"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\")\n        job.add_inputs(self._get_file(\"pegasus_data\"))\n        job.add_outputs(self._get_file(\"falcon-7b.zip\"), stage_out=True, register_replica=True)\n        self.jobs[\"ID0000001\"] = job\n        self.wf.add_jobs(job)\n\n    def _get_file(self, name):\n        if name not in self.files:\n            self.files[name] = File(name)\n        return self.files[name]\n\n    def write(self):\n        \"\"\"\n        Write the site, replica, transformation, and workflow to their respective catalogs and files.\n        \"\"\"\n        # Write all catalog files\n        self.sites.write()\n        self.replicas.write()\n        self.transformations.write()\n        self.wf.write()\n\nif __name__ == \"__main__\":\n    import os\n    import sys\n    \n    # Get the directory where this script is located\n    current_dir = os.path.dirname(os.path.abspath(__file__))\n    \n    # Initialize workflow with current directory\n    w = Falcon_7bWorkflow(base_dir=current_dir)\n    w.build_sites()\n    w.build_replicas()\n    w.build_transformations()\n    w.build_jobs()\n    w.write()\n    print(\"Workflow generated and written successfully in: \" + current_dir)"
}



    try:
        agent = PegasusAgent()
        result = await agent.analyze_event(example_event)
        corrected_code = result.get("execution", {}).get("corrected_generated_code", "")

        # If the code is incomplete, merge with the original
        if (
            "..." in corrected_code
            or "rest of the code" in corrected_code
            or "rest of the generated code" in corrected_code
            or "set_stdout(...)" in corrected_code
            or "add_args(...)" in corrected_code
        ):
            print("⚠️ Merging with original code to get a complete version.")
            original_code = example_event.get("generated_code", "")
            merged_code = await merge_codes_with_llm(agent.llm, original_code, corrected_code)
            result["execution"]["corrected_generated_code"] = merged_code
           
        # Write the code to a .py file named after the workflow
        py_filename = f"workflowtest.py"
        with open(py_filename, "w") as pyfile:
                pyfile.write(corrected_code)
        print(f"✅ Python code written to {py_filename}")

        print("\n=== Analysis Results ===")
        print(json.dumps(result, indent=2))
        with open("pegasus_analysis_result.json", "w") as f:
            json.dump(result, f, indent=2)
        if "analysis" in result and "root_cause" in result["analysis"]:
            print("\n=== Recommended Actions Summary ===")
            print(f"Root cause: {result['analysis']['root_cause']}")
            if "plan" in result and "steps" in result["plan"]:
                print("\nSteps to follow:")
                for i, step in enumerate(result["plan"]["steps"], 1):
                    print(f"{i}. {step}")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    # Exécuter le main asynchrone
    asyncio.run(main())