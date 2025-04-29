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
from uuid import uuid4
from dataclasses import dataclass

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
- Provide a JSON object with the following structure:
{
    "steps": [ "list of steps, can also consider the tools" ],
    "list_of_command": [ "list of command to run" ]
}
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
if no more information needed return:
{
  "analysis": {
    "root_cause": "Missing data preprocessing step in training pipeline",
    "error_type": "user_error",
    "summary": "The input data was not normalized before training, causing unstable gradients."
  },
  "plan": {
    "proposed_changes": "Add normalization step before feeding data into the model",
    "steps": [
      "Update data loader script",
      "Re-train model with normalized inputs"
    ],
    "rationale": "Normalizing data ensures stable training and faster convergence"
  },
  "execution": {
    "fix_status": "applied",
    "steps": [
      "Modified data loader script",
      "Re-executed training job"
    ],
    "commands": [
      "python bin/preprocess.py",
      "python bin/finetune.py"
    ],
    "corrected_generated_code": "def preprocess(data):\n    return (data - data.mean()) / data.std()\n# full script here...",
    "resubmission_status": "submitted"
  },
  "knowledge_update": {
    "insights": "Data normalization is critical for model stability",
    "actionable_lessons": "Include normalization in all future data pipelines"
  },
  "logs_needed": "NO",
  "workflow_id": "wf_123456789",
  "directory": "/tmp/job_execution_dir/",
  "workflow_directory": "/home/hsafri/LLM-Fine-Tune/generated_workflows/hsafri/pegasus/falcon-7b/run0055",
  "transformation_parent_path": [
    "/home/hsafri/LLM-Fine-Tune/bin/"
  ]
}



If you need more logs for the failed jobs you can ask for more logs for the failed jobs in this format:
{
    "job_id": "job_id",
    "logs": "YES",
    "directory": "directory"
}

⚠️ In the field "corrected_generated_code", you must always return the full, complete, and executable Python code after applying the corrections. Never use "...rest of the generated code...", "rest of the code", "remaining code", "the rest of the code remains the same", "the rest of the generated code remains the same", or any ellipsis or comment implying incomplete code. The code must be ready to run as-is, from the first import to the last line. If you modify a function or class, return the entire file content, not just the diff or a code fragment.

⚠️ No additional text, no markdown, no explanations outside of the JSON object. Only the JSON object must be returned. 
Failure to comply can cause system malfunction.

⚠️ If you need more details about the failed job you can ask for more logs for the failed jobs in this format:
{ "job_id": "job_id" , "status" :"logs needed" } and you will get the logs for this job.

⚠️ Whenever a file needs to be modified—such as during transformations or in the replicas (datat) workflow—always use the original file and directory paths from the submitted nodes, not the path of the node where the job is executed.

⚠️ Please for execution can u provide also intermidiate steps and commands to run, and the full code with the modifications like file creations .....

⚠️ Please create file and manage names of file in teh workflow generator and workflow submission, and also in the execution of the code.

Tools: 

- You have access also to these Tools:
        1. PegasusWorkflowGenerator: A tool to generate Pegasus workflows from python file. You can use this tool to generate workflows based on the provided Python code. PegasusworkflowGenerator.py <python_file_path>.
        2. PegasusPlanSubmission: A tool to submit Pegasus workflows. You can use this tool to submit the generated workflows for execution. PegasusPlanSubmission.py <Genrated_yaml_workflow_file_path>.
        3. JobsLogs: A tool to get the logs of the failed jobs. You can use this tool to get the logs of the failed jobs. JobsLogs.py <job_id>.
        4. TransformationsCodeSource: A tool to get the code of the transformations. You can use this tool to get the code of the transformations. TransformationsCodeSource.py <transformation_submitnode_path>.
        """

@dataclass
class LogRequest:
    """Représente une demande de logs"""
    context_id: str
    job_id: str
    workflow_id: str
    timestamp: datetime
    directory: str
    previous_analysis: Optional[Dict[str, Any]] = None

class LogContextManager:
    """Gestionnaire de contextes pour les demandes de logs"""
    
    def __init__(self):
        self._contexts: Dict[str, LogRequest] = {}
        self._workflow_contexts: Dict[str, List[str]] = {}  # workflow_id -> [context_ids]
        
    def create_context(self, workflow_id: str, job_id: str, directory: str, 
                      previous_analysis: Optional[Dict[str, Any]] = None) -> str:
        """Crée un nouveau contexte pour une demande de logs"""
        context_id = str(uuid4())
        log_request = LogRequest(
            context_id=context_id,
            job_id=job_id,
            workflow_id=workflow_id,
            timestamp=datetime.now(),
            directory=directory,
            previous_analysis=previous_analysis
        )
        
        self._contexts[context_id] = log_request
        if workflow_id not in self._workflow_contexts:
            self._workflow_contexts[workflow_id] = []
        self._workflow_contexts[workflow_id].append(context_id)
        
        return context_id
    
    def get_context(self, context_id: str) -> Optional[LogRequest]:
        """Récupère un contexte par son ID"""
        return self._contexts.get(context_id)
    
    def get_workflow_contexts(self, workflow_id: str) -> List[LogRequest]:
        """Récupère tous les contextes pour un workflow donné"""
        context_ids = self._workflow_contexts.get(workflow_id, [])
        return [self._contexts[cid] for cid in context_ids if cid in self._contexts]
    
    def update_context(self, context_id: str, 
                      additional_logs: Optional[str] = None,
                      analysis_result: Optional[Dict[str, Any]] = None) -> bool:
        """Met à jour un contexte avec de nouveaux logs ou résultats"""
        if context_id not in self._contexts:
            return False
        
        context = self._contexts[context_id]
        if additional_logs:
            context.logs = additional_logs
        if analysis_result:
            context.previous_analysis = analysis_result
        return True

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
        self.log_context_manager = LogContextManager()

    async def analyze_event(self, event: Dict[str, Any], 
                            context_id: Optional[str] = None) -> Dict[str, Any]:
        """Analyse un événement de workflow"""
        try:
            # Si un context_id est fourni, récupérer le contexte précédent
            previous_context = None
            if context_id:
                previous_context = self.log_context_manager.get_context(context_id)
                if previous_context:
                    event["previous_analysis"] = previous_context.previous_analysis

            if "analyzer_output" in event:
                event["parsed_analysis"] = self.workflow_analyzer.parse_analyzer_output(
                    event["analyzer_output"]
                )
            event_str = json.dumps(event, indent=2)
            
            # Inclure l'historique des contextes dans le prompt
            workflow_id = event.get("workflow_id")
            workflow_contexts = self.log_context_manager.get_workflow_contexts(workflow_id)
            context_history = [
                {
                    "context_id": ctx.context_id,
                    "job_id": ctx.job_id,
                    "timestamp": ctx.timestamp.isoformat(),
                    "previous_analysis": ctx.previous_analysis
                }
                for ctx in workflow_contexts
            ]
            
            # Appel direct du LLM (prompt classique)
            response = await self.llm.acomplete(
                prompt=f"{mape_k_system_prompt}\n\nContext History:\n{json.dumps(context_history, indent=2)}\n\nEvent:\n{event_str}"
            )

            print("Response from LLM:", response.text)
            validated_response = self._validate_response(response.text)


            if validated_response.get("logs_needed") == "YES":
                # Créer un nouveau contexte
                new_context_id = self.log_context_manager.create_context(
                    workflow_id=event["workflow_id"],
                    job_id=validated_response["job_id"],
                    directory=validated_response["directory"],
                    previous_analysis=validated_response
                )
                
                return {
                    "context_id": new_context_id,
                    "job_id": validated_response["job_id"],
                    "logs": "YES",
                    "directory": validated_response["directory"],
                    "workflow_id": event["workflow_id"]
                }

            # Réponse normale
            return {
                "analysis": validated_response.get("analysis", {}),
                "plan": validated_response.get("plan", {}),
                "execution": validated_response.get("execution", {}),
                "knowledge_update": validated_response.get("knowledge_update", {}),
                "logs_needed": "NO"
            }

        except Exception as e:
            logger.error(f"Erreur lors de l'analyse de l'événement: {str(e)}")
            return {
                "error": str(e),
                "status": "failed",
                "logs_needed": "NO"
            }

    def _validate_response(self, response: str) -> Dict[str, Any]:
        """Valide et nettoie la réponse de l'agent, gère aussi la demande de logs supplémentaires"""
        try:
            # Extraction robuste du JSON
            if isinstance(response, str):
                response_obj = _extract_json_object(response)
            else:
                response_obj = response

            # Cas 1 : Demande explicite de logs supplémentaires (format strict)
            if (
                isinstance(response_obj, dict)
                and "logs" in response_obj
                and str(response_obj.get("logs", "")).upper() == "YES"
            ):
                if "job_id" not in response_obj or "directory" not in response_obj:
                    raise ValueError("La réponse indique que des logs sont nécessaires, mais 'job_id' ou 'directory' est manquant.")
                logger.info("Le LLM demande plus de logs pour le job : %s", response_obj["job_id"])
                return response_obj

            # Cas 2 : Demande de logs via le format alternatif { "job_id": ..., "status": "logs needed" }
            if (
                isinstance(response_obj, dict)
                and str(response_obj.get("status", "")).lower() == "logs needed"
                and "job_id" in response_obj
            ):
                logger.info("Le LLM demande plus de logs (format alternatif) pour le job : %s", response_obj["job_id"])
                return response_obj

            # Cas 3 : Erreur simple
            if isinstance(response_obj, dict) and "error" in response_obj:
                logger.error(f"Erreur retournée par le LLM: {response_obj.get('error')}")
                return response_obj

            # Cas 4 : Réponse complète attendue
            required_keys = ["analysis", "plan", "execution", "knowledge_update"]
            for key in required_keys:
                if key not in response_obj:
                    raise ValueError(f"Clé manquante dans la réponse: {key}")

            # Vérification du code complet dans corrected_generated_code
            corrected_code = response_obj["execution"].get("corrected_generated_code", "")
            forbidden_patterns = [
                "rest of the generated code",
                "rest of the code",
                "remaining code",
                "the rest of the code remains the same",
                "...",
                "…"
            ]
            if any(pattern in corrected_code for pattern in forbidden_patterns):
                logger.warning("⚠️ Le code retourné n'est pas complet. Merci de reformuler la demande ou d'ajuster le prompt système.")

            return response_obj
        except json.JSONDecodeError as e:
            logger.error(f"Erreur de décodage JSON: {str(e)}")
            return {
                "error": "Invalid JSON format",
                "details": str(e)
            }
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

async def get_job_logs(job_id: str) -> str:
    """Simule la récupération des logs pour un job spécifique."""
    print(f"Simulating log retrieval for job ID: {job_id}")
    return f"Simulated logs for job ID: {job_id}"

def _extract_json_object(text: str) -> dict:
    """Extrait le premier objet JSON valide d'une chaîne."""
    import json
    start = text.find('{')
    while start != -1:
        try:
            return json.loads(text[start:])
        except json.JSONDecodeError:
            start = text.find('{', start + 1)
    raise ValueError("Aucun objet JSON valide trouvé dans la réponse.")

async def main():
    """Fonction principale pour tester l'agent"""

    # Exemple d'événement de workflow échoué plus détaillé
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

    
    example_event_1 ={
    "timestamp": "2025-04-29 14:00:50",
    "workflow_id": "d7aa7624-9cf5-453b-a042-f4f9b0f71b93",
    "directory": "/home/hsafri/LLM-Fine-Tune/generated_workflows/hsafri/pegasus/falcon-7b/run0065",
    "held_jobs": [
        {
            "job_id": "FineTuneLLM_ID0000001",
            "status": "Held",
            "hold_reason": "Transfer output files failure at execution point slot1_1@testpool-gpu-20250429134940.novalocal while sending files to access point pegasus. Details: reading from file /var/lib/condor/execute/dir_1929/falcon-7b.zip: (errno 2) No such file or directory",
            "site": "condorpool",
            "cmd": "/home/hsafri/LLM-Fine-Tune/generated_workflows/hsafri/pegasus/falcon-7b/run0065/00/00/FineTuneLLM_ID0000001.sh",
            "condor_platform": "$CondorPlatform: x86_64_AlmaLinux8 $",
            "condor_version": "$CondorVersion: 24.6.0 2025-03-05 BuildID: 790852 PackageID: 24.6.0-1 $",
            "job_priority": 20
        }
    ],
    "analyzer_output": "Database version: '5.1.0dev' (sqlite:////home/hsafri/LLM-Fine-Tune/generated_workflows/hsafri/pegasus/falcon-7b/run0065/falcon-7b-0.stampede.db)\nDatabase version: '5.1.0dev' (sqlite:////home/hsafri/LLM-Fine-Tune/generated_workflows/hsafri/pegasus/falcon-7b/run0065/falcon-7b-0.stampede.db)\nDatabase version: '5.1.0dev' (sqlite:////home/hsafri/LLM-Fine-Tune/generated_workflows/hsafri/pegasus/falcon-7b/run0065/falcon-7b-0.stampede.db)\n\n************************************Summary*************************************\n\n Submit Directory   : /home/hsafri/LLM-Fine-Tune/generated_workflows/hsafri/pegasus/falcon-7b/run0065\n Workflow Status    : running\n Total jobs         :      8 (100.00%)\n # jobs succeeded   :      2 (25.00%)\n # jobs failed      :      0 (0.00%)\n # jobs held        :      1 (12.50%)\n # jobs unsubmitted :      6 (75.00%)\n\n*******************************Held jobs' details*******************************\n\n=============================FineTuneLLM_ID0000001==============================\n\nsubmit file            : FineTuneLLM_ID0000001.sub\nlast_job_instance_id   : 6\nreason                 :  Transfer output files failure at execution point slot1_1@testpool-gpu-20250429134940.novalocal while sending files to access point pegasus. Details: reading from file /var/lib/condor/execute/dir_1929/falcon-7b.zip: (errno 2) No such file or directory\n\n*****************************Failing jobs' details******************************\n\n=============================FineTuneLLM_ID0000001==============================\n\n last state: POST_SCRIPT_FAILED\n       site: condorpool\nsubmit file: 00/00/FineTuneLLM_ID0000001.sub\noutput file: 00/00/FineTuneLLM_ID0000001.out.001\n error file: 00/00/FineTuneLLM_ID0000001.err.001\n\n-------------------------------Task #1 - Summary--------------------------------\n\nsite        : condorpool\nhostname    : testpool-gpu-20250428122556.novalocal\nexecutable  : /srv/FineTuneLLM\narguments   : -\nexitcode    : 1\nworking dir : /srv\n\n--------------Task #1 - FineTuneLLM - ID0000001 - Kickstart stderr--------------\n\n File \"/srv/./FineTuneLLM\", line 105\n    tokenizer=tokenizer\n              ^^^^^^^^^\nSyntaxError: invalid syntax. Perhaps you forgot a comma?\n\n**************************************Done**************************************\n\npegasus-analyzer: end of status report\n\n\n",
    "original_yaml": "/home/hsafri/LLM-Fine-Tune/generated_workflows/hsafri/pegasus/falcon-7b/run0065/falcon-7b.yml",
    "generated_code": "from Pegasus.api import *\nimport os\n\nclass Falcon_7bWorkflow:\n    \"\"\"\n    A Pegasus workflow for falcon-7b.\n    \"\"\"\n    def __init__(self, base_dir=\".\"):\n        \"\"\"\n        Initialize the workflow, sites, replicas, transformations, and job containers.\n        \n        :param base_dir: Base directory for the workflow (default: current directory)\n        \"\"\"\n        self.base_dir = base_dir\n        os.chdir(self.base_dir)\n        self.wf = Workflow(name=\"falcon-7b\")\n        self.sites = SiteCatalog()\n        self.replicas = ReplicaCatalog()\n        self.transformations = TransformationCatalog()\n        self.files = {}\n        self.jobs = {}\n\n    def build_sites(self):\n        local = Site(\"local\")\n        local.add_directories(Directory(Directory.SHARED_SCRATCH, \"/home/hsafri/LLM-Fine-Tune/scratch\").add_file_servers(FileServer(\"file:///home/hsafri/LLM-Fine-Tune/scratch\", Operation.ALL)))\n        local.add_directories(Directory(Directory.LOCAL_STORAGE, \"/home/hsafri/LLM-Fine-Tune/output\").add_file_servers(FileServer(\"file:///home/hsafri/LLM-Fine-Tune/output\", Operation.ALL)))\n        self.sites.add_sites(local)\n\n        condorpool = Site(\"condorpool\")\n        condorpool.add_profiles(Namespace.CONDOR, key=\"universe\", value=\"vanilla\")\n        condorpool.add_profiles(Namespace.PEGASUS, key=\"style\", value=\"condor\")\n        self.sites.add_sites(condorpool)\n\n    def build_replicas(self):\n        self.replicas.add_replica(\"local\", \"pegasus_data\", \"/home/hsafri/LLM-Fine-Tune/data/data.json\")\n\n    def build_transformations(self):\n        container = Container(\"FineTuneLLM\", Container.SINGULARITY, \"docker://swarmourr/finetune-pegasus:amd64\", image_site=\"docker_hub\")\n        self.transformations.add_containers(container)\n\n        transformation = Transformation(\"FineTuneLLM\", site=\"condorpool\", pfn=\"/home/hsafri/LLM-Fine-Tune/bin/finetune.py\", is_stageable=True)\n        transformation.add_profiles(Namespace.PEGASUS, key=\"cores\", value=\"4\")\n        transformation.add_profiles(Namespace.PEGASUS, key=\"memory\", value=\"10600\")\n        transformation.add_profiles(Namespace.PEGASUS, key=\"gpus\", value=\"1\")\n        self.transformations.add_transformations(transformation)\n\n    def build_jobs(self):\n        job = Job(\"FineTuneLLM\", _id=\"ID0000001\")\n        job.add_args(\"--data_path\", \"pegasus_data\", \"--model_name\", \"tiiuae/falcon-7b\", \"--output_dir\", \"tiiuae/falcon-7b\", \"--num_train_epochs\", \"3\", \"--batch_size\", \"4\", \"--save_steps\", \"5000\", \"--learning_rate\", \"3e-05\", \"--gpu\", \"1\", \"--auth_token\", \"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\")\n        job.add_inputs(self._get_file(\"pegasus_data\"))\n        job.add_outputs(self._get_file(\"falcon-7b.zip\"), stage_out=True, register_replica=True)\n        self.jobs[\"ID0000001\"] = job\n        self.wf.add_jobs(job)\n\n    def _get_file(self, name):\n        if name not in self.files:\n            self.files[name] = File(name)\n        return self.files[name]\n\n    def write(self):\n        self.sites.write()\n        self.replicas.write()\n        self.transformations.write()\n        self.wf.write()\n\nif __name__ == \"__main__\":\n    import os\n    current_dir = os.path.dirname(os.path.abspath(__file__))\n    w = Falcon_7bWorkflow(base_dir=current_dir)\n    w.build_sites()\n    w.build_replicas()\n    w.build_transformations()\n    w.build_jobs()\n    w.write()\n    print(\"Workflow generated and written successfully in: \" + current_dir)\n"
}



    try:
        agent = PegasusAgent()
        context_id = None
        
        # Première analyse
        result = await agent.analyze_event(example_event_1, context_id)
        
        print("=== logs Analysis Result ===")
        print(result.get("logs_needed"))
        print("==========================")
        # Boucle tant que des logs sont nécessaires
        while result.get("logs_needed") == "YES":
            job_id = result.get("job_id")
            if not job_id:
                logger.error("La clé 'job_id' est absente du résultat. Impossible de récupérer les logs.")
                break

            print(f"Récupération des logs pour le job {job_id}...")
            
            # Appel à la fonction simulée pour récupérer les logs
            new_logs = await get_job_logs(job_id)
            
            # Mettre à jour le contexte avec les nouveaux logs
            agent.log_context_manager.update_context(context_id, additional_logs=new_logs)
            
            # Refaire l'analyse avec les nouveaux logs
            result = await agent.analyze_event(example_event_1, context_id)

        # Traitement du résultat final...
        print("=== Analysis Result ===")
        print(json.dumps(result, indent=2))

        # Récupérer le code corrigé
        corrected_code = result.get("execution", {}).get("corrected_generated_code")
        
        if not corrected_code:
            logger.error("Aucun code corrigé à écrire dans le fichier.")
            return

        # Si le code est incomplet, fusionner avec l'original
        if (
            "..." in corrected_code
            or "rest of the code" in corrected_code
            or "rest of the generated code" in corrected_code
            or "set_stdout(...)" in corrected_code
            or "add_args(...)" in corrected_code
        ):
            print("⚠️ Merging with original code to get a complete version.")
            original_code = example_event_1.get("generated_code", "")
            merged_code = await merge_codes_with_llm(agent.llm, original_code, corrected_code)
            result["execution"]["corrected_generated_code"] = merged_code
            corrected_code = merged_code

        # Écrire le code dans un fichier .py
        py_filename = "workflowtest.py"
        with open(py_filename, "w") as pyfile:
            pyfile.write(corrected_code)
        print(f"✅ Python code written to {py_filename}")

        # Écrire les résultats d'analyse
        print("\n=== Analysis Results ===")
        print(json.dumps(result, indent=2))
        with open("pegasus_analysis_result.json", "w") as f:
            json.dump(result, f, indent=2)

        # Afficher le résumé des actions recommandées
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
    asyncio.run(main())