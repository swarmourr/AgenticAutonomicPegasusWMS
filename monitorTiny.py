from tinydb import TinyDB, Query
import subprocess
import json
import time
import os
import logging
from threading import Thread
from enum import Enum
import yaml
import os
from pathlib import Path
from textwrap import indent


# Initialize TinyDB
db = TinyDB("workflows.json")
workflows_table = db.table("workflows")
held_jobs_table = db.table("held_jobs")

# Create logs directory if it doesn't exist
if not os.path.exists("logs"):
    os.makedirs("logs")

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

def setup_logger(workflow_id=None):
    """
    Setup a logger for each workflow and create a dedicated folder for the logs and JSON files.
    If no workflow_id is provided, set up a general logger for the monitoring process.
    """
    if workflow_id is None:
        logger = logging.getLogger("main_monitor")
        logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler("logs/main_monitor.log")
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    workflow_log_dir = f"logs/{workflow_id}"
    if not os.path.exists(workflow_log_dir):
        os.makedirs(workflow_log_dir)

    logger = logging.getLogger(workflow_id)
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(f"{workflow_log_dir}/{workflow_id}_monitor.log")
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger



class PegasusWorkflowGenerator:
    """
    A generator that converts a YAML workflow configuration into a clean, 
    modular, and extensible Pegasus OOP-style Python script.
    The workflow is generated inside a directory with the same name.
    """

    def __init__(self, yaml_path: str):
        """
        Initialize the generator with a YAML path.

        :param yaml_path: Path to the YAML configuration file
        """
        self.yaml_path = yaml_path
        self.data = {}
        self.workflow_name = ""
        self.generated_code = ""
        self.workflow_dir = ""

    def load_yaml(self):
        """
        Loads the YAML configuration into the generator.

        This should be called before generating the script.
        """
        with open(self.yaml_path, "r") as f:
            self.data = yaml.safe_load(f)
            self.workflow_name = self.data.get("name", "GeneratedWorkflow")
            self.workflow_dir = self.workflow_name

    def generate(self, output_filename="workflow.py"):
        """
        Generate the Python script from the YAML workflow configuration
        inside a directory with the same name as the workflow.

        :param output_filename: Name of the Python script file (default: workflow.py)
        """
        self.load_yaml()
        
        # Create workflow directory if it doesn't exist
        os.makedirs(self.workflow_dir, exist_ok=True)
        
        # Set output path to be inside the workflow directory
        output_path = os.path.join(self.workflow_dir, output_filename)
        
        code = self._generate_header()
        code += self._generate_site_catalog()
        code += self._generate_replica_catalog()
        code += self._generate_transformation_catalog()
        code += self._generate_job_definitions()
        code += self._generate_main()
        
        Path(output_path).write_text(code)
        print(f"Generated workflow script written to: {output_path}")
        print(f"Workflow directory created: {self.workflow_dir}")
        return code

    def _generate_header(self):
        """
        Generate the header of the Python script with necessary imports
        and the Workflow class.

        :return: The header code
        """
        return f"""from Pegasus.api import *
import os

class {self.workflow_name.replace('-', '_').capitalize()}Workflow:
    \"\"\"
    A Pegasus workflow for {self.workflow_name}.
    \"\"\"
    def __init__(self, base_dir="."):
        \"\"\"
        Initialize the workflow, sites, replicas, transformations, and job containers.
        
        :param base_dir: Base directory for the workflow (default: current directory)
        \"\"\"
        self.base_dir = base_dir
        
        # Change to the workflow directory for all operations
        # This ensures catalog files are written to the correct location
        os.chdir(self.base_dir)
        
        self.wf = Workflow(name="{self.workflow_name}")
        self.sites = SiteCatalog()
        self.replicas = ReplicaCatalog()
        self.transformations = TransformationCatalog()
        self.files = {{}}
        self.jobs = {{}}
"""

    def _generate_site_catalog(self):
        """
        Generate the site catalog section for the workflow, defining sites and directories.

        :return: The code for the site catalog section
        """
        output = "\n    def build_sites(self):\n"
        for site in self.data.get("siteCatalog", {}).get("sites", []):
            site_block = f'        {site["name"]} = Site("{site["name"]}")\n'

            for d in site.get("directories", []):
                dir_type = d["type"]
                path = d["path"]
                fs = d.get("fileServers", [])[0]
                url = fs["url"]
                op = fs["operation"].upper()

                # Map directory type strings to Directory enum values
                dir_type_map = {
                    "sharedScratch": "SHARED_SCRATCH",
                    "localStorage": "LOCAL_STORAGE",
                    "localScratch": "LOCAL_SCRATCH",
                    "sharedStorage": "SHARED_STORAGE"
                }
                
                if dir_type not in dir_type_map:
                    raise ValueError(f"Unknown directory type: {dir_type}")
                
                # Create directory object with proper enum reference
                dir_obj = (
                    f'Directory(Directory.{dir_type_map[dir_type]}, "{path}")'
                    f'.add_file_servers(FileServer("{url}", Operation.{op}))'
                )
                site_block += f"        {site['name']}.add_directories({dir_obj})\n"

            for ns, profiles in site.get("profiles", {}).items():
                for k, v in profiles.items():
                    site_block += f'        {site["name"]}.add_profiles(Namespace.{ns.upper()}, key="{k}", value="{v}")\n'

            site_block += f'        self.sites.add_sites({site["name"]})\n\n'
            output += site_block
        return output

    def _generate_replica_catalog(self):
        """
        Generate the replica catalog section for the workflow, defining replicas.

        :return: The code for the replica catalog section
        """
        output = "\n    def build_replicas(self):\n"
        for r in self.data.get("replicaCatalog", {}).get("replicas", []):
            site = r["pfns"][0]["site"]
            pfn = r["pfns"][0]["pfn"]
            lfn = r["lfn"]
            output += f'        self.replicas.add_replica("{site}", "{lfn}", "{pfn}")\n'
        return output

    def _generate_transformation_catalog(self):
        """
        Generate the transformation catalog section for the workflow, defining transformations.

        :return: The code for the transformation catalog section
        """
        output = "\n    def build_transformations(self):\n"
        containers = self.data.get("transformationCatalog", {}).get("containers", [])
        for c in containers:
            output += f'        container = Container("{c["name"]}", Container.{c["type"].upper()}, "{c["image"]}", image_site="{c["image.site"]}")\n'
            output += "        self.transformations.add_containers(container)\n"

        for t in self.data.get("transformationCatalog", {}).get("transformations", []):
            pfn = t["sites"][0]["pfn"]
            site = t["sites"][0]["name"]
            container = t.get("container", None)
            output += f'\n        transformation = Transformation("{t["name"]}", site="{site}", pfn="{pfn}", is_stageable=True'
            if container:
                output += f", container=container"
            output += ")\n"
            for ns, profiles in t.get("profiles", {}).items():
                for k, v in profiles.items():
                    output += f'        transformation.add_profiles(Namespace.{ns.upper()}, key="{k}", value="{v}")\n'
            output += "        self.transformations.add_transformations(transformation)\n"
        return output

    def _generate_job_definitions(self):
        """
        Generate the job definitions section for the workflow, defining jobs with arguments, inputs, and outputs.

        :return: The code for the job definitions section
        """
        output = "\n    def build_jobs(self):\n"
        for job in self.data.get("jobs", []):
            job_id = job["id"]
            name = job["name"]
            args = job.get("arguments", [])
            uses = job.get("uses", [])

            output += f'        job = Job("{name}", _id="{job_id}")\n'
            if args:
                # Split long single string into individual args
                arg_list = " ".join(args).split()
                args_str = ", ".join(f'"{a}"' for a in arg_list)
                output += f"        job.add_args({args_str})\n"

            for use in uses:
                lfn = use["lfn"]
                use_type = use["type"]
                if use_type == "input":
                    output += f'        job.add_inputs(self._get_file("{lfn}"))\n'
                elif use_type == "output":
                    stage_out = use.get("stageOut", False)
                    register = use.get("registerReplica", False)
                    opts = []
                    if stage_out:
                        opts.append("stage_out=True")
                    if register:
                        opts.append("register_replica=True")
                    opt_str = ", " + ", ".join(opts) if opts else ""
                    output += f'        job.add_outputs(self._get_file("{lfn}"){opt_str})\n'

            output += f'        self.jobs["{job_id}"] = job\n'
            output += f"        self.wf.add_jobs(job)\n\n"
        return output + "\n    def _get_file(self, name):\n        if name not in self.files:\n            self.files[name] = File(name)\n        return self.files[name]\n"

    def _generate_main(self):
        """
        Generate the main section to build and write the workflow.

        :return: The code for the main section
        """
        return f"""
    def write(self):
        \"\"\"
        Write the site, replica, transformation, and workflow to their respective catalogs and files.
        \"\"\"
        # Write all catalog files
        self.sites.write()
        self.replicas.write()
        self.transformations.write()
        self.wf.write()

if __name__ == "__main__":
    import os
    import sys
    
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Initialize workflow with current directory
    w = {self.workflow_name.replace('-', '_').capitalize()}Workflow(base_dir=current_dir)
    w.build_sites()
    w.build_replicas()
    w.build_transformations()
    w.build_jobs()
    w.write()
    print("Workflow generated and written successfully in: " + current_dir)
"""

class WorkflowRegister:
    def __init__(self):
        self.registered_workflows = {}
        self.watchers = {}

    def add_workflow(self, workflow_id, iwd):
        if workflow_id not in self.registered_workflows:
            self.registered_workflows[workflow_id] = iwd
            self.start_watcher(workflow_id, iwd)

    def remove_workflow(self, workflow_id):
        if workflow_id in self.registered_workflows:
            del self.registered_workflows[workflow_id]
        if workflow_id in self.watchers:
            del self.watchers[workflow_id]

    def find_yaml_file(self,workflow_dir):
        """Find the workflow YAML file in the specified directory.

        This method searches for a YAML file in the given `workflow_dir` whose name matches
        the parent directory's name (excluding the trailing slash) with either a `.yml` or `.yaml` extension.
        It excludes files named `braindump.yml` from the search.

        Args:
            workflow_dir (str): The directory path where the workflow YAML file is expected to be located.

        Returns:
            str: The full path to the matching workflow YAML file.

        Raises:
            FileNotFoundError: If no matching YAML file is found in the specified directory."""
        
        wf_name = workflow_dir.split("/")[-2]
        for file_name in os.listdir(workflow_dir):
            if (file_name.endswith(wf_name + '.yml') or file_name.endswith(wf_name + '.yaml')) and file_name != 'braindump.yml':      
                   return os.path.join(workflow_dir, file_name)
        raise FileNotFoundError(f"No YAML file found in directory: {workflow_dir}")
    
    def start_watcher(self, workflow_id, iwd):
        watcher_thread = Thread(target=self.watch_workflow, args=(workflow_id, iwd), daemon=True)
        self.watchers[workflow_id] = watcher_thread
        watcher_thread.start()

    def run_pegasus_analyzer(self,workflow_dir):
        """
        Run the Pegasus Analyzer to generate workflow logs.
        :return: Combined standard output and error output from the analyzer.
        """
        try:
            result = subprocess.run(
                ['pegasus-analyzer', workflow_dir],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            return result.stdout + "\n" + result.stderr
        except subprocess.CalledProcessError as e:
            print(f"Error running pegasus-analyzer: {e.stderr}")
            return e.stdout + "\n" + e.stderr
        
    def watch_workflow(self, workflow_id, iwd):
        logger = setup_logger(workflow_id)
        retries = 0
        max_retries = 3

        while workflow_id in self.registered_workflows:
            try:
                result = subprocess.run(
                    ["pegasus-status", "-j", iwd],
                    capture_output=True,
                    text=True,
                    check=True
                )
                data = json.loads(result.stdout)

                # Extract workflow details
                totals = data.get("dags", {})
                percent_done = totals.get("root", {}).get("percent_done", 0.0)
                state = totals.get("root", {}).get("state", "unknown")

                # Add/update workflow in TinyDB
                workflows_table.upsert({
                    "workflow_id": workflow_id,
                    "iwd": iwd,
                    "state": state,
                    "percent_done": percent_done,
                    "last_checked": time.strftime("%Y-%m-%d %H:%M:%S")
                }, Query().workflow_id == workflow_id)

                # Detect held jobs
                held_jobs = [
                    job for job in data.get("condor_jobs", {}).values()
                    for job in job.get("DAG_CONDOR_JOBS", [])
                    if job.get("JobStatusName", "") == "Held"
                ]

                if held_jobs:
                    retries += 1
                    print(TerminalColor.RED.apply(f"Workflow {workflow_id}: Held jobs detected (Retry {retries}/{max_retries})."))
                    logger.warning(f"Workflow {workflow_id} has jobs in 'Held' state. Retry {retries}/{max_retries}.")

                    # Process each held job
                    for job in held_jobs:
                        held_job_data = {
                            "workflow_id": workflow_id,
                            "job_id": job.get("pegasus_wf_dag_job_id", "Unknown"),
                            "status": job.get("JobStatusName", "Unknown"),
                            "hold_reason": job.get("HoldReason", "No reason provided"),
                            "site": job.get("pegasus_site", "Unknown"),
                            "cmd": job.get("Cmd"),
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "condor_platform": job.get("CondorPlatform"),
                            "condor_version": job.get("CondorVersion"),
                            "job_priority": job.get("JobPrio")
                        }
                        held_jobs_table.insert(held_job_data)

                    # Save detailed JSON for analysis
                    workflow_info = {
                         "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "workflow_id": workflow_id,
                        "directory": iwd,
                        "held_jobs": [
                            {
                                "job_id": job.get("pegasus_wf_dag_job_id", "Unknown"),
                                "status": job.get("JobStatusName", "Unknown"),
                                "hold_reason": job.get("HoldReason", "No reason provided use the analzer"),
                                "site": job.get("pegasus_site", "Unknown"),
                                "cmd": job.get("Cmd"),
                                "condor_platform": job.get("CondorPlatform"),
                                "condor_version": job.get("CondorVersion"),
                                "job_priority": job.get("JobPrio"),
                            }
                            for job in held_jobs
                        ]
                    }

                    

                    # Stop workflow if max retries exceeded
                    if retries >= max_retries:
                        print(TerminalColor.RED.apply(f"Maximum retries reached for workflow {workflow_id}. Stopping workflow."))
                        logger.warning(f"Maximum retries reached for workflow {workflow_id}. Stopping workflow.")
                        subprocess.run(["pegasus-remove", iwd], check=True)
                        logger.info(f"Workflow {workflow_id} stopped.")
                        workflows_table.update({"state": "removed"}, Query().workflow_id == workflow_id)
                        self.remove_workflow(workflow_id)
                        workflow_info["analyzer_output"] = self.run_pegasus_analyzer(iwd)
                        original_yaml_path = self.find_yaml_file(iwd)
                        workflow_info["original_yaml"] = original_yaml_path
                        gen = PegasusWorkflowGenerator(original_yaml_path)
                        python_code= gen.generate()
                        print("Workflow generation process completed successfully.")
                        workflow_info["generated_code"] = python_code
                        # Write JSON file
                        json_file = f"logs/{workflow_id}/{workflow_id}_held_jobs.json"
                        with open(json_file, "w") as f:
                            json.dump(workflow_info, f, indent=2)
                        print(TerminalColor.GREEN.apply(f"Saved held jobs to {json_file}"))
                        logger.info(f"Saved held jobs to {json_file}")
                        print(workflow_info)
                        break
                else:
                    retries = 0
                # Stop monitoring on completion
                if state in ["Success", "Failure"]:
                    print(TerminalColor.GREEN.apply(f"Workflow {workflow_id} completed with state {state}."))
                    logger.info(f"Workflow {workflow_id} completed with state {state}.")
                    workflows_table.update({"state": state}, Query().workflow_id == workflow_id)
                    self.remove_workflow(workflow_id)
                    break

            except subprocess.CalledProcessError as e:
                logger.error(f"Error running pegasus-status for workflow {workflow_id}: {e}")
                break
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON for workflow {workflow_id}: {e}")
                break

            time.sleep(10)

    def get_all_workflows(self):
        return list(self.registered_workflows.items())

def get_workflow_details():
    try:
        result = subprocess.run(
            ["pegasus-status", "-j"],
            capture_output=True,
            text=True,
            check=True
        )
        data = json.loads(result.stdout)
        workflows = []
        for wf_id, workflow_data in data.get("condor_jobs", {}).items():
            iwd = workflow_data.get("DAG_CONDOR_JOBS", [{}])[0].get("Iwd", "Unknown")
            workflows.append((wf_id, iwd))
        return workflows
    except subprocess.CalledProcessError as e:
        print(f"Error running pegasus-status: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON output: {e}")
        return []

def monitor_workflows(interval=60):
    logger = setup_logger()
    register = WorkflowRegister()

    while True:
        logger.info("Checking workflows...")
        workflows = get_workflow_details()

        for wf_id, iwd in workflows:
            register.add_workflow(wf_id, iwd)

        monitored_workflows = register.get_all_workflows()
        print(f"Currently monitoring {len(monitored_workflows)} workflows:")
        logger.info(f"Currently monitoring {len(monitored_workflows)} workflows:")
        for wf_id, iwd in monitored_workflows:
            print(f"- Workflow {wf_id} in directory {iwd}")
            logger.info(f"- Workflow {wf_id} in directory {iwd}")

        time.sleep(interval)

if __name__ == "__main__":
    monitor_workflows()