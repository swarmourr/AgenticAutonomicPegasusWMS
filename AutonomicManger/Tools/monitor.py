import subprocess
import json
import time
import os
import logging
from threading import Thread
from enum import Enum
from fullanalyse import * 

class TerminalColor(Enum):
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'  # orange on some systems
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
    RESET = '\033[0m'  # Called to return to standard terminal text color

    def apply(self, text):
        """Apply the color to a string."""
        return f"{self.value}{text}{TerminalColor.RESET.value}"

# Create logs directory if it doesn't exist
if not os.path.exists("logs"):
    os.makedirs("logs")


def setup_logger(workflow_id=None):
    """
    Setup a logger for each workflow and create a dedicated folder for the logs and JSON files.
    If no workflow_id is provided, set up a general logger for the monitoring process.
    """
    # If workflow_id is None, set up the main monitoring logger
    if workflow_id is None:
        logger = logging.getLogger("main_monitor")
        logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler("logs/main_monitor.log")
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    # Create a folder for each workflow under the 'logs' directory
    workflow_log_dir = f"logs/{workflow_id}"
    if not os.path.exists(workflow_log_dir):
        os.makedirs(workflow_log_dir)

    # Set up logger for a specific workflow
    logger = logging.getLogger(workflow_id)
    logger.setLevel(logging.DEBUG)

    # Create file handler to save logs for each workflow
    handler = logging.FileHandler(f"{workflow_log_dir}/{workflow_id}_monitor.log")
    handler.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)

    return logger


class WorkflowRegister:
    def __init__(self):
        # Register: {workflow_id: iwd}
        self.registered_workflows = {}
        self.watchers = {}

    def add_workflow(self, workflow_id, iwd):
        """
        Add a workflow to the register and start a watcher for it.
        """
        if workflow_id not in self.registered_workflows:
            self.registered_workflows[workflow_id] = iwd
            self.start_watcher(workflow_id, iwd)

    def remove_workflow(self, workflow_id):
        """
        Remove a workflow and stop its watcher.
        """
        if workflow_id in self.registered_workflows:
            del self.registered_workflows[workflow_id]

        if workflow_id in self.watchers:
            del self.watchers[workflow_id]

    def start_watcher(self, workflow_id, iwd):
        """
        Start a watcher thread for a specific workflow.
        """
        watcher_thread = Thread(target=self.watch_workflow, args=(workflow_id, iwd), daemon=True)
        self.watchers[workflow_id] = watcher_thread
        watcher_thread.start()

    def watch_workflow(self, workflow_id, iwd):
        """
        Monitor a specific workflow by calling `pegasus-status -j <iwd>`.
        """
        logger = setup_logger(workflow_id)  # Get logger for the specific workflow
        held_retries = 0  # Counter for retries on detecting 'Held' jobs
        max_retries = 3  # Number of retries before deciding to remove the workflow

        while workflow_id in self.registered_workflows:
            try:
                # Run `pegasus-status -j <iwd>`
                result = subprocess.run(
                    ["pegasus-status", "-j", iwd],
                    capture_output=True,
                    text=True,
                    check=True
                )
                # Parse the output (assumes JSON format)
                data = json.loads(result.stdout)

                # Extract general workflow status
                totals = data.get("dags", {})
                percentage_status = totals.get("root", 0.0).get("percent_done", 0.0)
                general_status = totals.get("root", 0.0).get("state","unknown")
                

                # Check for held jobs
                held_jobs = [
                    job for job in data.get("condor_jobs", {}).values()
                    for job in job.get("DAG_CONDOR_JOBS", [])
                    if job.get("JobStatusName", "") == "Held"
                ]

                if held_jobs:
                    held_retries += 1
                    print(TerminalColor.RED.apply(f"Workflow {workflow_id} has jobs in 'Held' state. Retry {held_retries}/{max_retries}."))
                    logger.info(f"Workflow {workflow_id} has jobs in 'Held' state. Retry {held_retries}/{max_retries}.")

                    # Build JSON describing the workflow and held jobs
                    workflow_info = {
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

                    # Write JSON to a file in the workflow's log folder
                    json_file = f"logs/{workflow_id}/{workflow_id}_held_jobs.json"
                    with open(json_file, "w") as f:
                        json.dump(workflow_info, f, indent=2)
                    print(TerminalColor.GREEN.apply(f"Saved held jobs to {json_file}"))
                    logger.info(f"Saved held jobs to {json_file}")

                    # If retries exceed max_retries, stop the workflow
                    if held_retries >= max_retries:
                        print(TerminalColor.RED.apply("Maximum retries reached for workflow {workflow_id}. Stopping workflow."))
                        logger.warning(f"Maximum retries reached for workflow {workflow_id}. Stopping workflow.")
                        subprocess.run(["pegasus-remove", iwd], check=True)
                        logger.info(f"Workflow {workflow_id} stopped.")
                        self.remove_workflow(workflow_id)
                        workflow_manager = PegasusWorkflowManager(
                            workflow_dir=iwd,#"//home//hsafri//LLM-Fine-Tune//hsafri//pegasus//falcon-7b//run0001",
                            #workflow_dir="//home//hsafri//LLM-Fine-Tune//generated_workflows//hsafri/pegasus//falcon-7b//run0013",
                            api_url="https://api.together.xyz/v1/chat/completions",
                            api_key="9330fcf33a0d19c088c90a6799e1208f30c405627bc12cf27c4d4eaba36dcb96"
                        )
                        workflow_manager.process_workflow()
                        break

                else:
                    # Reset retries if no held jobs are found
                    held_retries = 0

                # Stop monitoring if workflow is completed
                if general_status == "Success":
                    logger.info(f"Workflow {workflow_id} is completed. Stopping watcher.")
                    self.remove_workflow(workflow_id)
                    break
                    
                elif general_status == "Failure":
                    logger.info(f"Workflow {workflow_id} is Failed. Stopping watcher.")
                    self.remove_workflow(workflow_id)
                    break
                    
                    

            except subprocess.CalledProcessError as e:
                logger.error(f"Error running pegasus-status or pegasus-remove for workflow {workflow_id}: {e}")
                break

            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON for workflow {workflow_id}: {e}")
                break

            time.sleep(10)  # Check every 10 seconds

    def get_all_workflows(self):
        """
        Get all registered workflows as a list of (workflow_id, iwd).
        """
        return list(self.registered_workflows.items())


def get_workflow_details():
    """
    Parse workflows from `pegasus-status -j`.
    """
    try:
        result = subprocess.run(
            ["pegasus-status", "-j"],
            capture_output=True,
            text=True,
            check=True
        )
        data = json.loads(result.stdout)
        workflows = []

        # Parse workflow IDs and directories
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
    """
    Periodically retrieve workflows and start watchers for them.
    """
    logger = setup_logger()  # Set up main logger
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


# Start monitoring
monitor_workflows()
    