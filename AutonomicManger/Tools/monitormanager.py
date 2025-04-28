import subprocess
import json
import time
from fullanalyse import PegasusWorkflowManager


class WorkflowManager:
    def __init__(self):
        self.current_workflows = {}  # Dictionary to store current workflows

    def get_workflow_details(self):
        """
        Use pegasus-status -j to retrieve detailed information about workflows and their jobs.
        :return: List of workflows with detailed information, including general status.
        """
        try:
            # Run pegasus-status with JSON output
            result = subprocess.run(
                ["pegasus-status", "-j"],
                capture_output=True,
                text=True,
                check=True
            )
            # Parse the JSON output
            data = json.loads(result.stdout)
            return self.parse_workflow_details(data)
        except subprocess.CalledProcessError as e:
            print(f"Error running pegasus-status: {e}")
            return []
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON output: {e}")
            return []

    def parse_workflow_details(self, data):
        """
        Parse the JSON data from pegasus-status to extract detailed workflow information.
        :param data: JSON data from pegasus-status -j.
        :return: List of workflows with additional details, including general status and hold reasons.
        """
        workflows = []

        # Extract general workflow information
        totals = data.get("totals", {})
        general_status = self.determine_general_status(totals)

        aggregate_info = {
            "total_jobs": totals.get("total", 0),
            "completed_jobs": totals.get("succeeded", 0),
            "failed_jobs": totals.get("failed", 0),
            "percent_done": totals.get("percent_done", 0.0),
            "general_status": general_status,  # Include overall status
        }

        # Extract details from "condor_jobs" and "dags"
        for wf_id, workflow in data.get("condor_jobs", {}).items():
            workflow_name = workflow.get("DAG_NAME", "Unknown")
            workflow_jobs = []
            directory = None  # Assume all jobs share the same directory for this workflow

            for job in workflow.get("DAG_CONDOR_JOBS", []):
                job_details = {
                    "job_id": job.get("ClusterId"),
                    "job_name": job.get("pegasus_wf_dag_job_id", "Unknown"),
                    "status": job.get("JobStatusName"),
                    "cmd": job.get("Cmd"),
                    "condor_platform": job.get("CondorPlatform"),
                    "condor_version": job.get("CondorVersion"),
                    "job_priority": job.get("JobPrio"),
                    "site": job.get("pegasus_site"),
                }

                # Set the directory (it should be the same for all jobs)
                if not directory:
                    directory = job.get("Iwd")

                # If the job is held, add the HoldReason
                if job.get("JobStatusName") == "Held":
                    job_details["hold_reason"] = job.get("HoldReason", "No reason provided")

                workflow_jobs.append(job_details)

            workflows.append({
                "workflow_id": wf_id,
                "workflow_name": workflow_name,
                "directory": directory,
                "jobs": workflow_jobs,
                "aggregate_info": aggregate_info,
            })
        return workflows

    def map_job_status_by_workflow(self, workflows):
        """
        Organize job statuses by workflow ID with a single directory entry per workflow.
        :param workflows: List of workflows with jobs and aggregate information.
        :return: A dictionary with workflow IDs as keys and directories as values.
        """
        workflow_status_mapping = {}

        for workflow in workflows:
            workflow_id = workflow["workflow_id"]
            workflow_name = workflow["workflow_name"]
            directory = workflow["directory"]
            job_statuses = []

            for job in workflow.get("jobs", []):
                job_name = job.get("job_name", "Unknown")
                job_status = job.get("status", "Unknown")
                if job_status == "Held":
                    workflow_manager = PegasusWorkflowManager(
                    workflow_dir=workflow["directory"],
                    api_url="https://api.together.xyz/v1/chat/completions",
                    api_key="9330fcf33a0d19c088c90a6799e1208f30c405627bc12cf27c4d4eaba36dcb96"
                    )
                    workflow_manager.process_workflow()
                    description = job.get("hold_reason", "No reason provided")
                else:
                    description = ""

                job_statuses.append({
                    "job_name": job_name,
                    "job_status": job_status,
                    "description": description
                })

            # Add workflow details to the mapping
            workflow_status_mapping[workflow_id] = {
                "workflow_name": workflow_name,
                "directory": directory,
                "job_statuses": job_statuses
            }

        return workflow_status_mapping

    def determine_general_status(self, totals):
        """
        Determine the overall status of the workflow based on job totals.
        :param totals: A dictionary containing job counts and percent_done.
        :return: A string representing the general workflow status.
        """
        if totals.get("failed", 0) > 0:
            return "Failed"
        elif totals.get("percent_done", 0.0) == 100.0:
            return "Completed"
        elif totals.get("total", 0) > 0:
            return "In Progress"
        else:
            return "Not Started"

    def update_workflows(self):
        """
        Update the current workflows by adding new workflows and removing finished or removed workflows.
        """
        new_workflows = self.get_workflow_details()

        # Add new workflows or update existing ones
        for workflow in new_workflows:
            workflow_id = workflow["workflow_id"]
            self.current_workflows[workflow_id] = workflow

        # Remove workflows that are no longer detected or completed
        workflow_ids = set(wf["workflow_id"] for wf in new_workflows)
        for workflow_id in list(self.current_workflows.keys()):
            if workflow_id not in workflow_ids:
                print(f"Removing workflow {workflow_id} as it is no longer active.")
                del self.current_workflows[workflow_id]

    def monitor_workflows(self, interval=60):
        """
        Periodically monitor and print detailed information about workflows.
        :param interval: Time in seconds between checks.
        """
        while True:
            print("Checking workflows and their details...")
            self.update_workflows()

            if self.current_workflows:
                print(f"Detected {len(self.current_workflows)} workflows:")
                organized_status = self.map_job_status_by_workflow(self.current_workflows.values())
                print(json.dumps(organized_status, indent=2))
            else:
                print("No workflows detected.")

            time.sleep(interval)


# Start monitoring with WorkflowManager
workflow_manager = WorkflowManager()
workflow_manager.monitor_workflows()
