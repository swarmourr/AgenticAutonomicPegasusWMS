import subprocess
import json
import time

def get_workflow_details():
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
        return parse_workflow_details(data)
    except subprocess.CalledProcessError as e:
        print(f"Error running pegasus-status: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON output: {e}")
        return []

def parse_workflow_details(data):
    """
    Parse the JSON data from pegasus-status to extract detailed workflow information.
    :param data: JSON data from pegasus-status -j.
    :return: List of workflows with additional details, including general status and hold reasons.
    """
    workflows = []

    # Extract general workflow information
    totals = data.get("totals", {})
    general_status = determine_general_status(totals)

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

        for job in workflow.get("DAG_CONDOR_JOBS", []):
            job_details = {
                "job_id": job.get("ClusterId"),
                "job_name": job.get("pegasus_wf_dag_job_id", "Unknown"),
                "status": job.get("JobStatusName"),
                "directory": job.get("Iwd"),
                "cmd": job.get("Cmd"),
                "condor_platform": job.get("CondorPlatform"),
                "condor_version": job.get("CondorVersion"),
                "job_priority": job.get("JobPrio"),
                "site": job.get("pegasus_site"),
            }

            # If the job is held, add the HoldReason
            if job.get("JobStatusName") == "Held":
                
                job_details["hold_reason"] = job.get("HoldReason", "No reason provided")

            workflow_jobs.append(job_details)

        workflows.append({
            "workflow_id": wf_id,
            "workflow_name": workflow_name,
            "jobs": workflow_jobs,
            "aggregate_info": aggregate_info,
        })
    return workflows

def map_job_status_by_name(data):
    """
    Map job statuses by job names from the structured data.
    :param data: List of workflows with jobs and aggregate information.
    :return: A dictionary mapping job names to their statuses.
    """
    job_status_mapping = {}

    for workflow in data:
        for job in workflow.get("jobs", []):
            job_name = job.get("job_name", "Unknown")
            job_status = job.get("status", "Unknown")
            if job_status=="Held":
                print(job)
                description=job.get("hold_reason", "Unknown")
            else: 
                description=""
            job_status={"job_status":job_status,"Description":description}
            job_status_mapping[job_name] = job_status

    return job_status_mapping

    
def determine_general_status(totals):
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

def monitor_workflows(interval=60):
    """
    Periodically monitor and print detailed information about workflows.
    :param interval: Time in seconds between checks.
    """
    while True:
        print("Checking workflows and their details...")
        workflows = get_workflow_details()

        if workflows:
            print(f"Detected {len(workflows)} workflows:")
            print(json.dumps(workflows, indent=2))
            print(map_job_status_by_name(workflows))
            #print(workflows)
        else:
            print("No workflows detected.")

        time.sleep(interval)

# Start monitoring
monitor_workflows()
