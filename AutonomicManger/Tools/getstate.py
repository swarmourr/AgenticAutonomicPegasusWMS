import os
import sqlite3
import time

STAMPED_DB = "/home/hsafri/ACCESS-Pegasus-Examples/Artificial-Intelligence/MaskDetection/hsafri/pegasus/mask_detection_workflow/run0001/mask_detection_workflow-0.stampede.db"

def validate_db_columns(db_path, table_name, required_columns):
    """
    Validates if the required columns exist in the specified table.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name});")
            existing_columns = {row[1] for row in cursor.fetchall()}
            missing_columns = set(required_columns) - existing_columns
            if missing_columns:
                raise ValueError(f"Missing columns in {table_name}: {missing_columns}")
    except Exception as e:
        raise RuntimeError(f"Error validating database schema: {e}")

def get_failed_jobs(db_path):
    """
    Retrieves details of failed jobs from the Stampede database.
    """
    failed_jobs = []
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Query for failed jobs
        query = """
        SELECT job_id, exec_job_id AS job_name, state, dag_job_id, exec_site
        FROM job_instance
        WHERE state = 'FAILURE';
        """
        cursor.execute(query)
        jobs = cursor.fetchall()

        # Process each failed job
        for job in jobs:
            job_id, job_name, state, dag_job_id, exec_site = job
            failed_jobs.append({
                "job_id": job_id,
                "job_name": job_name,
                "state": state,
                "dag_job_id": dag_job_id,
                "exec_site": exec_site
            })
        return failed_jobs
    except sqlite3.Error as e:
        print(f"Error querying failed jobs: {e}")
    finally:
        if conn:
            conn.close()

    return failed_jobs

def find_error_file(job_name, workflow_dir):
    """
    Attempts to locate the stderr file for the given job.
    """
    possible_files = [f"{job_name}.err", f"{job_name}.stderr"]
    for file_name in possible_files:
        stderr_path = os.path.join(workflow_dir, file_name)
        if os.path.exists(stderr_path):
            return stderr_path
    return None

def get_job_error_message(job_name, workflow_dir):
    """
    Retrieves error details from the stderr file of a failed job.
    """
    stderr_file = find_error_file(job_name, workflow_dir)
    if stderr_file:
        with open(stderr_file, "r") as file:
            return file.read().strip()
    else:
        return "No stderr file found."

def monitor_workflow(db_path, workflow_dir, interval=60):
    """
    Periodically checks the workflow status and reports errors if failed.
    """
    # Validate schema before starting
    required_columns = ["job_id", "exec_job_id", "state", "dag_job_id", "exec_site"]
    validate_db_columns(db_path, "job_instance", required_columns)

    while True:
        print(f"[{time.ctime()}] Checking workflow status...")

        # Check failed jobs
        failed_jobs = get_failed_jobs(db_path)
        if failed_jobs:
            print("\n--- Workflow has failed jobs. Retrieving error details ---")
            for job in failed_jobs:
                print(f"Job Name: {job['job_name']}, Execution Site: {job['exec_site']}")
                error_message = get_job_error_message(job['job_name'], workflow_dir)
                print(f"Error Message:\n{error_message}")
            break

        print("No failed jobs detected. Workflow is still running...")
        time.sleep(interval)

if __name__ == "__main__":
    workflow_dir = "/home/hsafri/ACCESS-Pegasus-Examples/Artificial-Intelligence/MaskDetection/hsafri/pegasus/mask_detection_workflow/run0001/"  # Path to workflow logs and .err files
    if not os.path.exists(STAMPED_DB):
        print(f"Stampede database not found: {STAMPED_DB}")
    else:
        try:
            monitor_workflow(STAMPED_DB, workflow_dir)
        except RuntimeError as e:
            print(e)
