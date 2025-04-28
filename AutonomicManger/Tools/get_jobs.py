import sqlite3
import os

class WorkflowMonitor:
    def __init__(self, wf_uuid):
        """
        Initialize the workflow monitor.
        Args:
            wf_uuid (str): UUID of the workflow to monitor.
        """
        self.wf_uuid = wf_uuid  # Set wf_uuid first
        self.db_path = self._find_pegasus_database()
        self.tasks_db_path = self._get_tasks_db_path()

    def _find_pegasus_database(self):
        """
        Locate the Pegasus SQLite database in the user's home directory.
        Returns:
            str: Full path to the workflow database.
        Raises:
            FileNotFoundError: If the database is not found.
        """
        home_dir = os.path.expanduser("~")
        pegasus_db_path = os.path.join(home_dir, ".pegasus", "workflow.db")
        if not os.path.exists(pegasus_db_path):
            raise FileNotFoundError(f"Pegasus database not found at {pegasus_db_path}")
        return pegasus_db_path

    def _get_tasks_db_path(self):
        """
        Extract the `db_url` (tasks database path) from the workflow database.
        Returns:
            str: Path to the tasks database.
        Raises:
            RuntimeError: If the `db_url` cannot be found.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                query = "SELECT db_url FROM master_workflow WHERE wf_uuid = ?;"
                cursor.execute(query, (self.wf_uuid,))
                row = cursor.fetchone()
                if row and row[0]:
                    tasks_db_path = os.path.expanduser(row[0].replace("sqlite:///", ""))  # Convert URL to path
                    if os.path.exists(tasks_db_path):
                        return tasks_db_path
                    else:
                        raise FileNotFoundError(f"Tasks database not found at {tasks_db_path}")
                else:
                    raise RuntimeError("Tasks database path (`db_url`) not found in the workflow database.")
        except sqlite3.Error as e:
            raise RuntimeError(f"Error accessing the workflow database: {e}")

    def get_workflow_details(self):
        """
        Fetch details of the specified workflow from master_workflow table.
        Returns:
            dict: Workflow details if found, otherwise None.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                query = "SELECT * FROM master_workflow WHERE wf_uuid = ?;"
                cursor.execute(query, (self.wf_uuid,))
                row = cursor.fetchone()
                if row:
                    columns = [desc[0] for desc in cursor.description]
                    return dict(zip(columns, row))
                else:
                    print("Workflow not found.")
                    return None
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return None

    def get_tasks_from_tasks_db(self):
        """
        Fetch task information from the tasks database based on its schema.
        Returns:
            list: List of tasks with their states and related information.
        """
        try:
            with sqlite3.connect(self.tasks_db_path) as conn:
                cursor = conn.cursor()
                
                # Inspect the schema of the task table
                cursor.execute("PRAGMA table_info(task);")
                schema_info = cursor.fetchall()
                columns = [col[1] for col in schema_info]  # Extract column names
                
                print("\n--- Task Table Schema ---")
                for col in columns:
                    print(f"Column: {col}")
                
                # Attempt to fetch tasks filtered by wf_id
                cursor.execute("""
                    SELECT * FROM task WHERE wf_id = (
                        SELECT wf_id FROM master_workflow WHERE wf_uuid = ?
                    );
                """, (self.wf_uuid,))
                rows = cursor.fetchall()

                if not rows:
                    print("No tasks found for the given workflow ID. Fetching all tasks instead.")
                    cursor.execute("SELECT * FROM task;")
                    rows = cursor.fetchall()

                # Format rows as dictionaries
                return [dict(zip(columns, row)) for row in rows]
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return []

    def get_host_info(self):
        """
        Fetch host information from the host table.
        Returns:
            list: List of hosts and their related information.
        """
        try:
            with sqlite3.connect(self.tasks_db_path) as conn:
                cursor = conn.cursor()
                
                # Inspect the schema of the host table
                cursor.execute("PRAGMA table_info(host);")
                schema_info = cursor.fetchall()
                host_columns = [col[1] for col in schema_info]  # Extract column names
                
                print("\n--- Host Table Schema ---")
                for col in host_columns:
                    print(f"Column: {col}")
                
                # Fetch host information related to the workflow's tasks
                cursor.execute("""
                    SELECT * FROM host WHERE wf_id = (
                        SELECT wf_id FROM master_workflow WHERE wf_uuid = ?
                    );
                """, (self.wf_uuid,))
                rows = cursor.fetchall()

                if not rows:
                    print("No hosts found for the given workflow ID. Fetching all hosts instead.")
                    cursor.execute("SELECT * FROM host;")
                    rows = cursor.fetchall()

                # Format rows as dictionaries
                return [dict(zip(host_columns, row)) for row in rows]
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return []

    def display_workflow_and_tasks_and_hosts(self):
        """
        Display workflow details, task information, and host information.
        """
        # Get workflow details
        workflow_details = self.get_workflow_details()
        if workflow_details:
            print("\n--- Workflow Details ---")
            for key, value in workflow_details.items():
                print(f"{key}: {value}")

            # Get task information from the tasks database
            tasks = self.get_tasks_from_tasks_db()
            print("\n--- Task Information ---")
            for task in tasks:
                print("Task Details:")
                for key, value in task.items():
                    print(f"  {key}: {value}")

            # Get host information from the host table
            hosts = self.get_host_info()
            print("\n--- Host Information ---")
            for host in hosts:
                print("Host Details:")
                for key, value in host.items():
                    print(f"  {key}: {value}")
        else:
            print("No workflow found with the specified UUID.")

# Usage example
if __name__ == "__main__":
    WORKFLOW_UUID = "2282fa8f-4055-4148-80da-3c9d7923a294"  # Example UUID

    try:
        monitor = WorkflowMonitor(WORKFLOW_UUID)
        monitor.display_workflow_and_tasks_and_hosts()
    except (FileNotFoundError, RuntimeError) as e:
        print(e)
