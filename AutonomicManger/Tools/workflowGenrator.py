import yaml
import os
from pathlib import Path
from textwrap import indent
from Pegasus.api import Directory, FileServer, Operation, Site, Namespace, Transformation, Container, Job, Workflow, ReplicaCatalog, TransformationCatalog, SiteCatalog


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

# Example usage
if __name__ == "__main__":
    gen = PegasusWorkflowGenerator("workflow.yaml")
    gen.generate()
    print("Workflow generation process completed successfully.")