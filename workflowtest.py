from Pegasus.api import *
import os

class Falcon_7bWorkflow:
    """
    A Pegasus workflow for falcon-7b.
    """
    def __init__(self, base_dir="."):
        """
        Initialize the workflow, sites, replicas, transformations, and job containers.
        
        :param base_dir: Base directory for the workflow (default: current directory)
        """
        self.base_dir = base_dir
        
        # Change to the workflow directory for all operations
        # This ensures catalog files are written to the correct location
        os.chdir(self.base_dir)
        
        self.wf = Workflow(name="falcon-7b")
        self.sites = SiteCatalog()
        self.replicas = ReplicaCatalog()
        self.transformations = TransformationCatalog()
        self.files = {}
        self.jobs = {}

    def build_sites(self):
        local = Site("local")
        local.add_directories(Directory(Directory.SHARED_SCRATCH, "/home/hsafri/LLM-Fine-Tune/scratch").add_file_servers(FileServer("file:///home/hsafri/LLM-Fine-Tune/scratch", Operation.ALL)))
        local.add_directories(Directory(Directory.LOCAL_STORAGE, "/home/hsafri/LLM-Fine-Tune/output").add_file_servers(FileServer("file:///home/hsafri/LLM-Fine-Tune/output", Operation.ALL)))
        self.sites.add_sites(local)

        condorpool = Site("condorpool")
        condorpool.add_profiles(Namespace.CONDOR, key="universe", value="vanilla")
        condorpool.add_profiles(Namespace.PEGASUS, key="style", value="condor")
        self.sites.add_sites(condorpool)

    def build_replicas(self):
        self.replicas.add_replica("local", "pegasus_data", "/home/hsafri/LLM-Fine-Tune/data/data.json")

    def build_transformations(self):
        container = Container("FineTuneLLM", Container.SINGULARITY, "docker://swarmourr/finetune-pegasus:amd64", image_site="docker_hub")
        self.transformations.add_containers(container)

        transformation = Transformation("FineTuneLLM", site="condorpool", pfn="/home/hsafri/LLM-Fine-Tune/bin/finetune.py", is_stageable=True)
        transformation.add_profiles(Namespace.PEGASUS, key="cores", value="4")
        transformation.add_profiles(Namespace.PEGASUS, key="memory", value="16384")  # Increased memory request
        transformation.add_profiles(Namespace.PEGASUS, key="gpus", value="1")
        self.transformations.add_transformations(transformation)

    def build_jobs(self):
        job = Job("FineTuneLLM", _id="ID0000001")
        job.add_args("--data_path", "pegasus_data", "--model_name", "tiiuae/falcon-7b", "--output_dir", "tiiuae/falcon-7b", "--num_train_epochs", "3", "--batch_size", "4", "--save_steps", "5000", "--learning_rate", "3e-05", "--gpu", "1", "--auth_token", "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")
        job.add_inputs(self._get_file("pegasus_data"))
        job.add_outputs(self._get_file("falcon-7b.zip"), stage_out=True, register_replica=True)
        self.jobs["ID0000001"] = job
        self.wf.add_jobs(job)

    def _get_file(self, name):
        if name not in self.files:
            self.files[name] = File(name)
        return self.files[name]

    def write(self):
        """
        Write the site, replica, transformation, and workflow to their respective catalogs and files.
        """
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
    w = Falcon_7bWorkflow(base_dir=current_dir)
    w.build_sites()
    w.build_replicas()
    w.build_transformations()
    w.build_jobs()
    w.write()
    print("Workflow generated and written successfully in: " + current_dir)