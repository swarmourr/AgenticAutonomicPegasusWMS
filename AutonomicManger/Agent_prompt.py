class AgentPrompt:
  """
  This class contains the prompts for the different agents in the system.
  """

  # Prompts for the different agents
  code_generator_prompt = """Generate Python code based on the given requirements."""
  code_refactorer_prompt = """Refactor the provided Python code to improve readability and performance."""
  code_analyzer_prompt = """Analyze the given Python code and provide feedback on potential issues."""
  workflow_generator_prompt = """Generate a workflow diagram based on the provided system requirements."""

  @classmethod
  def get_code_generator_prompt(cls):
    return cls.code_generator_prompt

  @classmethod
  def get_code_refactorer_prompt(cls):
    return cls.code_refactorer_prompt

  @classmethod
  def get_code_analyzer_prompt(cls):
    return cls.code_analyzer_prompt

  @classmethod
  def get_workflow_generator_prompt(cls):
    return cls.workflow_generator_prompt

  @classmethod
  def get_extended_prompt(cls, agent_type, additional_input):
    """
    Extend the prompt with additional input and return the result.
    """
    prompts = {
      "code_generator": cls.code_generator_prompt,
      "code_refactorer": cls.code_refactorer_prompt,
      "code_analyzer": cls.code_analyzer_prompt,
      "workflow_generator": cls.workflow_generator_prompt,
    }
    base_prompt = prompts.get(agent_type, "Invalid agent type.")
    return f"{base_prompt}\n\nAdditional Input: {additional_input}"
