from open_cursor.main import OpenCursorAgent

# Initialize the agent
agent = OpenCursorAgent(
    model_name="gpt-4o",
    workspace_path=".",
)

# Example task
task_description = """
Create a transformer model in pytorch in a file called transformer.py"
"""

result = agent.run(task_description)

print(result)
