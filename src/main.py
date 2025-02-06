"""
This module sets up and compiles a workflow application, providing an example
of how to invoke it with a sample input.

Imports:
- setup_workflow: A function from the workflow_setup module used to initialize
the workflow.
- utils: Contains utility functions, including logging functionality.

Workflow Setup:
1. Initializes the workflow using setup_workflow().
2. Compiles the initialized workflow into a callable application object.

Example Invocation:
- Demonstrates how to use the compiled app with a sample state input.
- Logs the result of the invocation for debugging or output purposes.

Purpose:
This module serves as an entry point for setting up and running a workflow-based
application, providing both the setup steps and an example usage.
"""

from workflow_setup import setup_workflow
import utils


# Compile app
workflow = setup_workflow()
app = workflow.compile()


# Example invocation
if __name__ == "__main__":
    state = {"question": "What tasks are Michael involved in?"}
    result = app.invoke(state)
    utils.log_with_horizontal_line(f"Final answer: {result.get('llm_output', 'Process ended.')}")
