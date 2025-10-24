# Open Cursor Agent

[![Powered by Swarms](https://img.shields.io/badge/Powered%20by-Swarms-blue)](https://github.com/kyegomez/swarms)
[![Swarms Framework](https://img.shields.io/badge/Built%20with-Swarms%20Framework-orange)](https://docs.swarms.world)

An open-source autonomous AI agent implementation inspired by Cursor Agent, built on top of **[Swarms](https://github.com/kyegomez/swarms)** - the enterprise-grade production-ready multi-agent orchestration framework. This production-grade agent can autonomously plan, execute, and complete complex tasks using a combination of Large Language Model reasoning and tool execution.

> **Built with Swarms Framework** - Leveraging the power of [Swarms](https://github.com/kyegomez/swarms), the leading open-source framework for building production-ready multi-agent systems. Swarms provides the robust infrastructure, agent orchestration, and enterprise-grade reliability that makes this agent possible.

## Overview

Open Cursor Agent is a sophisticated AI agent capable of:

- **Autonomous Task Planning**: Breaking down complex tasks into manageable, sequential subtasks
- **Multi-Tool Execution**: Leveraging various tools including file operations, command execution, and web search
- **Intelligent Reasoning**: Using LLM-powered thinking to analyze situations and decide next actions
- **State Management**: Tracking task progress through well-defined execution states
- **Error Handling**: Robust error detection and recovery mechanisms

## Features

| Feature                                             | Description                                                 |
|-----------------------------------------------------|-------------------------------------------------------------|
| File system operations                              | Read, write, search, and manage files                       |
| Command execution                                   | Execute commands with timeout and security controls         |
| Web search integration                              | Access real-time information via web search                 |
| Task dependency management                          | Manage tasks with priority awareness                        |
| Execution history tracking and logging              | Record and monitor action history and logs                  |
| Workspace isolation                                | Ensure security-first approach to isolate workspace         |

## Installation

### Prerequisites

- Python 3.8 or higher
- API key for your chosen LLM provider (e.g., OpenAI)

### Setup

```bash
# Clone the repository
git clone https://github.com/kyegomez/Open-Cursor-Agent
cd Open-Cursor-Agent

# Install dependencies
pip install -r requirements.txt
```


## Environment Variables


```txt
WORKSPACE_DIR=""
OPENAI_API_KEY=""
ANTHROPIC_API_KEY=""
```

## Usage

```python
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
```

## Architecture

```mermaid
graph LR
    A[User Task] --> B[Initialize]
    B --> C[Planning]
    C --> D[Execution]
    D --> E[Thinking]
    E --> F{Complete?}
    F -->|No| D
    F -->|Yes| G[Results]
    
    C -.-> H[LLM]
    D -.-> H
    E -.-> H
    D -.-> I[Tools]
    
    style B fill:#4a90e2,color:#fff
    style C fill:#9b59b6,color:#fff
    style D fill:#e74c3c,color:#fff
    style E fill:#f39c12,color:#fff
    style G fill:#27ae60,color:#fff
```

### Execution Flow

The agent operates through a state machine with the following phases:

1. **Initialization**: Task context is created and main task is registered
2. **Planning Phase**: LLM generates a detailed execution plan with subtasks
3. **Execution Phase**: Each subtask is executed using appropriate tools
4. **Thinking Phase**: Results are analyzed and next actions determined
5. **Completion**: All tasks are finalized and results are returned

### Agent States

- `INITIALIZING`: Setting up the task context
- `PLANNING`: Creating a detailed execution plan
- `EXECUTING`: Performing planned actions
- `THINKING`: Analyzing results and determining next steps
- `COMPLETED`: Task successfully finished
- `ERROR`: Error encountered during execution
- `PAUSED`: Execution temporarily halted


## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes with appropriate tests
4. Submit a pull request with a clear description

## License

This project is licensed under the terms specified in the LICENSE file.

## Acknowledgments

**Special Thanks**: To [Swarms Team](https://twitter.com/swarms_corp) and the entire Swarms community for building the infrastructure that makes advanced AI agents accessible to everyone. This project stands on the shoulders of giants.
