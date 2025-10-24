# Open Cursor Agent

An open-source autonomous AI agent implementation inspired by Cursor Agent, built with the Swarms framework. This production-grade agent can autonomously plan, execute, and complete complex tasks using a combination of Large Language Model reasoning and tool execution.

## Overview

Open Cursor Agent is a sophisticated AI agent capable of:

- **Autonomous Task Planning**: Breaking down complex tasks into manageable, sequential subtasks
- **Multi-Tool Execution**: Leveraging various tools including file operations, command execution, and web search
- **Intelligent Reasoning**: Using LLM-powered thinking to analyze situations and decide next actions
- **State Management**: Tracking task progress through well-defined execution states
- **Error Handling**: Robust error detection and recovery mechanisms

## Features

- File system operations (read, write, search, manage)
- Command execution with timeout and security controls
- Web search integration for real-time information
- Task dependency management with priority awareness
- Execution history tracking and logging
- Workspace isolation with security-first approach

## Installation

### Prerequisites

- Python 3.8 or higher
- API key for your chosen LLM provider (e.g., OpenAI)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/Open-Cursor-Agent.git
cd Open-Cursor-Agent

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

## Usage

```python
import asyncio
from open_cursor.main import OpenCursorAgent

async def main():
    # Initialize the agent
    agent = OpenCursorAgent(
        model_name="gpt-4o",
        workspace_path="./workspace",
        temperature=0.7,
        verbose=True
    )
    
    # Define your task
    task_description = """
    Create a Python script that processes CSV files and generates a summary report.
    """
    
    # Run the agent
    result = await agent.run(task_description)
    
    print(f"Task Status: {result['status']}")

if __name__ == "__main__":
    asyncio.run(main())
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

## Project Structure

```
Open-Cursor-Agent/
├── open_cursor/
│   ├── __init__.py
│   ├── main.py          # Core agent implementation
│   └── prompts.py       # System prompts
├── example.py           # Usage examples
├── requirements.txt     # Project dependencies
├── README.md           # This file
└── LICENSE             # License information
```

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes with appropriate tests
4. Submit a pull request with a clear description

## License

This project is licensed under the terms specified in the LICENSE file.

## Acknowledgments

Built with the Swarms framework and inspired by the Cursor Agent architecture. Special thanks to the open-source community for their contributions to AI agent development.
