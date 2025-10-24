import asyncio
import json
import subprocess
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv
from loguru import logger

from open_cursor.prompts import (
    default_system_prompt,
    execution_agent_system_prompt,
    execution_prompt_no_tasks,
    planning_prompt,
    thinking_agent_system_prompt,
    thinking_prompt,
)
from swarms.structs.agent import Agent

load_dotenv()


class AgentState(Enum):
    """Agent execution states."""

    INITIALIZING = "initializing"
    PLANNING = "planning"
    EXECUTING = "executing"
    THINKING = "thinking"
    COMPLETED = "completed"
    ERROR = "error"
    PAUSED = "paused"


class TaskPriority(Enum):
    """Task priority levels."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Task:
    """Represents a single task in the agent's workflow."""

    id: str
    description: str
    priority: TaskPriority
    status: str = "pending"
    dependencies: List[str] = field(default_factory=list)
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


@dataclass
class AgentContext:
    """Context information for the agent."""

    task_id: str
    workspace_path: str
    current_state: AgentState
    tasks: List[Task] = field(default_factory=list)
    execution_history: List[Dict] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    current_task_index: int = 0


class OpenCursorAgent:
    """
    Production-grade Cursor Agent implementation.

    This agent can autonomously plan, execute, and complete complex tasks
    using a combination of LLM reasoning and tool execution.
    """

    def __init__(
        self,
        agent_name: str = "OpenCursorAgent",
        agent_description: str = "A production-grade Cursor Agent implementation.",
        model_name: str = "gpt-4o",
        system_prompt: Optional[str] = None,
        workspace_path: str = ".",
        temperature: float = 0.7,
        max_tokens: int = 8000,
        verbose: bool = True,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the Cursor Agent.

        Args:
            model_name: LLM model to use for reasoning
            system_prompt: Custom system prompt (uses default if None)
            workspace_path: Working directory for the agent
            temperature: LLM temperature for creativity
            max_tokens: Maximum tokens per LLM call
            verbose: Enable verbose logging
            api_key: API key for LLM service
            base_url: Base URL for LLM service
            **kwargs: Additional arguments for Agent
        """
        self.model_name = model_name
        self.workspace_path = Path(workspace_path).resolve()
        self.verbose = verbose

        # Initialize logging
        self._setup_logging()

        # Initialize Agent with tool support
        self.llm = Agent(
            model_name=model_name,
            system_prompt=system_prompt or default_system_prompt(),
            temperature=temperature,
            max_tokens=max_tokens,
            tools_list_dictionary=self._get_tool_definitions(),
            tool_choice="auto",
            output_type="final",
            **kwargs,
        )

        # Agent state
        self.context: Optional[AgentContext] = None
        self.is_running = False
        self.max_iterations = 50  # Prevent infinite loops
        self.current_iteration = 0

        # Tool registry
        self.tools = self._initialize_tools()

        logger.info(f"Cursor Agent initialized with model: {model_name}")

    def _setup_logging(self):
        return logger

    def _get_execution_tool_definitions(self) -> List[Dict]:
        """Get tool definitions for execution phase (no planning tools)."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "start_line": {"type": "integer"},
                            "end_line": {"type": "integer"},
                        },
                        "required": ["file_path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "content": {"type": "string"},
                            "append": {"type": "boolean"},
                        },
                        "required": ["file_path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_files",
                    "description": "Search for files matching a pattern",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string"},
                            "directory": {"type": "string"},
                            "file_types": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["pattern"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": "List contents of a directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "directory": {"type": "string"},
                            "recursive": {"type": "boolean"},
                            "include_hidden": {"type": "boolean"},
                        },
                        "required": ["directory"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_command",
                    "description": "Execute a shell command",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                            "working_directory": {"type": "string"},
                            "timeout": {"type": "integer"},
                        },
                        "required": ["command"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "num_results": {"type": "integer"},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "create_directory",
                    "description": "Create a new directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "directory": {"type": "string"},
                            "parents": {"type": "boolean"},
                        },
                        "required": ["directory"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_file",
                    "description": "Delete a file or directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "recursive": {"type": "boolean"},
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "subtask_done",
                    "description": "Mark a subtask as completed and move to the next task in the plan",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_id": {"type": "string"},
                            "summary": {"type": "string"},
                            "success": {"type": "boolean"},
                        },
                        "required": ["task_id", "summary", "success"],
                    },
                },
            },
        ]

    def _get_tool_definitions(self) -> List[Dict]:
        """Get tool definitions for the LLM."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "create_plan",
                    "description": "Create a detailed plan for completing a task",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_description": {
                                "type": "string",
                                "description": "Description of the task to be completed",
                            },
                            "steps": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "step_id": {"type": "string"},
                                        "description": {"type": "string"},
                                        "priority": {
                                            "type": "string",
                                            "enum": [
                                                "low",
                                                "medium",
                                                "high",
                                                "critical",
                                            ],
                                        },
                                        "dependencies": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "tools_needed": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                    },
                                    "required": [
                                        "step_id",
                                        "description",
                                        "priority",
                                    ],
                                },
                            },
                            "estimated_duration": {
                                "type": "string",
                                "description": "Estimated time to complete the task",
                            },
                        },
                        "required": ["task_description", "steps"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "think",
                    "description": "Analyze current situation and decide next actions",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "current_state": {
                                "type": "string",
                                "description": "Current state of the task execution",
                            },
                            "last_action_result": {
                                "type": "string",
                                "description": "Result of the last action performed",
                            },
                            "analysis": {
                                "type": "string",
                                "description": "Analysis of the current situation",
                            },
                            "next_actions": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of next actions to take",
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                                "description": "Confidence level in the analysis (0-1)",
                            },
                        },
                        "required": [
                            "current_state",
                            "analysis",
                            "next_actions",
                            "confidence",
                        ],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "complete_task",
                    "description": "Mark a task as complete and provide summary",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_id": {"type": "string"},
                            "summary": {"type": "string"},
                            "results": {"type": "string"},
                            "success": {"type": "boolean"},
                            "lessons_learned": {"type": "string"},
                        },
                        "required": ["task_id", "summary", "success"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "start_line": {"type": "integer"},
                            "end_line": {"type": "integer"},
                        },
                        "required": ["file_path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "content": {"type": "string"},
                            "append": {"type": "boolean"},
                        },
                        "required": ["file_path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_files",
                    "description": "Search for files matching a pattern",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string"},
                            "directory": {"type": "string"},
                            "file_types": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["pattern"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": "List contents of a directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "directory": {"type": "string"},
                            "recursive": {"type": "boolean"},
                            "include_hidden": {"type": "boolean"},
                        },
                        "required": ["directory"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_command",
                    "description": "Execute a shell command",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                            "working_directory": {"type": "string"},
                            "timeout": {"type": "integer"},
                        },
                        "required": ["command"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "num_results": {"type": "integer"},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "create_directory",
                    "description": "Create a new directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "directory": {"type": "string"},
                            "parents": {"type": "boolean"},
                        },
                        "required": ["directory"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_file",
                    "description": "Delete a file or directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "recursive": {"type": "boolean"},
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "subtask_done",
                    "description": "Mark a subtask as completed and move to the next task in the plan",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_id": {"type": "string"},
                            "summary": {"type": "string"},
                            "success": {"type": "boolean"},
                        },
                        "required": ["task_id", "summary", "success"],
                    },
                },
            },
        ]

    def _initialize_tools(self) -> Dict[str, Callable]:
        """Initialize the tool registry."""
        return {
            "create_plan": self._create_plan,
            "think": self._think,
            "complete_task": self._complete_task,
            "subtask_done": self._subtask_done,
            "read_file": self._read_file,
            "write_file": self._write_file,
            "search_files": self._search_files,
            "list_directory": self._list_directory,
            "execute_command": self._execute_command,
            "web_search": self._web_search,
            "create_directory": self._create_directory,
            "delete_file": self._delete_file,
        }

    def _get_execution_tools(self) -> Dict[str, Callable]:
        """Get tools available during execution phase (no complete_task)."""
        return {
            "subtask_done": self._subtask_done,
            "read_file": self._read_file,
            "write_file": self._write_file,
            "search_files": self._search_files,
            "list_directory": self._list_directory,
            "execute_command": self._execute_command,
            "web_search": self._web_search,
            "create_directory": self._create_directory,
            "delete_file": self._delete_file,
        }

    async def _run(
        self, task_description: str, task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the agent on a given task.

        Args:
            task_description: Description of the task to complete
            task_id: Optional custom task ID

        Returns:
            Dictionary containing execution results
        """
        if self.is_running:
            raise RuntimeError("Agent is already running")

        self.is_running = True
        self.current_iteration = 0  # Reset iteration counter
        task_id = task_id or str(uuid.uuid4())

        try:
            # Initialize context
            self.context = AgentContext(
                task_id=task_id,
                workspace_path=str(self.workspace_path),
                current_state=AgentState.INITIALIZING,
            )

            # Main execution loop
            while self.context.current_state not in [
                AgentState.COMPLETED,
                AgentState.ERROR,
            ]:
                # Check for infinite loop
                self.current_iteration += 1
                if self.current_iteration > self.max_iterations:
                    logger.error(
                        f"Maximum iterations ({self.max_iterations}) reached. Stopping execution."
                    )
                    self.context.current_state = AgentState.ERROR
                    break

                logger.info(
                    f"Current state: {self.context.current_state.value} (iteration {self.current_iteration})"
                )

                try:
                    if self.context.current_state == AgentState.INITIALIZING:
                        await self._initialize_task(task_description)
                    elif self.context.current_state == AgentState.PLANNING:
                        await self._planning_phase()
                    elif self.context.current_state == AgentState.EXECUTING:
                        await self._execution_phase()
                    elif self.context.current_state == AgentState.THINKING:
                        await self._thinking_phase()
                    else:
                        logger.warning(f"Unknown state: {self.context.current_state}")
                        break

                except Exception as e:
                    logger.error(f"Error in execution: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    self.context.current_state = AgentState.ERROR
                    break

                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.1)

            # Finalize execution
            result = await self._finalize_execution()

            logger.info(f"Task execution completed: {task_id}")
            return result

        finally:
            self.is_running = False

    def run(self, task: str):
        return asyncio.run(self._run(task_description=task))

    async def _initialize_task(self, task_description: str):
        """Initialize the task and start planning."""
        logger.info("Initializing task...")

        # Create initial task
        main_task = Task(
            id="main_task",
            description=task_description,
            priority=TaskPriority.HIGH,
        )
        self.context.tasks.append(main_task)

        # Move to planning phase
        self.context.current_state = AgentState.PLANNING
        logger.info("Task initialized, moving to planning phase")

    async def _planning_phase(self):
        """Execute the planning phase."""
        logger.info("Executing planning phase...")

        # Get the main task
        main_task = next(
            (t for t in self.context.tasks if t.id == "main_task"),
            None,
        )
        if not main_task:
            raise ValueError("Main task not found")

        # Use LLM to create a plan
        planning_prompt_text = planning_prompt(
            main_task.description, str(self.workspace_path)
        )

        try:
            response = self.llm.run(planning_prompt_text)

            logger.debug(f"Planning response: {response}")

            # Check if the response contains a create_plan tool call
            if isinstance(response, list) and len(response) > 0:
                tool_call = response[0]
                if (
                    isinstance(tool_call, dict)
                    and tool_call.get("function", {}).get("name") == "create_plan"
                ):
                    # Execute the create_plan tool call manually
                    arguments = json.loads(tool_call["function"]["arguments"])
                    result = await self._create_plan(**arguments)
                    logger.info(f"Plan created successfully: {result}")

                    # Move to execution phase
                    self.context.current_state = AgentState.EXECUTING
                    logger.info("Planning completed, moving to execution phase")
                else:
                    logger.error("Expected create_plan tool call in planning phase")
                    self.context.current_state = AgentState.ERROR
            elif response:
                logger.info(f"Planning completed: {response}")

                # Move to execution phase
                self.context.current_state = AgentState.EXECUTING
                logger.info("Planning completed, moving to execution phase")
            else:
                logger.error("No valid response from Agent during planning")
                self.context.current_state = AgentState.ERROR

        except Exception as e:
            logger.error(f"Error in planning phase: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.context.current_state = AgentState.ERROR

    async def _execution_phase(self):
        """Execute actions based on the plan."""
        logger.info("Executing actions...")

        # Find the next task to execute
        next_task = self._get_next_executable_task()

        # Debug logging for task status
        logger.info("Current task status:")
        for task in self.context.tasks:
            logger.info(f"  - {task.id}: {task.status} - {task.description}")

        if next_task:
            logger.info(
                f"Executing next task: {next_task.id} - {next_task.description}"
            )

            # Create a more direct execution prompt
            execution_prompt_text = f"""
You need to execute this specific task: {next_task.description}

Task ID: {next_task.id}
Priority: {next_task.priority.value}
Dependencies: {next_task.dependencies}

Based on the task description, choose and execute the appropriate tool:

1. If the task involves creating a directory, use create_directory
2. If the task involves writing code or files, use write_file with actual content (not empty)
3. If the task involves running commands, use execute_command
4. If the task involves reading files, use read_file
5. If the task involves listing directories, use list_directory

IMPORTANT: 
- For code files, include actual working code with proper structure, functions, and functionality
- Write meaningful content, not empty files
- After executing the task, use subtask_done to mark it as completed

Execute the task now using the appropriate tool.
"""

        else:
            logger.info("No executable tasks found")

            # No tasks to execute, check if we should complete
            execution_prompt_text = execution_prompt_no_tasks(
                workspace_path=str(self.workspace_path),
                tasks_status=[
                    (task.id, task.status, task.description)
                    for task in self.context.tasks
                ],
                execution_history=self.context.execution_history,
            )

        try:
            # Create a temporary Agent instance without planning tools for execution
            execution_agent = Agent(
                model_name=self.model_name,
                system_prompt=execution_agent_system_prompt(),
                temperature=self.llm.temperature,
                tools_list_dictionary=self._get_execution_tool_definitions(),
                tool_choice="auto",
                output_type="final",
            )

            response = execution_agent.run(execution_prompt_text)

            logger.debug(f"Execution response: {response}")

            # Handle tool calls from the execution agent
            if isinstance(response, list) and len(response) > 0:
                tool_call = response[0]
                if isinstance(tool_call, dict) and "function" in tool_call:
                    function_name = tool_call["function"]["name"]
                    arguments = json.loads(tool_call["function"]["arguments"])

                    logger.info(f"Executing tool: {function_name}")

                    # Execute the tool call
                    if function_name in self.tools:
                        result = await self.tools[function_name](**arguments)
                        logger.info(
                            f"Tool {function_name} executed successfully: {result}"
                        )

                        # Mark the current task as completed if we have one and it's not subtask_done
                        if next_task and function_name != "subtask_done":
                            next_task.status = "completed"
                            next_task.completed_at = datetime.now()
                            next_task.result = str(result)
                            logger.info(
                                f"Task {next_task.id} marked as completed: {next_task.description}"
                            )

                            # Also mark the main task as completed if all plan steps are done
                            if self._all_plan_steps_completed():
                                main_task = next(
                                    (
                                        t
                                        for t in self.context.tasks
                                        if t.id == "main_task"
                                    ),
                                    None,
                                )
                                if main_task:
                                    main_task.status = "completed"
                                    main_task.completed_at = datetime.now()
                                    logger.info(
                                        "Main task marked as completed - all plan steps finished"
                                    )
                    else:
                        logger.error(f"Unknown tool: {function_name}")
                        self.context.current_state = AgentState.ERROR
                        return
            elif response:
                logger.info(f"Execution completed: {response}")

                # Mark the current task as completed if we have one
                if next_task:
                    next_task.status = "completed"
                    next_task.completed_at = datetime.now()
                    next_task.result = str(response)
                    logger.info(
                        f"Task {next_task.id} marked as completed: {next_task.description}"
                    )

                    # Also mark the main task as completed if all plan steps are done
                    if self._all_plan_steps_completed():
                        main_task = next(
                            (t for t in self.context.tasks if t.id == "main_task"),
                            None,
                        )
                        if main_task:
                            main_task.status = "completed"
                            main_task.completed_at = datetime.now()
                            logger.info(
                                "Main task marked as completed - all plan steps finished"
                            )
            else:
                logger.error("No valid response from Agent during execution")
                self.context.current_state = AgentState.ERROR
                return

            # Move to thinking phase
            self.context.current_state = AgentState.THINKING
            logger.info("Execution completed, moving to thinking phase")

        except Exception as e:
            logger.error(f"Error in execution phase: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.context.current_state = AgentState.ERROR

    async def _thinking_phase(self):
        """Execute the thinking phase."""
        logger.info("Executing thinking phase...")

        thinking_prompt_text = thinking_prompt(
            current_state=str(self.context.current_state),
            execution_history=self.context.execution_history,
            tasks_status=[
                (task.id, task.status, task.description) for task in self.context.tasks
            ],
        )

        try:
            # Create a temporary Agent instance for thinking phase
            thinking_agent = Agent(
                model_name=self.model_name,
                system_prompt=thinking_agent_system_prompt(),
                temperature=self.llm.temperature,
                tools_list_dictionary=[
                    {
                        "type": "function",
                        "function": {
                            "name": "think",
                            "description": "Analyze current situation and decide next actions",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "current_state": {
                                        "type": "string",
                                        "description": "Current state of the task execution",
                                    },
                                    "last_action_result": {
                                        "type": "string",
                                        "description": "Result of the last action performed",
                                    },
                                    "analysis": {
                                        "type": "string",
                                        "description": "Analysis of the current situation",
                                    },
                                    "next_actions": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "List of next actions to take",
                                    },
                                    "confidence": {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 1,
                                        "description": "Confidence level in the analysis (0-1)",
                                    },
                                },
                                "required": [
                                    "current_state",
                                    "analysis",
                                    "next_actions",
                                    "confidence",
                                ],
                            },
                        },
                    }
                ],
                tool_choice="auto",
                output_type="final",
            )

            response = thinking_agent.run(thinking_prompt_text)

            logger.debug(f"Thinking response: {response}")

            # Handle tool calls from the thinking agent
            if isinstance(response, list) and len(response) > 0:
                tool_call = response[0]
                if isinstance(tool_call, dict) and "function" in tool_call:
                    function_name = tool_call["function"]["name"]
                    arguments = json.loads(tool_call["function"]["arguments"])

                    logger.info(f"Executing thinking tool: {function_name}")

                    # Execute the tool call
                    if function_name in self.tools:
                        result = await self.tools[function_name](**arguments)
                        logger.info(
                            f"Thinking tool {function_name} executed successfully: {result}"
                        )
                    else:
                        logger.error(f"Unknown thinking tool: {function_name}")
                        self.context.current_state = AgentState.ERROR
                        return
            elif response:
                logger.info(f"Thinking completed: {response}")
            else:
                logger.error("No valid response from Agent during thinking")
                self.context.current_state = AgentState.ERROR
                return

            # Check if we should complete or continue
            if self._should_complete_task():
                self.context.current_state = AgentState.COMPLETED
            else:
                # Continue with execution
                self.context.current_state = AgentState.EXECUTING

        except Exception as e:
            logger.error(f"Error in thinking phase: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.context.current_state = AgentState.ERROR

    async def _execute_tool_calls(self, tool_calls):
        """Execute tool calls from the LLM response."""
        logger.info(f"Executing {len(tool_calls)} tool calls...")

        for tool_call in tool_calls:
            try:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)

                logger.info(f"Executing tool: {function_name} with args: {arguments}")

                # Execute the tool
                if function_name in self.tools:
                    result = await self.tools[function_name](**arguments)

                    # Record execution
                    execution_record = {
                        "tool": function_name,
                        "arguments": arguments,
                        "result": str(result),
                        "timestamp": datetime.now().isoformat(),
                    }
                    self.context.execution_history.append(execution_record)

                    logger.info(f"Tool {function_name} executed successfully")
                else:
                    logger.error(f"Unknown tool: {function_name}")

            except json.JSONDecodeError as e:
                logger.error(f"Error parsing tool call arguments: {str(e)}")
                logger.error(f"Raw arguments: {tool_call.function.arguments}")
            except Exception as e:
                logger.error(
                    f"Error executing tool {tool_call.function.name}: {str(e)}"
                )
                logger.error(f"Traceback: {traceback.format_exc()}")

    def _get_next_executable_task(self) -> Optional[Task]:
        """Get the next task that can be executed based on the current task index."""
        # Get all subtasks (excluding main_task)
        subtasks = [t for t in self.context.tasks if t.id != "main_task"]

        if not subtasks:
            return None

        # Find the first pending task
        for task in subtasks:
            if task.status == "pending":
                return task

        return None

    def _is_task_successful(self, function_name: str, result: str) -> bool:
        """Determine if a task was completed successfully based on the function and result."""
        result_lower = str(result).lower()

        # Check for success indicators in the result
        success_indicators = [
            "successfully",
            "created",
            "completed",
            "finished",
            "done",
            "wrote",
            "installed",
            "executed",
        ]

        # Check for error indicators
        error_indicators = [
            "error",
            "failed",
            "exception",
            "traceback",
            "not found",
            "permission denied",
            "access denied",
            "timeout",
        ]

        # If there are error indicators, it's not successful
        if any(indicator in result_lower for indicator in error_indicators):
            return False

        # For specific functions, check for appropriate success indicators
        if function_name == "create_directory":
            return "created directory" in result_lower or "successfully" in result_lower
        elif function_name == "write_file":
            return "wrote" in result_lower or "successfully" in result_lower
        elif function_name == "execute_command":
            return "success" in result_lower or result_lower.count("error") == 0
        elif function_name == "web_search":
            return "search" in result_lower and "error" not in result_lower
        else:
            # For other functions, check for general success indicators
            return any(indicator in result_lower for indicator in success_indicators)

    def _all_plan_steps_completed(self) -> bool:
        """Check if all plan steps (excluding main_task) are completed."""
        plan_tasks = [task for task in self.context.tasks if task.id != "main_task"]
        return all(task.status == "completed" for task in plan_tasks)

    def _should_complete_task(self) -> bool:
        """Determine if the task should be completed."""
        # Check if all subtasks are completed
        subtasks = [t for t in self.context.tasks if t.id != "main_task"]
        all_subtasks_completed = all(
            task.status in ["completed", "failed"] for task in subtasks
        )

        return all_subtasks_completed

    async def _finalize_execution(self) -> Dict[str, Any]:
        """Finalize the execution and return results."""
        logger.info("Finalizing execution...")

        # Create completion summary
        summary = {
            "task_id": self.context.task_id,
            "status": self.context.current_state.value,
            "tasks": [
                {
                    "id": task.id,
                    "description": task.description,
                    "status": task.status,
                    "priority": task.priority.value,
                    "result": (str(task.result) if task.result else None),
                    "error": task.error,
                }
                for task in self.context.tasks
            ],
            "execution_history": self.context.execution_history,
            "workspace_path": str(self.workspace_path),
            "completed_at": datetime.now().isoformat(),
        }

        logger.info(
            f"Execution finalized with status: {self.context.current_state.value}"
        )
        return summary

    # Tool implementations
    async def _create_plan(
        self, task_description: str, steps: List[Dict], **kwargs
    ) -> str:
        """Create a detailed plan for task execution."""
        logger.info(f"Creating plan for task: {task_description}")

        # Create task objects from steps
        for step in steps:
            task = Task(
                id=step["step_id"],
                description=step["description"],
                priority=TaskPriority[step["priority"].upper()],
                dependencies=step.get("dependencies", []),
            )
            self.context.tasks.append(task)
            logger.info(f"Added task {step['step_id']}: {step['description']}")

        logger.info(f"Plan created with {len(steps)} steps")
        logger.info(f"Total tasks in context: {len(self.context.tasks)}")

        # Log all tasks for debugging
        for task in self.context.tasks:
            logger.info(f"Task: {task.id} - {task.description} - Status: {task.status}")

        return f"Plan created with {len(steps)} steps"

    async def _think(
        self,
        current_state: str,
        analysis: str,
        next_actions: List[str],
        confidence: float,
        **kwargs,
    ) -> str:
        """Analyze current situation and plan next actions."""
        logger.info(f"Thinking: {analysis}")
        logger.info(f"Next actions: {next_actions}")
        logger.info(f"Confidence: {confidence}")

        # Update context based on analysis
        if confidence < 0.5:
            logger.warning(
                "Low confidence in analysis, may need to reconsider approach"
            )

        return f"Analysis complete. Confidence: {confidence}. Next actions: {', '.join(next_actions)}"

    async def _complete_task(
        self, task_id: str, summary: str, success: bool, **kwargs
    ) -> str:
        """Mark a task as complete."""
        logger.info(f"Completing task {task_id}: {summary}")

        # Find and update task
        task = next((t for t in self.context.tasks if t.id == task_id), None)
        if task:
            task.status = "completed" if success else "failed"
            task.completed_at = datetime.now()
            task.result = summary
        else:
            logger.warning(f"Task {task_id} not found")

        # Check if all tasks are complete
        if all(t.status in ["completed", "failed"] for t in self.context.tasks):
            self.context.current_state = AgentState.COMPLETED

        return f"Task {task_id} marked as {'completed' if success else 'failed'}"

    async def _subtask_done(
        self, task_id: str, summary: str, success: bool, **kwargs
    ) -> str:
        """Mark a subtask as completed and move to the next task in the plan."""
        logger.info(f"Completing subtask {task_id}: {summary}")

        # Find and update task
        task = next((t for t in self.context.tasks if t.id == task_id), None)
        if task:
            task.status = "completed" if success else "failed"
            task.completed_at = datetime.now()
            task.result = summary

            # Move to the next task in the plan
            if success:
                self.context.current_task_index += 1
                logger.info(
                    f"Moved to next task. Current index: {self.context.current_task_index}"
                )
            else:
                logger.warning(f"Subtask {task_id} failed, staying on current task")
        else:
            logger.warning(f"Task {task_id} not found")

        # Check if all subtasks are complete
        subtasks = [t for t in self.context.tasks if t.id != "main_task"]
        all_subtasks_complete = all(
            t.status in ["completed", "failed"] for t in subtasks
        )

        if all_subtasks_complete:
            # Mark main task as completed
            main_task = next(
                (t for t in self.context.tasks if t.id == "main_task"),
                None,
            )
            if main_task:
                main_task.status = "completed"
                main_task.completed_at = datetime.now()
                logger.info("All subtasks completed, marking main task as completed")
                self.context.current_state = AgentState.COMPLETED

        return f"Subtask {task_id} marked as {'completed' if success else 'failed'}"

    async def _read_file(
        self,
        file_path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Read contents of a file."""
        try:
            full_path = self.workspace_path / file_path
            full_path = full_path.resolve()

            # Security check - ensure path is within workspace
            if not str(full_path).startswith(str(self.workspace_path)):
                raise ValueError("File path outside workspace")

            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()

            if start_line is not None and end_line is not None:
                lines = content.split("\n")
                selected_lines = lines[start_line - 1 : end_line]
                content = "\n".join(selected_lines)

            logger.info(f"Read file: {file_path} ({len(content)} characters)")
            return content

        except Exception as e:
            error_msg = f"Error reading file {file_path}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    async def _write_file(
        self,
        file_path: str,
        content: str,
        append: bool = False,
        **kwargs,
    ) -> str:
        """Write content to a file."""
        try:
            full_path = self.workspace_path / file_path
            full_path = full_path.resolve()

            # Security check - ensure path is within workspace
            if not str(full_path).startswith(str(self.workspace_path)):
                raise ValueError("File path outside workspace")

            # Create parent directories if needed
            full_path.parent.mkdir(parents=True, exist_ok=True)

            mode = "a" if append else "w"
            with open(full_path, mode, encoding="utf-8") as f:
                f.write(content)

            logger.info(f"Wrote file: {file_path} ({len(content)} characters)")
            return f"Successfully wrote {len(content)} characters to {file_path}"

        except Exception as e:
            error_msg = f"Error writing file {file_path}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    async def _search_files(
        self,
        pattern: str,
        directory: Optional[str] = None,
        file_types: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """Search for files matching a pattern."""
        try:
            search_dir = self.workspace_path
            if directory:
                search_dir = self.workspace_path / directory
                search_dir = search_dir.resolve()

                # Security check
                if not str(search_dir).startswith(str(self.workspace_path)):
                    raise ValueError("Directory outside workspace")

            import glob

            # Build search pattern
            if file_types:
                patterns = [f"{search_dir}/**/*.{ext}" for ext in file_types]
            else:
                patterns = [f"{search_dir}/**/*"]

            matches = []
            for pattern_path in patterns:
                matches.extend(glob.glob(pattern_path, recursive=True))

            # Filter by pattern if provided
            if pattern:
                import re

                regex = re.compile(pattern, re.IGNORECASE)
                matches = [m for m in matches if regex.search(Path(m).name)]

            # Convert to relative paths
            relative_matches = [
                str(Path(m).relative_to(self.workspace_path)) for m in matches
            ]

            logger.info(
                f"Found {len(relative_matches)} files matching pattern: {pattern}"
            )
            return json.dumps(relative_matches, indent=2)

        except Exception as e:
            error_msg = f"Error searching files: {str(e)}"
            logger.error(error_msg)
            return error_msg

    async def _list_directory(
        self,
        directory: str,
        recursive: bool = False,
        include_hidden: bool = False,
        **kwargs,
    ) -> str:
        """List contents of a directory."""
        try:
            full_path = self.workspace_path / directory
            full_path = full_path.resolve()

            # Security check
            if not str(full_path).startswith(str(self.workspace_path)):
                raise ValueError("Directory outside workspace")

            if not full_path.exists():
                return f"Directory does not exist: {directory}"

            if not full_path.is_dir():
                return f"Path is not a directory: {directory}"

            items = []
            if recursive:
                for item in full_path.rglob("*"):
                    if not include_hidden and item.name.startswith("."):
                        continue
                    items.append(
                        {
                            "name": str(item.relative_to(self.workspace_path)),
                            "type": ("directory" if item.is_dir() else "file"),
                            "size": (item.stat().st_size if item.is_file() else None),
                        }
                    )
            else:
                for item in full_path.iterdir():
                    if not include_hidden and item.name.startswith("."):
                        continue
                    items.append(
                        {
                            "name": str(item.relative_to(self.workspace_path)),
                            "type": ("directory" if item.is_dir() else "file"),
                            "size": (item.stat().st_size if item.is_file() else None),
                        }
                    )

            logger.info(f"Listed directory: {directory} ({len(items)} items)")
            return json.dumps(items, indent=2)

        except Exception as e:
            error_msg = f"Error listing directory {directory}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    async def _execute_command(
        self,
        command: str,
        working_directory: Optional[str] = None,
        timeout: int = 30,
        **kwargs,
    ) -> str:
        """Execute a shell command."""
        try:
            cwd = self.workspace_path
            if working_directory:
                cwd = self.workspace_path / working_directory
                cwd = cwd.resolve()

                # Security check
                if not str(cwd).startswith(str(self.workspace_path)):
                    raise ValueError("Working directory outside workspace")

            logger.info(f"Executing command: {command} in {cwd}")

            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            output = {
                "command": command,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0,
            }

            logger.info(f"Command executed with return code: {result.returncode}")
            return json.dumps(output, indent=2)

        except subprocess.TimeoutExpired:
            error_msg = f"Command timed out after {timeout} seconds: {command}"
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Error executing command {command}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    async def _web_search(self, query: str, num_results: int = 5, **kwargs) -> str:
        """Search the web for information."""
        try:
            from swarms_tools import exa_search

            return exa_search(query)
        except Exception as e:
            error_msg = f"Error in web search: {str(e)}"
            logger.error(error_msg)
            return error_msg

    async def _create_directory(
        self, directory: str, parents: bool = True, **kwargs
    ) -> str:
        """Create a new directory."""
        try:
            full_path = self.workspace_path / directory
            full_path = full_path.resolve()

            # Security check
            if not str(full_path).startswith(str(self.workspace_path)):
                raise ValueError("Directory path outside workspace")

            full_path.mkdir(parents=parents, exist_ok=True)

            logger.info(f"Created directory: {directory}")
            return f"Successfully created directory: {directory}"

        except Exception as e:
            error_msg = f"Error creating directory {directory}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    async def _delete_file(self, path: str, recursive: bool = False, **kwargs) -> str:
        """Delete a file or directory."""
        try:
            full_path = self.workspace_path / path
            full_path = full_path.resolve()

            # Security check
            if not str(full_path).startswith(str(self.workspace_path)):
                raise ValueError("Path outside workspace")

            if not full_path.exists():
                return f"Path does not exist: {path}"

            if full_path.is_file():
                full_path.unlink()
                logger.info(f"Deleted file: {path}")
                return f"Successfully deleted file: {path}"
            elif full_path.is_dir():
                if recursive:
                    import shutil

                    shutil.rmtree(full_path)
                    logger.info(f"Deleted directory recursively: {path}")
                    return f"Successfully deleted directory: {path}"
                else:
                    return f"Cannot delete directory without recursive=True: {path}"
            else:
                return f"Path is neither file nor directory: {path}"

        except Exception as e:
            error_msg = f"Error deleting {path}: {str(e)}"
            logger.error(error_msg)
            return error_msg
