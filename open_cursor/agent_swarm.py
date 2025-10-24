from swarms import HierarchicalSwarm
from open_cursor.main import OpenCursorAgent
from typing import List, Union, Any, Callable, Optional
from swarms.structs.agent import Agent


class DevelopmentTeam:
    def __init__(
        self,
        name: str,
        description: str,
        agents: List[OpenCursorAgent] = [],
        max_loops: int = 1,
        output_type: str = "dict-all-except-first",
        feedback_director_model_name: str = "gpt-4.1",
        director_name: str = "Director",
        director_model_name: str = "gpt-4.1",
        verbose: bool = False,
        add_collaboration_prompt: bool = True,
        planning_director_agent: Optional[Union[Agent, Callable, Any]] = None,
        director_feedback_on: bool = True,
        interactive: bool = False,
        director_reasoning_model_name: str = "o3-mini",
        director_reasoning_enabled: bool = False,
        multi_agent_prompt_improvements: bool = False,
        n_of_workers: int = 3,
        workspace_path: str = ".",
        temperature: float = 0.5,
    ):
        self.name = name
        self.description = description
        self.agents = agents
        self.max_loops = max_loops
        self.output_type = output_type
        self.feedback_director_model_name = feedback_director_model_name
        self.director_name = director_name
        self.director_model_name = director_model_name
        self.verbose = verbose
        self.add_collaboration_prompt = add_collaboration_prompt
        self.planning_director_agent = planning_director_agent
        self.director_feedback_on = director_feedback_on
        self.n_of_workers = n_of_workers
        self.workspace_path = workspace_path
        self.temperature = temperature

        agents = self.initialize_worker_agents()

        self.swarm = HierarchicalSwarm(
            name=name,
            description=description,
            agents=agents,
            max_loops=max_loops,
            output_type=output_type,
            feedback_director_model_name=feedback_director_model_name,
            director_name=director_name,
            director_model_name=director_model_name,
            verbose=verbose,
            add_collaboration_prompt=add_collaboration_prompt,
            planning_director_agent=planning_director_agent,
        )

    def initialize_worker_agents(self):
        for i in range(self.n_of_workers):
            self.agents.append(
                OpenCursorAgent(
                    name=f"Worker Agent {i+1}",
                    description="A worker agent that can complete tasks for the development team",
                    model_name=self.director_model_name,
                    workspace_path=self.workspace_path,
                    temperature=self.temperature,
                )
            )
        return self.agents

    def run(self, task: str):
        return self.swarm.run(task=task)
