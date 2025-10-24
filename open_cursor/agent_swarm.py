from swarms import HierarchicalSwarm
from open_cursor.main import OpenCursorAgent
from typing import List, Union, Any, Callable, Optional
from swarms.structs.agent import Agent

class DevelopmentTeam:
    def __init__(
        self,
        name: str,
        description: str,
        agents: List[OpenCursorAgent],
        max_loops: int = 1,
        output_type: str = "dict-all-except-first",
        feedback_director_model_name: str = "gpt-4.1",
        director_name: str = "Director",
        director_model_name: str = "gpt-4.1",
        verbose: bool = False,
        add_collaboration_prompt: bool = True,
        planning_director_agent: Optional[
            Union[Agent, Callable, Any]
        ] = None,
        director_feedback_on: bool = True,
        interactive: bool = False,
        director_reasoning_model_name: str = "o3-mini",
        director_reasoning_enabled: bool = False,
        multi_agent_prompt_improvements: bool = False,
    ):
        self.name = name
        self.description = description
        self.agents = agents
        self.swarm = HierarchicalSwarm(
            name = name,
            description = description,
            agents = agents,
            max_loops = max_loops,
            output_type = output_type,
            feedback_director_model_name = feedback_director_model_name,
            director_name = director_name,
            director_model_name = director_model_name,
            verbose = verbose,
            add_collaboration_prompt = add_collaboration_prompt,
            planning_director_agent = planning_director_agent,
        )

    def run(self, task: str):
        return self.swarm.run(task=task)