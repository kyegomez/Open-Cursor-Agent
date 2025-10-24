from open_cursor.agent_swarm import DevelopmentTeam
from dotenv import load_dotenv

load_dotenv()


def main():
    team = DevelopmentTeam(
        name="Development Team",
        description="A team of developers that can complete tasks",
        n_of_workers=3,
        director_model_name="gpt-4.1",
    )
    team.run("Create a transformer model in pytorch in a file called transformer.py")


if __name__ == "__main__":
    main()
