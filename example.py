import asyncio
import logging
import traceback

from loguru import logger

from open_cursor.main import OpenCursorAgent


# Example usage and testing
async def main():
    """Example usage of the Cursor Agent."""
    # Initialize the agent
    agent = OpenCursorAgent(
        model_name="gpt-4o",
        workspace_path=".",
    )

    # Example task
    task_description = """
    Create a fun game file named game.py in the video_game folder with actual game code using pygame. Do not create an empty file - include real Python game code. The code should be around 100-200 lines
    """

    try:
        # Run the agent
        result = await agent.run(task_description)

        print(result)

    except Exception as e:
        print(f"Error running agent: {str(e)}")
        logger.error(f"Error in main: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
