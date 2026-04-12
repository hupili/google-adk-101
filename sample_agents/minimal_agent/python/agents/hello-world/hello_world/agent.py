from google.adk.agents import Agent

from .tools import hello_world

# Using explicitly synchronous tools
hello_agent = Agent(
    name="hello_agent",
    description="A simple hello world agent",
    instruction="Greet the user with the provided name.",
    tools=[hello_world],
)

root_agent = hello_agent
