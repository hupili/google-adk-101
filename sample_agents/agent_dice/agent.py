import random
from google.adk.agents import Agent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.agents.loop_agent import LoopAgent
from google.adk.tools.tool_context import ToolContext
from typing import Tuple

MODEL = "gemini-2.0-flash-001"

def exit_loop(tool_context: ToolContext):
  tool_context.actions.escalate = True

def roll_dice() -> int:
  """Roll a dice. Return a number between 1 and 6 (inclusive).

  Returns:
      int: A random integer between 1 and 6 (inclusive).
  """
  return random.randint(1, 6)

def update_results(player1: int, player2: int, tool_context: ToolContext) -> str:
  '''Updates the current round of dice rolling results and returns the cumulative results.

  This function takes the results of a single round of dice rolling for two players,
  updates the cumulative scores for each player, and returns the updated cumulative
  scores. It also clears the previous round's results.

  Args:
      player1: The result of player 1's dice roll for the current round.
      player2: The result of player 2's dice roll for the current round.
      tool_context: The tool context object containing the current state.

  Returns:
      A string containing the updated cumulative scores for player 1 and player 2.
  '''
  state = tool_context.state
  if 'play1_total' not in state:
    state['play1_total'] = 0
  state['play1_total'] += player1
  if 'play2_total' not in state:
    state['play2_total'] = 0
  state['play2_total'] += player2
  # Clear the previous round
  state['player_1_result'] = ''
  state['player_2_result'] = ''
  return f"Cumulative points so far. player 1: {state['play1_total']}, player 2: {state['play2_total']}"

player1 = Agent(
    model=MODEL,
    name='player_1',
    description='Play 1',
    instruction='''
    Roll a dice with tool roll_dice. Tell the result in a funny and creative way.
    Do not look at previous rounds result.
    Do not look at at the other player's result.
    Do not respond to the user.
    Only roll your own dice and tell everyone the result.
    ''',
    generate_content_config=types.GenerateContentConfig(temperature=0.2),
    tools=[roll_dice],
    output_key='player_1_result',
)

player2 = Agent(
    model=MODEL,
    name='player_2',
    description='Play 2',
    instruction='''
    Roll a dice with tool roll_dice. Tell the result in a funny and creative way.
    Do not look at previous rounds result.
    Do not look at at the other player's result.
    Do not respond to the user.
    Only roll your own dice and tell everyone the result.
    ''',
    generate_content_config=types.GenerateContentConfig(temperature=0.2),
    tools=[roll_dice],
    output_key='player_2_result',
)

judge = Agent(
    model=MODEL,
    name='judge',
    description='Judge',
    instruction='''Analyze both players result and make a short summary of this round.
    Only say the numeric value of player 1 and player 2 result.

    Use the tool `update_results` to add player 1 and player 2 current round results to our scoreboard,
    and get their cumulative results returned by the tool.

    If any player's cumulative result is greater than 15, call `exit_loop` to end the game.
    Otherwise, let the game continue.

    Output in the format:
    - First state both player's current round result
    - Second state both player's cumulative result
    - Third state which player is leading

    Following is your input:

    Play 1 result: {player_1_result}
    Play 2 result: {player_2_result}
    ''',
    tools=[update_results, exit_loop],
)

dice_game = LoopAgent(
    name="dice_game",
    description="A dice game that player 1, player 2 and judge take turns to act.",
    sub_agents=[player1, player2, judge],
)
