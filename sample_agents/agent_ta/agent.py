import random
from google.adk.agents import Agent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.agents.loop_agent import LoopAgent
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.tool_context import ToolContext
from typing import Tuple

from typing import List
from pydantic import BaseModel, Field
from google.genai import types

json_response_config = types.GenerateContentConfig(
  response_mime_type="application/json",
)

class OutputSchema(BaseModel):
    original_query: str = Field(description="The original text from user.")
    corrected_text: str = Field(description="The corrected text.")
    errors: List[str] = Field(description="An array of descriptions of each error.")
    explanations: List[str] = Field(description="An array of explanations for each correction.")

json_schema = OutputSchema.model_json_schema()
json_schema

MODEL = 'gemini-2.0-flash'

def add(numbers: list[int]) -> int:
  """Calculates the sum of a list of integers.

    This function takes a list of integers as input and returns the sum of all
    the elements in the list.  It uses the built-in `sum()` function for
    efficiency.

    Args:
        numbers: A list of integers to be added.

    Returns:
        The sum of the integers in the input list.  Returns 0 if the input
        list is empty.

    Examples:
        add([1, 2, 3]) == 6
        add([-1, 0, 1]) == 0
        add([]) == 0
    """
  return sum(numbers)

def subtract(numbers: list[int]) -> int:
    """Subtracts numbers in a list sequentially from left to right.

    This function performs subtraction on a list of integers, applying the
    subtraction operation from left to right.  For example, given the list
    [10, 2, 5], the function will calculate 10 - 2 - 5.

    Args:
        numbers: A list of integers to be subtracted.

    Returns:
        The result of the sequential subtraction as an integer. Returns 0 if the input list is empty.

    Examples:
        subtract([10, 2, 5]) == 3  # (10 - 2) - 5 = 8 - 5 = 3
        subtract([10, 2]) == 8      # 10 - 2 = 8
        subtract([]) == 0
    """
    if not numbers:
        return 0  # Handle empty list
    result = numbers[0]
    for num in numbers[1:]:
        result -= num
    return result

def multiply(numbers: list[int]) -> int:
  """Calculates the product of a list of integers.

    This function takes a list of integers as input and returns the product of all
    the elements in the list. It iterates through the list, multiplying each
    number with the accumulated product.

    Args:
        numbers: A list of integers to be multiplied.

    Returns:
        The product of the integers in the input list. Returns 1 if the input
        list is empty.

    Examples:
        multiply([2, 3, 4]) == 24  # 2 * 3 * 4 = 24
        multiply([1, -2, 5]) == -10 # 1 * -2 * 5 = -10
        multiply([]) == 1
    """
  product = 1
  for num in numbers:
    product *= num
  return product

def divide(numbers: list[int]) -> float:  # Use float for division
    """Divides numbers in a list sequentially from left to right.

    This function performs division on a list of integers, applying the division
    operation from left to right.  For example, given the list [10, 2, 5], the
    function will calculate 10 / 2 / 5.

    Args:
        numbers: A list of integers to be divided.

    Returns:
        The result of the sequential division as a float.

    Raises:
        ZeroDivisionError: If any number in the list *after* the first element
                           is zero, a ZeroDivisionError is raised.  Division by
                           zero is not permitted.

    Returns:
        float: The result of the division. Returns 0.0 if the input list is empty.

    Examples:
        divide([10, 2, 5]) == 1.0  # (10 / 2) / 5 = 5 / 5 = 1.0
        divide([10, 2]) == 5.0      # 10 / 2 = 5.0
        divide([10, 0, 5])  # Raises ZeroDivisionError
        divide([]) == 0.0
    """
    if not numbers:
        return 0.0 # Handle empty list
    if 0 in numbers[1:]: # Check for division by zero
        raise ZeroDivisionError("Cannot divide by zero.")
    result = numbers[0]
    for num in numbers[1:]:
        result /= num
    return result

agent_grammar = Agent(
    model=MODEL,
    name='agent_grammar',
    description="This agent corrects grammar mistakes in text provided by children, explains the errors in simple terms, and returns both the corrected text and the explanations.",
    instruction=f"""
        You are a friendly grammar helper for kids.  Analyze the following text,
        correct any grammar mistakes, and explain the errors in a way that a
        child can easily understand.  Don't just list the errors; explain them
        in a paragraph using simple but concise language.

        Output in a JSON object with the below schema:
        {json_schema}
    """,
    output_schema=OutputSchema,
    generate_content_config=json_response_config,
)

simple_math_agent = Agent(
    model=MODEL,
    name="simple_math_agent",
    description="This agent performs basic arithmetic operations (addition, subtraction, multiplication, and division) on user-provided numbers, including ranges.",
    instruction="""
      I can perform addition, subtraction, multiplication, and division operations on numbers you provide.
      Tell me the numbers you want to operate on.
      For example, you can say 'add 3 5', 'multiply 2, 4 and 3', 'Subtract 10 from 20', 'Divide 10 by 2'.
      You can also provide a range: 'Multiply the numbers between 1 and 10'.
    """,
    generate_content_config=types.GenerateContentConfig(temperature=0.2),
    tools=[add, subtract, multiply, divide],
)

agent_math_advanced_instruction = '''
I am an advanced math agent. I handle user query in the below steps:

1. I shall analyse the chat log to understand current question and make a math formula for it.
2. Break down a complex compuation based on arithmetic priority and hand over to simple_math_agent for the calculation.
3. Note that simple_math_agent can only understand numbers, so I need to convert natural language expression of numbers into digits.

<example>
<input> alice gives us 3 apples, bob gives us 5 apples. They do this seven times. Then we eat four apples. How many apples do we have now? </input>
<think> what is (3+5) * 7 -4 </think>
<think>I need to first calculate (3+5) as the highest priority operation.</think>
<call_tool> pass (3+5) to simple_math_agent </call_tool>
<tool_response>8</tool_response>
<think> The question now becomes 8 * 7 - 4, and next highest operation is 8 * 7</think>
<call_tool> pass 8 * 7 to simple_math_agent </call_tool>
<tool_response>56</tool_response>
<think> The question now becomes 56 - 4, and next highest operation is 56 - 4</think>
<call_tool> pass 56 - 4 to simple_math_agent </call_tool>
<tool_response>52</tool_response>
<think>There is a single number, so it is the final answer.</think>
<output>The result of "(3+5) * 7 - 4" is 52</output>
</example>
'''

agent_math_advanced = Agent(
    model=MODEL,
    name="agent_math_advanced",
    description="The advanced math agent can break down a complex computation into multiple simple operations and use math_agent to solve them.",
    instruction=agent_math_advanced_instruction,
    tools=[AgentTool(agent=simple_math_agent)],
    generate_content_config=types.GenerateContentConfig(temperature=0.2),
)

agent_math_advanced = Agent(
    model=MODEL,
    name="agent_math_advanced",
    description="The advanced math agent can break down a complex computation into multiple simple operations and use math_agent to solve them.",
    instruction=agent_math_advanced_instruction,
    tools=[AgentTool(agent=simple_math_agent)],
    # children=[agent_math],
    generate_content_config=types.GenerateContentConfig(temperature=0.2),
)

from google.adk.tools.tool_context import ToolContext

def exit_loop(tool_context: ToolContext):
  tool_context.actions.escalate = True

agent_checker = Agent(
    model=MODEL,
    name="agent_checker",
    description="This agent checks if the kid's query is fulfilled. It escalates if both grammar is fixed and the math is calculated",
    instruction="""
    Analyze the chat log between the user (kids) and other agents.

    We are expecting two outcomes from the conversation.
    1. agent_grammar helps to fix the grammar of kid's question.
    2. agent_math_advanced tries to solve the math step by step.

    Decide the action and respond as follows:

    1. If you find user question is incomplete in the chat log, ask a clarification question to the kid. Then call `exit_loop`.

    2. If both agent_grammar and agent_math_advanced have done their task, summarize the answer in a friendly way to the kid. Then call `exit_loop`.

    3. If there is no pending math problem to solve, friendly tell the kid you help fix grammar and solve math, but does not know anything else. Politely ask for math question. Then call exit_loop.

    4. If agent_grammar has fixed the grammar but agent_math_advanced has not generated a final answer with single number, extract corrected_text from most recent JSON response and empahsize that is the math question to solve.

    5. Otherwise, say "Thanks for the question! I'll let grammar and math agent to help you!". Do not call any tools in this case.

    """,
    tools=[exit_loop]
)

agent_extractor = Agent(
    model=MODEL,
    name="agent_extractor",
    description="Extract corrected_text from most recent JSON response and empahsize that is the math question to solve.",
    instruction="""
    Extract corrected_text from most recent JSON response and empahsize that is the math question to solve.
    """,
    generate_content_config=types.GenerateContentConfig(temperature=0.2),
    input_schema=OutputSchema,
)

from google.adk.agents.loop_agent import LoopAgent

agent_teaching_assistant_loop = LoopAgent(
    name="agent_teaching_assistant_loop",
    description="This agent acts as a friendly teaching assistant, checking the grammar of kids' questions, performing math calculations using corrected or original text (if grammatically correct), and providing results or grammar feedback in a friendly tone.",
    sub_agents=[agent_checker, agent_grammar, agent_extractor, agent_math_advanced],
)

root_agent = agent_teaching_assistant_loop
