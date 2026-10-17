import copy
import re

from balrog.agents.base import BaseAgent

from collections import deque

SR_PROMPT = """ 
Analyze why the previous episode failed or succeeded based on the trajectory and reward above. 
Write a short, direct rule or adjustment for the next trial. Focus on what specific movement, item interaction, or navigation choice went wrong.

Output your reflection strictly in the following format:

<|REFLECTION|>YOUR_SHORT_REFLECTION_TEXT<|END|>

No other text or explanation.
""".strip()

NAIVE_INSTRUCTION = """
Consider your past lessons below and the current observation to choose your next action.

Past Lessons:
{experiences}

You must choose a valid action and output it strictly in the following format:
<|ACTION|>YOUR_CHOSEN_ACTION<|END|>

Do not output any explanation or extra text.
""".strip()

NAIVE_INSTRUCTION_B = """
Consider your past lessons below and the current observation to choose your next action.

Past Lessons:
{experiences}

Then, you must choose exactly one of the listed actions and output it strictly in the following format:

<|ACTION|>CHOSEN_ACTION<|END|>

Replace CHOSEN_ACTION with the chosen action as plain text.

For example: 

<|ACTION|>some_action<|END|>
"""

class RobustNaiveReflexion(BaseAgent):
    """An agent that generates actions based on observations without complex reasoning."""

    def __init__(self, client_factory, prompt_builder, config):
        """Initialize the NaiveAgent with a client and prompt builder."""
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        self.exp_buffer = deque(maxlen=config.agent.max_experiences)
        self.last_ep_reward = None

    def update_reward(self, reward):
        self.last_ep_reward = reward

    def self_reflect(self):
        messages = self.prompt_builder.get_prompt()
        messages[-1].content += "\n\n" + f"Last episode reward: {self.last_ep_reward}" + SR_PROMPT
        response = self.client.generate(messages)

        completion_text = response.completion
        match = re.search(r"<\|REFLECTION\|>(.*?)<\|END\|>", completion_text, re.DOTALL)
        if match:
            reflection_text = match.group(1).strip()
        else:
            reflection_text = completion_text.strip()

        self.exp_buffer.append(reflection_text)


    def act(self, obs, prev_action=None):
        """Generate the next action based on the observation and previous action.

        Args:
            obs (dict): The current observation in the environment.
            prev_action (str, optional): The previous action taken.

        Returns:
            str: The selected action from the LLM response.
        """
        if prev_action:
            self.prompt_builder.update_action(prev_action)

        self.prompt_builder.update_observation(obs)

        messages = self.prompt_builder.get_prompt()
        if len(self.exp_buffer) > 0:
            experiences = "\n".join(self.exp_buffer)
        else:
            experiences = "None."

        # Updated instructions to require a very strict output format + reflexion
        instruction = NAIVE_INSTRUCTION_B.format(experiences=experiences)

        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + experiences + "\n" + instruction

        response = self.client.generate(messages)
        final_answer = self._extract_final_answer(response)
        return final_answer

    def _extract_final_answer(self, answer):
        """Extract the action from the completion by looking for <|ACTION|> and <|END|> tags.

        Args:
            answer (LLMResponse): The response from the LLM.

        Returns:
            LLMResponse: The sanitized response containing just the extracted action.
        """
        completion_text = answer.completion
        # Use a regex to find the text inside <|ACTION|> and <|END|>
        match = re.search(r"<\|ACTION\|>(.*?)<\|END\|>", completion_text, re.DOTALL)
        if match:
            extracted_action = match.group(1).strip()
        else:
            # If no match is found, fallback to the original completion or handle as needed
            extracted_action = completion_text.strip()

        final_answer = copy.deepcopy(answer)
        final_answer = final_answer._replace(completion=extracted_action)

        return final_answer
