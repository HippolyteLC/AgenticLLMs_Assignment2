import re
import copy

from balrog.agents.base import BaseAgent

from collections import deque

EVALUATION_A = """
Evaluate your trajectory so far, focusing on the outcome of your previous action relative to the current observation.
Output your evaluation strictly in the following format:

<|EVALUATION|>YOUR_EVALUATION_TEXT<|END|>
"""
EVALUATION_B = """
Consider the observations and the actions you have performed so far. 
Each observation is a consequence of the action that precedes it. 
When evaluating, consider the goal of this game and the game description provided earlier.  
Evaluate the previous action relative to the current observation. 
You must output a binary reward signal: success/fail. 

Example output: success
Example output: fail
"""

SELF_REFLECTION_A= """ 
Based on your previous action relative to the current observation, and the latest evaluation above, reflect on what went right, what went wrong, and how you should adapt your strategy going forward.
Output your reflection strictly in the following format:

<|REFLECTION|>YOUR_REFLECTION_TEXT<|END|>
"""
SELF_REFLECTION_B = """
Given your previous action and the current observation, and the evaluation above of your previous action (success/fail),
reflect on the following: 
1. Why was my action a success or fail?
2. What actions help me achieve the goal of this game, as described in the game description earlier?
3. How can I provide feedback to myself for future actions so that I maximize the likelihood of 
selecting an action that will help me achieve the goal of the game? 

Make sure to provide a concise and compact reflection, which provides clear verbal cues for future decision making and reasoning. 
"""

ACTION_A = """
Review your accumulated past experiences and consider your previous action relative to the current observation, 
then choose the single best next action. 
You must output your final answer strictly in the following format:

<|ACTION|>YOUR_CHOSEN_ACTION<|END|>
"""

ACTION_B = """
Review your past experiences. Consider your previous action and the current observation. 
Then, decide on the next best action to achieve the goal of this game. 
You must choose exactly one of the listed actions and output it strictly in the following format:

<|ACTION|>YOUR_CHOSEN_ACTION<|END|>

Replace YOUR_CHOSEN_ACTION with the chosen action. Output no other text, explanation, or reasoning.
"""


class ReflexionAgent(BaseAgent):
    """An agent that generates actions and plans based on observations with a customizable approach."""

    def __init__(self, client_factory, prompt_builder, config):
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        self.max_experiences = config.agent.max_experiences
        self.experience = deque(maxlen=self.max_experiences)
        self.evaluations = []
        self.actions = []
        
    def reset(self):
        self.prompt_builder.reset()
        self.experience = deque(maxlen=self.max_experiences)
        self.evaluations=[]
        self.actions=[]

    def check_tokens(self, text: str, begin_token: str, end_token: str="<|END|>") -> str:
        if begin_token not in text:
            text = begin_token + text

        if end_token not in text:
            text = text + end_token

        return text

    def act(self, obs, prev_action=None):
        """Generate the next action and update the plan based on the current observation.

        Args:
            obs (dict): The current observation in the environment.
            prev_action (str, optional): The previous action taken.

        Returns:
            LLMResponse: The response containing the updated plan and selected action.
        """
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)

        # user is the environment, assistant is the llm reasoning, self reflection, evaluation

        trajectory = copy.deepcopy(self.prompt_builder.get_prompt()) 
        self_reflection_input = copy.deepcopy(trajectory)
        action_input = copy.deepcopy(trajectory)
        # last_obs = self.prompt_builder._last_short_term_obs
        evaluator_instructions = """

        """
        evaluator_instructions = EVALUATION_A.strip()
        trajectory[-1].content += "\n\n" + evaluator_instructions 
        evaluation = self.client.generate(trajectory).completion
        self.evaluations.append(evaluation)
        
        self_reflection_instructions = SELF_REFLECTION_B.strip()
        self_reflection_input[-1].content += "\n\n" + evaluation + "\n" + self_reflection_instructions 
        self_reflection = self.client.generate(self_reflection_input).completion
        self.experience.append(self_reflection)
        
        experiences =  "Here are your past experiences. The most recent is at the end: " + "\n".join(self.experience)
        
        action_instruction = ACTION_B.strip()
        action_input[-1].content += "\n\n" + experiences + "\n" + action_instruction
        response = self.client.generate(action_input)
        action = self.check_tokens(response.completion, "<|ACTION|>")
        self.actions.append(action)
        response._replace(completion=action)
        final_response = self._extract_action(response)
        if self.config.agent.verbose:
            print("Current action:\n",final_response.completion)

        return final_response

    def _extract_action(self, response_text):
        """Extract the plan and action from the LLM's response text.

        Args:
            response_text (str): The response text from the LLM.

        Returns:
            tuple: A tuple containing the extracted plan and action.
        """
        final_answer = copy.deepcopy(response_text)
        completion_text = final_answer.completion
        match = re.search(r"<\|ACTION\|>(.*?)<\|END\|>", completion_text, re.DOTALL)
        if match:
            extracted_action = match.group(1).strip()
        else:
            # Fallback to the entire completion if not matched
            extracted_action = "Failed to obtain a valid action from the reasoning."

        final_answer = final_answer._replace(completion=extracted_action)

        return final_answer
