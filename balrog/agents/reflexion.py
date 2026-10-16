import re
import copy

from balrog.agents.base import BaseAgent


class ReflexionAgent(BaseAgent):
    """An agent that generates actions and plans based on observations with a customizable approach."""

    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        self.experience = []
        self.evaluations = []
        self.actions = []

    def reset(self):
        self.prompt_builder.reset()
        self.experience = []

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

        trajectory = copy.deepcopy(self.prompt_builder.get_prompt()) # contains only 
        self_reflection_input = copy.deepcopy(trajectory)
        action_input = copy.deepcopy(trajectory)
        # last_obs = self.prompt_builder._last_short_term_obs

        evaluator_instructions = """
Evaluate your trajectory so far, focusing on the outcome of your previous action relative to the current observation.
Output your evaluation strictly in the following format:

<|EVALUATION|>YOUR_EVALUATION_TEXT<|END|>
""".strip()
        trajectory[-1].content += "\n\n" + evaluator_instructions 
        evaluation = self.client.generate(trajectory)
        self.evaluations.append(evaluation.completion)
        
        self_reflection_instructions = f"""
Based on the latest evaluation above, reflect on what went right, what went wrong, and how you should adapt your strategy going forward.
Output your reflection strictly in the following format:

<|REFLECTION|>YOUR_REFLECTION_TEXT<|END|>
""".strip()
        self_reflection_input[-1].content += "\n\n" + evaluation.completion + "\n" + self_reflection_instructions 
        self_reflection = self.client.generate(self_reflection_input)
        self.experience.append(self_reflection.completion)
        
        experiences =  "<|EXPERIENCES|>\n" + "\n".join(self.experience) + "\n<|END|>"
        action_instruction = f"""
Review your accumulated past experiences, then choose the single best next action. 
You must output your final answer strictly in the following format:

<|ACTION|>YOUR_CHOSEN_ACTION<|END|>
""".strip()

        action_input[-1].content += "\n\n" + experiences + "\n" + action_instruction
        response = self.client.generate(action_input)
        self.actions.append(response.completion)
        
        return self._extract_action(response)

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
