import re
import copy

from balrog.agents.base import BaseAgent


class CustomAgent(BaseAgent):
    """An agent that generates actions and plans based on observations with a customizable approach."""

    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
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
        """
        trajectory[-1].content += "\n\n" + evaluator_instructions 
        evaluation = self.client.generate(trajectory)
        
        self_reflection_instructions = f"""
        {evaluation}
        """
        self_reflection_input[-1].content += "\n\n" + self_reflection_instructions 
        self_reflection = self.client.generate(self_reflection_input)
        self.experience.append(self_reflection.completion)
        
        experiences = "\n".join(self.experience)
        action_instruction = f"""
        {experiences}
        """

        action_input[-1].content += "\n\n" + action_instruction
        response = self.client.generate(action_input)
        
        return response._extract_action(response.completion)

    def _extract_action(self, response_text):
        """Extract the plan and action from the LLM's response text.

        Args:
            response_text (str): The response text from the LLM.

        Returns:
            tuple: A tuple containing the extracted plan and action.
        """
        action_match = re.search(r"ACTION:\s*(.*)", response_text, re.IGNORECASE | re.DOTALL)

        action = action_match.group(1).strip() if action_match else None

        return action
