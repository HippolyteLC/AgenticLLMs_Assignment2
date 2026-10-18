import re

from balrog.agents.base import BaseAgent


class CustomAgent(BaseAgent):
    """An agent that generates actions and plans based on observations with a customizable approach."""
    
    DIRECTIONS = ["NORTH", "EAST", "SOUTH", "WEST"]

    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        self.plan = None
        self.memory = None
        self.action_history = []
        self.direction_idx = 0


    def reset(self):
        #We eset the agent's internal state between episodes

        super().reset()
        self.plan = None
        self.action_history = []
        self.facing_idx = 0

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
            self.action_history.append(prev_action)
                            
            act_lower = prev_action.lower()    #  update facing direction on turns
            if "turn left" in act_lower:
                self.facing_idx = (self.facing_idx - 1) % 4
            elif "turn right" in act_lower:

                self.facing_idx = (self.facing_idx + 1) % 4
        self.prompt_builder.update_observation(obs)
        
        loop_warning = ""         # we check if last 3 actions were identical
        if len(self.action_history) >= 3 and self.action_history[-1] == self.action_history[-2] == self.action_history[-3]:
            loop_warning = f"\nWARNING: You repeated '{self.action_history[-1]}' you had 3 times a similar action! Pick a different action to avoid getting stuck.\n"
        

        compass_text = f"Compass: You are currently facing {self.DIRECTIONS[self.facing_idx]}.\n"  # compass helper prompt
        # Plan & Memory text
        plan_text = f"Current Plan & Memory:\n{self.plan}\n" if self.plan else "You have no plan yet.\n"
        planning_instructions = f"""
{compass_text}{loop_warning}

Review the current plan above if present. Decide whether to continue with it or make changes.
If you make changes, provide the updated plan. Then, provide the next action to take.
You must output an action at every step.
Format your answer in the following way:
PLAN: <your updated plan if changed, or "No changes to the plan." if the current plan is good>
ACTION: <your next action>
        """.strip()

        messages = self.prompt_builder.get_prompt()
        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + plan_text + "\n" + planning_instructions

        response = self.client.generate(messages)

        # Extract the plan and action from the LLM's response
        plan, action = self._extract_plan_and_action(response.completion)

        # Update the internal plan if it has changed
        if plan != "No changes to the plan.":
            self.plan = plan

        # Save the plan in the response.reasoning field and the action in response.completion
        response = response._replace(reasoning=plan, completion=action)
        return response

    def _extract_plan_and_action(self, response_text):
        """Extract the plan and action from the LLM's response text.

        Args:
            response_text (str): The response text from the LLM.

        Returns:
            tuple: A tuple containing the extracted plan and action.
        """
        plan_match = re.search(r"PLAN:\s*(.*?)(?=\nACTION:|\Z)", response_text, re.IGNORECASE | re.DOTALL)
        action_match = re.search(r"ACTION:\s*(.*)", response_text, re.IGNORECASE | re.DOTALL)

        plan = plan_match.group(1).strip() if plan_match else "No changes to the plan."
        action = action_match.group(1).strip() if action_match else None

        return plan, action
