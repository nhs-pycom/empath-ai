import unittest
from agents import AgentState
from empathai_agent import Agent
from langchain_core.messages import HumanMessage

class TestAgentChat(unittest.TestCase):
    """Unit test for the Agent class chat functionality."""

    def setUp(self):
        """Set up the initial conditions for the test."""
        # Define a sample persona and scenario for testing
        self.scenario = "A clinical setting where the agent acts as a patient."
        self.persona = "A 45-year-old patient with mild anxiety, here for a routine check-up."

        # Create an instance of the Agent class with the scenario and persona
        self.agent = Agent(scenario=self.scenario, persona=self.persona)

        # Set up the initial agent state with a simple message from the user
        self.initial_message = HumanMessage(content="Hello, I would like to know more about your symptoms.")
        self.message_history = []  # Start with an empty message history

        # Create the agent state object
        self.agent_state = AgentState(
            message=self.initial_message,
            message_history=self.message_history,
            world_state={}
        )

    def test_chat_response(self):
        """Test that the chat method returns a valid response."""
        # Invoke the chat method to get a response from the agent
        response = self.agent.chat(self.agent_state)

        # Print the response to see the output (for debugging)
        print("Agent Response:", response)

        # Check that the response is not None (you can add more specific assertions as needed)
        self.assertIsNotNone(response, "The agent's response should not be None.")

# Run the test case
if __name__ == '__main__':
    unittest.main()
