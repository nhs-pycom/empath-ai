# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines a base class for all conversational agents."""

from abc import ABCMeta, abstractmethod  # Used for creating abstract base classes
from dataclasses import dataclass  # Provides a decorator and functions for creating data classes
from typing import Any, List, Tuple  # Used for type annotations
from langchain_core.messages import BaseMessage  # Assuming this is a custom message class from langchain_core

# Type alias for a list of messages that represent a conversation history
MessageHistory = List[BaseMessage]

# Type alias for the state of the world, which can be any dictionary
WorldState = dict[str, Any]

@dataclass
class AgentState:
    """
    Holds the internal state of a conversational agent, including:
        - The latest message
        - The history of messages in the conversation
        - The current state of the world/context
    """
    message: str
    message_history: MessageHistory
    world_state: WorldState

# Type alias for the agent's response, which includes a response message and the updated agent state
AgentResponse = Tuple[str, AgentState]

class ConversationalAgent(metaclass=ABCMeta):
    """Abstract base class for creating conversational AI agents."""
    
    def __init__(self) -> None:
        # Initialises the agent's state, which will be set later
        self._state: AgentState = None

    @abstractmethod
    def chat(self, agent_state: AgentState) -> AgentResponse:
        """
        Processes a single message exchange with the agent.
        
        Args:
            agent_state: The current state of the agent, including the incoming message.
        
        Returns:
            A tuple containing the agent's response message and the updated state.
        """
        pass  # This method must be implemented by subclasses

    def get_system_prompt(self) -> str:
        """
        Provides the main prompt used by the agent during conversation.
        This method can be overridden if needed, particularly when using interactive tools
        in related scripts.
        
        Returns:
            A string representing the system's prompt, or None if not needed.
        """
        return None
