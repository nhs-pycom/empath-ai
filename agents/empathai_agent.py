from .agents import AgentState, AgentResponse, ConversationalAgent
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage
from langchain_google_vertexai import ChatVertexAI, VertexAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.pydantic_v1 import BaseModel, Field

# Template for setting up the agent's persona in conversations and the teacher model used for evaluating student performance.
PERSONA_SYSTEM_PROMPT = """
    Embody this patient:
    {persona}

    The patient should be portrayed in a realistic and relatable manner, avoiding overly dramatic or sentimental language.
    Only return what the patient says and do not include anything else.
    """

TEACHER_PROMPT = """
    You are an examiner for clinicians on a communication assessment for breaking bad news.

    Objective:
    Use the mark scheme below to assess the candidates performance on the following transcript.

    Response:
    # To be decided

    Mark Scheme:
    # To be decided

    Transcript:
    {transcript}

    Your task: Carefully review the transcript so far and provide feedback to the clinician.
    Return a score for each of the elements in the mark scheme.

    {format_instructions}
    """

class Agent(ConversationalAgent):
    """
    A conversational agent class that uses AI models to embody a patient persona and evaluate
    clinician performance based on given transcripts.

    Attributes:
        chat_model: The chat model used for generating responses as the patient persona.
        teacher_model: The model used for evaluating clinician performance.
        scenario: The scenario in which the agent operates.
        persona: The persona details used to guide the agent's responses.
        system_context: The prompt context for the persona system.
    """
    
    def __init__(self, scenario, persona):
        """
        Initializes the Agent with a scenario and persona.

        Args:
            scenario: A description or object representing the scenario.
            persona: A string describing the patient persona that the agent should embody.
        """
        # Initialize chat model for persona responses
        self.chat_model = ChatVertexAI(model="gemini-1.5-flash")
        # Initialize teacher model for evaluation purposes
        self.teacher_model = VertexAI(model_name="gemini-1.5-flash", temperature=0)

        # Store scenario and persona attributes
        self.scenario = scenario
        self.persona = persona

        # Set up the system prompt context for the persona
        self.system_context = PromptTemplate.from_template(
            template=PERSONA_SYSTEM_PROMPT
        ).format(persona=self.persona)

    def get_system_prompt(self) -> str:
        """
        Returns the formatted system prompt for the agent based on the persona.

        Returns:
            str: The system prompt with the persona information embedded.
        """
        return self.system_context
    
    def chat(self, agent_state: AgentState) -> AgentResponse:
        """
        Handles a chat exchange with the agent using the current state.

        Args:
            agent_state: The current state of the agent including message history and the latest message.

        Returns:
            AgentResponse: The response from the chat model as the agent's persona.
        """
        # Compile the chat messages, including the system prompt, message history, and the current message.
        chat_messages = [
            SystemMessage(content=self.system_context),
            *agent_state.message_history,
            agent_state.message,
        ]

        # Invoke the chat model to get the next response
        chat_response = self.chat_model.invoke(chat_messages)

        return chat_response

    def evaluate(self, agent_state: AgentState) -> AgentResponse:
        """
        Evaluates the clinician's performance based on a transcript using the teacher model.

        Args:
            agent_state: The current state of the agent including message history and the latest message.

        Returns:
            AgentResponse: The evaluation feedback from the teacher model, structured by the JSON output parser.
        """
        # Set up the parser with the expected Pydantic model
        parser = JsonOutputParser(pydantic_object=MarkingRubric)
        
        # Create a prompt context for the teacher model with format instructions from the parser
        teacher_context = PromptTemplate(
            template=TEACHER_PROMPT,
            input_variables=["transcript"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        # Chain the teacher context, model, and parser into a processing pipeline
        teacher_chain = teacher_context | self.teacher_model | parser
        
        try:
            # Invoke the chain with the transcript data from the agent state
            teacher_feedback = teacher_chain.invoke(
                {"transcript": [*agent_state.message_history, agent_state.message]}
            )
        except OutputParserException as e:
            # Handle parsing errors gracefully by returning an error message
            teacher_feedback = {"error": str(e)}

        return teacher_feedback
    
class MarkingRubric(BaseModel):
    # To be decided
    pass