from .agents import AgentState, AgentResponse, ConversationalAgent
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage
from langchain.chat_models import AzureChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.pydantic_v1 import BaseModel, Field
import math
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Constants
ENDPOINT_URL = os.getenv("ENDPOINT_URL")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

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
    Consider all elements of the mark scheme below.

    Mark Scheme:
    Section 1. Professional Introduction 
    Introduces self with full name and role; verifies the patient’s name. (1 mark)

    Section 2. Building Rapport and Sharing News 
    Establishes rapport by beginning with an open-ended question. (2 marks)

    Section 3. Understanding the Situation and Delivering News 
    Confirms the patient’s current understanding of the situation. (1 mark)
    Secures patient consent for the consultation and provides a summary of the situation so far. (2 mark)
    Inquires if the patient would like a companion present (e.g., a family member). (1 mark)
    Provides a warning before sharing bad news. (1 mark)
    Effectively communicates bad news and investigation results. (2 mark)
    Uses pauses to allow the patient time to process the information. (2 mark)

    Section 4. Communication Techniques 
    Recognizes and appropriately addresses the patient’s emotions and concerns. (2 mark)
    Determines what the patient wants to know and how much detail they need. (2 mark)
    Offers reassurance without creating false hope. (2 mark)
    Demonstrates empathy both verbally and through body language. (1 mark)
    Avoids using medical jargon. (1 mark)
    Actively listens, picking up on and responding to cues from the patient. (1 mark)

    Section 5. Information Delivery 
    Clearly explains the diagnosis and likely prognosis. (2 mark)
    Discusses the referral process and any further necessary investigations. (2 mark)
    Thoroughly outlines treatment options in detail. (2 mark)
    Encourages the patient to discuss the news with family and offers support for follow-up appointments. (1 mark)
    Offers counseling as appropriate. (1 mark)
    Schedules follow-up and provides relevant informational materials. (2 mark)

    Section 6. Concluding the Consultation 
    Invites the patient to ask questions and provides contact information for future inquiries or emergencies. (2 mark)
    Summarizes the discussion and confirms the patient’s understanding. (2 mark)
    Ensures the patient is ready to leave and is safe to return home. (1 mark)
    Maintains a structured approach, clearly indicating changes in the conversation’s focus. (1 mark)

    Transcript:
    {transcript}

    Your task: Carefully review the transcript so far and assign marks to the clinician, the maximum available marks for each item are shown in brackets.
    Return a score for each of the elements in the mark scheme.

    {format_instructions}
    """

emotions = {
    "😊": (0.8, 0.6),
    "😄": (0.9, 0.9),
    "😌": (0.4, 0.3),
    "😎": (0.2, -0.1),
    "😐": (0.0, 0.0),
    "😑": (-0.25, 0.0),
    "😕": (0.0, 0.1),
    "😌": (0.5, -0.2),
    "😢": (-0.7, -0.4), 
    "😭": (-0.9, 0.25),
    "😟": (-0.8, -0.7),
    "😒": (-0.5, -0.6),
    "😴": (-0.4, -0.8),
    "😣": (-0.8, 0.7),
    "😡": (-0.9, 0.8),
    "😠": (-0.7, 0.6),
    "😨": (-0.3, 0.3),
    "😲": (0.1, 0.8),
    "😱": (-0.6, 0.9),
    "😟": (-0.5, 0.1)
}


SENTIMENT_ANALYSIS_PROMPT = """Objective: Based on the transcript between the patient and their doctor, assess the patient’s emotion using the Emotion Scale, considering the patient’s persona.

    Emotions Scale:
    Valence (-1 = Unpleasant, 1 = Pleasant)
    Arousal (-1 = Low, 1 = High)

    Persona:
    {persona}

    Your task:
    Carefully review the transcript, weighting recent dialog more heavily, and assign a score between -1 and 1 for each item of the emotion scale. 0 represents a neutral score, scores towards the extremes of the scale should be reserved for only the most extreme cases (eg. when the patient is very angry with the doctor or elated with joy). Return a score for each element in the mark scheme.

    Formatting instructions:
    Only return your scores. Return your assessment in JSON format, for example:

        {{Valence: 0.1,
        Arousal: 0.2}}

    Transcript:
    {transcript}
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
        self.chat_model = AzureChatOpenAI(azure_endpoint=ENDPOINT_URL, 
                        api_key=AZURE_OPENAI_API_KEY, 
                        azure_deployment=DEPLOYMENT_NAME, 
                        api_version="2024-05-01-preview")
        # Initialize teacher model for evaluation purposes
        self.teacher_model = AzureChatOpenAI(azure_endpoint=ENDPOINT_URL, 
                        api_key=AZURE_OPENAI_API_KEY, 
                        azure_deployment=DEPLOYMENT_NAME, 
                        api_version="2024-05-01-preview")
        # Initialize sentiment analysis model for analysis purposes
        self.sentiment_analysis_model = AzureChatOpenAI(azure_endpoint=ENDPOINT_URL, 
                        api_key=AZURE_OPENAI_API_KEY, 
                        azure_deployment=DEPLOYMENT_NAME, 
                        api_version="2024-05-01-preview")

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
                {"transcript": [*agent_state.message_history]}
            )
        except OutputParserException as e:
            # Handle parsing errors gracefully by returning an error message
            teacher_feedback = {"error": str(e)}

        return teacher_feedback
    
    def sentiment(self, agent_state: AgentState) -> AgentResponse:
        """
        Determines the patients emotion returning the most relevant emoji to reflect their state

        Args:
            agent_state: The current state of the agent including message history and the latest message.

        Returns:
            AgentResponse: The unicode of representing the patients emotion.
        """

        # Set up the parser with the expected Pydantic model
        parser = JsonOutputParser(pydantic_object=SentimentAnalysis)

        # Create a prompt context for the teacher model with format instructions from the parser
        sentiment_analysis_context = PromptTemplate(
            template=SENTIMENT_ANALYSIS_PROMPT,
            input_variables=["transcript"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        # Chain the teacher context, model, and parser into a processing pipeline
        sentiment_analysis_chain = sentiment_analysis_context | self.sentiment_analysis_model | parser

        try:
            # Invoke the chain with the transcript data from the agent state
            input_dict = sentiment_analysis_chain.invoke(
                {"transcript": [*agent_state.message_history],
                 "persona": self.persona}
            )

            # Function to calculate Euclidean distance
            def euclidean_distance(point1, point2):
                return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

            # Input point
            input_point = (input_dict['Valence'], input_dict['Arousal'])

            # Find the closest emotion
            best_emotion = None
            min_distance = float('inf')

            for emoji, emotion in emotions.items():
                dist = euclidean_distance(input_point, emotion)
                if dist < min_distance:
                    min_distance = dist
                    best_emotion = emoji
            
            return best_emotion

        except OutputParserException as e:
            # Handle parsing errors gracefully by returning an error message
            return {"error": str(e)}

class MarkingRubric(BaseModel):
    """Data structure for marking rubric"""

    # Section 1. Professional Introduction
    introducesSelfWithFullNameAndRole: int = Field(
        default=0, description="Introduces self with full name and role; verifies the patient’s name. (1 mark)"
    )

    # Section 2. Building Rapport and Sharing News
    establishesRapportWithOpenEndedQuestion: int = Field(
        default=0, description="Establishes rapport by beginning with an open-ended question. (2 marks)"
    )

    # Section 3. Understanding the Situation and Delivering News
    confirmsPatientUnderstandingOfSituation: int = Field(
        default=0, description="Confirms the patient’s current understanding of the situation. (1 mark)"
    )
    securesPatientConsentAndProvidesSummary: int = Field(
        default=0, description="Secures patient consent for the consultation and provides a summary of the situation so far. (2 marks)"
    )
    inquiresIfPatientWantsCompanionPresent: int = Field(
        default=0, description="Inquires if the patient would like a companion present (e.g., a family member). (1 mark)"
    )
    providesWarningBeforeSharingBadNews: int = Field(
        default=0, description="Provides a warning before sharing bad news. (1 mark)"
    )
    effectivelyCommunicatesBadNews: int = Field(
        default=0, description="Effectively communicates bad news and investigation results. (2 marks)"
    )
    usesPausesToAllowProcessing: int = Field(
        default=0, description="Uses pauses to allow the patient time to process the information. (2 marks)"
    )

    # Section 4. Communication Techniques
    addressesPatientEmotionsAndConcerns: int = Field(
        default=0, description="Recognizes and appropriately addresses the patient’s emotions and concerns. (2 marks)"
    )
    determinesPatientWantsToKnow: int = Field(
        default=0, description="Determines what the patient wants to know and how much detail they need. (2 marks)"
    )
    offersReassuranceWithoutFalseHope: int = Field(
        default=0, description="Offers reassurance without creating false hope. (2 marks)"
    )
    demonstratesEmpathy: int = Field(
        default=0, description="Demonstrates empathy both verbally and through body language. (1 mark)"
    )
    avoidsMedicalJargon: int = Field(
        default=0, description="Avoids using medical jargon. (1 mark)"
    )
    activelyListensAndRespondsToCues: int = Field(
        default=0, description="Actively listens, picking up on and responding to cues from the patient. (1 mark)"
    )

    # Section 5. Information Delivery
    clearlyExplainsDiagnosisAndPrognosis: int = Field(
        default=0, description="Clearly explains the diagnosis and likely prognosis. (2 marks)"
    )
    discussesReferralAndFurtherInvestigations: int = Field(
        default=0, description="Discusses the referral process and any further necessary investigations. (2 marks)"
    )
    outlinesTreatmentOptions: int = Field(
        default=0, description="Thoroughly outlines treatment options in detail. (2 marks)"
    )
    encouragesDiscussionWithFamily: int = Field(
        default=0, description="Encourages the patient to discuss the news with family and offers support for follow-up appointments. (1 mark)"
    )
    offersCounselingAsAppropriate: int = Field(
        default=0, description="Offers counseling as appropriate. (1 mark)"
    )
    schedulesFollowUpAndProvidesInfo: int = Field(
        default=0, description="Schedules follow-up and provides relevant informational materials. (2 marks)"
    )

    # Section 6. Concluding the Consultation
    invitesQuestionsAndProvidesContact: int = Field(
        default=0, description="Invites the patient to ask questions and provides contact information for future inquiries or emergencies. (2 marks)"
    )
    summarizesDiscussionAndConfirmsUnderstanding: int = Field(
        default=0, description="Summarizes the discussion and confirms the patient’s understanding. (2 marks)"
    )
    ensuresPatientIsReadyAndSafeToLeave: int = Field(
        default=0, description="Ensures the patient is ready to leave and is safe to return home. (1 mark)"
    )
    maintainsStructuredApproach: int = Field(
        default=0, description="Maintains a structured approach, clearly indicating changes in the conversation’s focus. (1 mark)"
    )

class SentimentAnalysis(BaseModel):
    """Data structure for sentiment analysis"""

    # Valence
    valence: float = Field(
        default=0.0, description="Valence"
    )

    # Arousal
    arousal: float = Field(
        default=0.0, description="Arousal"
    )