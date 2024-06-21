import os
import requests
from dotenv import load_dotenv
import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory
)

# Load environment variables from .env file
load_dotenv()
GCLOUD_PROJECT_ID = os.getenv('GCLOUD_PROJECT_ID')
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')

# Initialize Vertex AI with the specified project and location
vertexai.init(project=GCLOUD_PROJECT_ID, location="us-central1")

# Define model identifiers
TEXT_GEN_MODEL = "gemini-1.5-flash-preview-0514"
EMOTION_CLASSIFIER_URL = "https://api-inference.huggingface.co/models/ayoubkirouane/BERT-Emotions-Classifier"

# Setup headers for Hugging Face API authentication
HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

# Configuration for text generation
generation_config = GenerationConfig(
    temperature=0.9,
    top_p=1.0,
    top_k=32,
    candidate_count=1,
    max_output_tokens=8192,
)

# Define safety settings for content generation
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
}

def get_response(model: GenerativeModel, prompt: str) -> str:
    """
    Generate a response from the model based on the given prompt.
    
    Args:
        model (GenerativeModel): The generative model to use.
        prompt (str): The prompt to generate a response for.
    
    Returns:
        str: The generated response text.
    """
    response = model.generate_content(
        prompt,
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    return response.text

def make_model(system_instructions: str) -> GenerativeModel:
    """
    Create and return a generative model with the given system instructions.
    
    Args:
        system_instructions (str): System instructions for the generative model.
    
    Returns:
        GenerativeModel: The configured generative model.
    """
    return GenerativeModel(
        TEXT_GEN_MODEL,
        system_instruction=system_instructions
    )

def evaluate_patient_emotions(payload: dict) -> dict:
    """
    Evaluate patient emotions using the Hugging Face emotion classifier model.
    
    Args:
        payload (dict): The payload containing data to classify.
    
    Returns:
        dict: The classification results from the emotion classifier.
    """
    response = requests.post(EMOTION_CLASSIFIER_URL, headers=HEADERS, json=payload)
    response.raise_for_status()
    return response.json()

def format_custom_gen_prompt(prompt_template: dict, requested_scenario: str) -> str:
    """
    Extract and return the prompt string based on the provided data and requested scenario.
    
    Args:
        prompt_template (dict): A dictionary containing the scenario generation data.
        requested_scenario (str): The desired scenario as a string.
    
    Returns:
        str: The formatted prompt based on the requested scenario.
    """
    # Extract relevant information
    context = prompt_template["context"]
    examples = prompt_template["examples"]
    
    # Build the prompt string
    prompt_string = f"Context: {context}\n\nExamples:\n"
    for example in examples:
        prompt_string += f"input: {example['inputs'][0]}\n"
        prompt_string += f"output: {example['outputs'][0]}\n"
        prompt_string += "\n"
    prompt_string += f"input: {requested_scenario} \n"
    prompt_string += f"output: "

    return prompt_string

def clean_markdown(raw_markdown: str) -> str:
    """
    Cleans a raw markdown string by removing triple backticks and the language indicator.
    
    Args:
        raw_markdown (str): The raw markdown string to be cleaned. 
                            For example, a string containing JSON data wrapped in markdown code blocks.
    
    Returns:
        str: The cleaned string with markdown formatting removed.
    """
    # Remove the triple backticks and the language indicator
    cleaned_string = raw_markdown.replace('```json\n', '').replace('\n```', '').strip()
    
    return cleaned_string