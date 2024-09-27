from flask import Flask, render_template, request, jsonify
import base64
from agents.empathai_agent import Agent, AgentState
from langchain_core.messages import HumanMessage, AIMessage
import json
import google.cloud.texttospeech_v1 as texttospeech

app = Flask(__name__)
tts_client = texttospeech.TextToSpeechClient()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/rubric')
def rubric():
    return render_template('rubric.html')

@app.route('/chat', methods=['POST'])
def chat():
    """
    Handles chat requests from the client.

    Receives a POST request with a JSON payload containing the user's
    message, message history, scenario, and persona details. It uses the EmpathAI Agent 
    to generate a response based on the given input and, if the mode is set to 'voice-to-voice', 
    synthesises the response into speech using Google's Text-to-Speech API.

    Returns:
        - A JSON response containing the agent's textual response.
        - If mode is 'voice-to-voice', also includes a base64-encoded MP3 audio of the response.

    Raises:
        ValueError: If the request data is malformed or incomplete.
    """
    # Message History
    q = request.json.get("message")
    message_hist_json = request.json.get("history")
    message_history = build_message_history(message_hist_json)

    # World State
    world_state_json = request.args.get("world_state")
    world_state = json.loads(world_state_json) if world_state_json else None

    # Directly retrieve scenario and persona from the request
    scenario_dict = request.json.get("selectedScenario")

    # Get the right model for this use-case
    agent = Agent(scenario=scenario_dict['Scenario'], persona=scenario_dict['Persona'])
    agent_response = agent.chat(agent_state=AgentState(q, message_history, world_state))
    agent_world_state = agent.evaluate(agent_state=AgentState(q, message_history, world_state))

    # Check if mode is voice-to-voice and generate audio if needed
    mode = request.json.get("mode")
    if mode == 'voice-to-voice':
        synthesis_input = texttospeech.SynthesisInput(text=agent_response.content)

        if scenario_dict['Gender'] == "Female":
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-US",
                name="en-GB-Standard-C",
                ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
            )
        else:
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-US",
                name="en-GB-Standard-D",
                ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
            )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16
        )
        response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        audio_bytes = response.audio_content
        
        # Encode audio bytes to base64 to include in JSON response
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        # Return both the text response and audio in base64 format
        return jsonify({
            "response": agent_response.content, 
            "world_state": agent_world_state,
            "audio": audio_base64}), 200
    else:
        # Return the chat response without audio
        return jsonify({
            "response": agent_response.content,
            "world_state": agent_world_state}), 200

def build_message_history(message_hist_json):
    """
    Parses client provided message history into ChatMessage objects

    Args:
        message_hist_json (_type_): [ {content="xxx", author="patient|human"} ]

    Returns:
        list[langchain.BaseMessage]: List of either AIMessage or HumanMessage
        objects, or None if empty list.

    Raises:
        ValueError: If the message author is neither "patient" nor "human" or the
        message history can't be parsed.
    """
    messages = []
    if not message_hist_json:
        return messages
    
    try:
        for message in json.loads(message_hist_json):
            if message["author"] == "human":
                messages.append(HumanMessage(message["content"]))
            elif message["author"] == "patient":
                messages.append(AIMessage(message["content"]))
            else:
                raise ValueError(f"Unknown message type: {message}")
    except Exception as e:
        raise ValueError(
            f"Couldn't parse message history: {message_hist_json}: {e}"
        ) from e
    return messages

if __name__ == '__main__':
    app.run(debug=True)