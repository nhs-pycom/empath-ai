import streamlit as st
import json
from src.utils import get_response, make_model, evaluate_patient_emotions, format_custom_gen_prompt, clean_markdown
import plotly.express as px
import pandas as pd
import numpy as np

st.set_page_config(page_title="empath.ai, powered by Gemini", initial_sidebar_state="collapsed", menu_items=None)
st.markdown('''
            <style>
                section.main > div {max-width:50rem}
            </style>
            ''', unsafe_allow_html=True)

st.markdown("""
<style >
.stDownloadButton, div.stButton {text-align:center}
.stDownloadButton button, div.stButton > button:first-child {
    padding-left: 20px;
    padding-right: 20px;
}

.stDownloadButton button:hover, div.stButton > button:hover {
    }
</style>""", unsafe_allow_html=True)

st.image('images/logo.png')

if 'emotions_dict' not in st.session_state:
    emotions = ['joy','love','optimism','trust','anticipation','fear',
                'surprise','sadness','pessimism','disgust','anger']
    st.session_state['emotions_dict'] = {k: [] for k in emotions}

if 'scenario_dict' not in st.session_state:
    st.session_state['scenario_dict'] = json.load(open('scenarios/scenarios.json'))

if 'scenario' not in st.session_state:
    st.session_state['scenario'] = ''

if 'persona' not in st.session_state:
    st.session_state['persona'] = ''

scenario = st.selectbox('Select a scenario', list(st.session_state['scenario_dict'].keys())+['Custom']) 

if scenario == 'Custom':
    requested_scenario = st.text_input('Enter what you want the doctor to tell the patient.')
    
    scenario_gen_prompt = json.load(open('prompts/scenario_gen.json'))
    persona_gen_prompt = json.load(open('prompts/persona_gen.json'))
    formatted_scenario_gen_prompt = format_custom_gen_prompt(scenario_gen_prompt, requested_scenario)
    
    generate_scenario = st.button('Generate Scenario')
    if generate_scenario:
        scenario_model = make_model(system_instructions='')
        generated_scenario = get_response(scenario_model, formatted_scenario_gen_prompt)
        st.session_state['scenario'] = json.loads(clean_markdown(generated_scenario))['Custom']
        
        persona_model = make_model(system_instructions='')
        formatted_persona_gen_prompt = format_custom_gen_prompt(persona_gen_prompt, st.session_state['scenario'])
        generated_persona = get_response(scenario_model, formatted_persona_gen_prompt)
        st.session_state['persona'] = 'Embody this patient: \n' + json.loads(clean_markdown(generated_persona))['Custom'] + '\n\nOnly return what the patient says and do not include anything else.'
else:
    st.session_state['scenario'] = st.session_state['scenario_dict'][scenario]['Scenario']
    st.session_state['persona'] = 'Embody this patient: \n' + st.session_state['scenario_dict'][scenario]['Persona'] + '\n\nOnly return what the patient says and do not include anything else.'

st.write(st.session_state['scenario'])
patient_model = make_model(st.session_state['persona'])

left_col, _, right_col = st.columns([10,1,10])

with left_col:
    chat_window = st.container(height=350)
    if 'messages' not in st.session_state:
        st.session_state.messages = [{"role": "ai", "content": "Hello Doctor..."}]
    
    if prompt := st.chat_input("Your response"):
        st.session_state.messages.append({"role": "user", "content": prompt})
    
    conversation_history = ''
    for message in st.session_state.messages:
        with chat_window.chat_message(message["role"]):
            st.write(message["content"])
            conversation_history += f'{message["role"].replace("ai", "Patient").replace("user", "Clinician")}: {message["content"]}\n'
    
    if st.session_state.messages[-1]["role"] != "ai":
        with chat_window.chat_message("ai"):
            with st.spinner("Thinking..."):
                response = get_response(patient_model, conversation_history + 'Patient: ')
                
                emotions = evaluate_patient_emotions({"inputs": response})
                for emotion in emotions[0]:
                    st.session_state['emotions_dict'][emotion['label']].append(emotion['score'])
                
                message = {"role": "ai", "content": response}
                st.session_state.messages.append(message)
                st.write(message["content"])

                                
with right_col:
    if st.button('Evaluate Performance'):
        with open('prompts/teacher_prompt.json') as f:
            teacher_prompt = json.load(f)
        
        teacher_model = make_model('You are to evaluate the performance of this candidate')
        response = get_response(teacher_model, teacher_prompt['prompt']['parts'][0]['text'].replace('{TRANSCRIPT}', conversation_history))
        
        df = pd.DataFrame({
            'r': [np.mean(scores[-3:]) for scores in st.session_state['emotions_dict'].values()],
            'theta': [emotion.capitalize() for emotion in st.session_state['emotions_dict'].keys()]
        })
        
        fig = px.line_polar(df, r='r', theta='theta', line_close=True)
        fig.update_traces(line_color='#d4a5bc', fill='toself', fillcolor='#DDC6D1')
        fig.update_layout(
            width=350,
            height=350,
            polar=dict(
                radialaxis=dict(showgrid=True, ticks='', color='white'),
                angularaxis=dict(showgrid=True, color='white', tickfont=dict(color='black'))
            )
        )
        fig.update_polars(
            angularaxis_showgrid=False,
            radialaxis_gridwidth=0,
            gridshape='linear',
            bgcolor="#e8ecf8",
            radialaxis_showticklabels=False
        )
        
        with st.popover("--- View Detailed Feedback ---", use_container_width=True):
            st.write(response)
        
        st.plotly_chart(fig, use_container_width=True)
    


        