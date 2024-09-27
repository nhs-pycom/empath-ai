document.addEventListener('DOMContentLoaded', () => {
    // Global state variable to store the selected scenario
    let selectedScenario = null;

    // Load scenario options from an external JSON file
    fetch('/static/data/scenarios.json')  // Adjust path as necessary
        .then(response => response.json())
        .then(scenarios => {
            const scenarioSelect = document.getElementById('scenario-select');
            const customScenarioInput = document.getElementById('custom-scenario-input');
            const customScenarioDiv = document.getElementById('custom-scenario');
            const generateScenarioBtn = document.getElementById('generate-scenario-btn');
            const scenarioDisplay = document.getElementById('scenario-display'); // Element to display the scenario text

            // Add "Custom" option to the select box
            let customOption = document.createElement('option');
            customOption.value = 'Custom';
            customOption.text = 'Custom';

            // Populate select box with scenarios
            for (let scenario in scenarios) {
                let option = document.createElement('option');
                option.value = scenario;
                option.text = scenario;
                scenarioSelect.appendChild(option);
            }
            // Append the "Custom" option
            scenarioSelect.appendChild(customOption);
            
            // Show custom input if 'Custom' is selected
            scenarioSelect.addEventListener('change', () => {

                if (scenarioSelect.value === 'Custom') {
                    customScenarioDiv.style.display = 'block';
                    scenarioDisplay.textContent = ''; // Clear the display area for custom scenarios
                } else {
                    customScenarioDiv.style.display = 'none';
                    selectedScenario = scenarios[scenarioSelect.value];
                    displayScenario(selectedScenario); // Display the selected scenario text
                }
            });

            // Generate scenario based on selection or custom input
            generateScenarioBtn.addEventListener('click', () => {
                let requestedScenario = {};
                
                if (scenarioSelect.value === 'Custom') {
                    requestedScenario['Scenario'] = customScenarioInput.value;
                    // You could add more custom fields if necessary
                } else {
                    requestedScenario = scenarios[scenarioSelect.value];
                }

                // Replace this with your actual model generation and response handling logic
                console.log('Generating scenario for:', requestedScenario);

                // Simulate a response (replace with actual logic)
                const generatedScenario = `Generated scenario based on ${requestedScenario['Scenario']}.`;

                // Display the results (This is where you would update the UI)
                console.log(generatedScenario);

                // Display the generated scenario and persona
                scenarioDisplay.textContent = `${generatedScenario}`;
            });

            // Function to display the scenario text
            function displayScenario(scenarioData) {
                scenarioDisplay.textContent = `${scenarioData.Scenario}`;
            }
        })
        .catch(error => console.error('Error loading scenarios:', error));

    // Mode selection
    const modeSelector = document.getElementById('interaction-mode');
    const textInputContainer = document.querySelector('.text-input-container');
    const voiceInputContainer = document.querySelector('.voice-input-container');

    modeSelector.addEventListener('change', () => {
        if (modeSelector.value === 'voice-to-voice') {
            textInputContainer.style.display = 'none';
            voiceInputContainer.style.display = 'flex';
            console.log('Switched to Voice-to-Voice mode');
            // Add any additional functionality for Voice-to-Voice mode here
        } else {
            textInputContainer.style.display = 'flex';
            voiceInputContainer.style.display = 'none';
            console.log('Switched to Text-to-Text mode');
            // Add any additional functionality for Text-to-Text mode here
        }
    });

    // Reusable function to send data to the backend
    let teacherResponse;

    function getResponse(data) {
        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),

        })
        .then(response => response.json())
        .then(data => {
            const aiResponse = data.response;
            teacherResponse = data.world_state;
            appendMessage('patient', aiResponse);

            // Add the AI's response to the conversation history
            conversationHistory.push({ author: 'patient', content: aiResponse });

            // Check if there is audio data and play it
            if (data.audio) {
                // Decode the base64 audio data and create a new Audio object
                const audio = new Audio(`data:audio/mp3;base64,${data.audio}`);
                audio.play().catch(error => console.error('Audio playback error:', error));
            }

            // Return latest evaluation
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }

    // Chatbot logic
    const chatWindow = document.getElementById('chat-window');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    
    let conversationHistory = '';

    function appendMessage(role, content) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message');
        messageElement.classList.add(role);
        messageElement.textContent = content;
        chatWindow.appendChild(messageElement);
        chatWindow.scrollTop = chatWindow.scrollHeight;  // Auto-scroll to bottom
    }

    sendBtn.addEventListener('click', function () {
        if (selectedScenario !== null) {
            const userMessage = chatInput.value.trim();
            if (userMessage !== '') {
                // Append the user's message to the chat
                appendMessage('human', userMessage);

                // Initialize conversation history as a JSON array if not already done
                if (!conversationHistory) {
                    conversationHistory = [];
                }
        
                // Add the user's message to the conversation history
                conversationHistory.push({ author: 'human', content: userMessage });
                chatInput.value = '';
        
                // Send the message and conversation history to the back-end
                getResponse({
                    message: userMessage,
                    history: JSON.stringify(conversationHistory),  // Convert conversationHistory to a JSON string
                    selectedScenario: selectedScenario,
                    mode: modeSelector.value
                });
            }
        } else {
            // Display a pop-up message if no scenario is selected
            alert('Please select a scenario before sending a message.');
        }
    });

    // Evaluation logic
    const endBtn = document.getElementById('end-btn');

    endBtn.addEventListener('click', function () {
        if (teacherResponse) {
            // Open the rubric HTML file in a new window
            const scoringWindow = window.open('/rubric', '', 'width=800,height=600');

            // Wait for the new window to fully load before populating the table
            scoringWindow.onload = function() {
                // Section 1: Professional Introduction
                scoringWindow.document.getElementById('introducesSelfWithFullNameAndRole').textContent = `${teacherResponse.introducesSelfWithFullNameAndRole} / 1`;

                // Section 2: Building Rapport and Sharing News
                scoringWindow.document.getElementById('establishesRapportWithOpenEndedQuestion').textContent = `${teacherResponse.establishesRapportWithOpenEndedQuestion} / 2`;

                // Section 3: Understanding the Situation and Delivering News
                scoringWindow.document.getElementById('confirmsPatientUnderstandingOfSituation').textContent = `${teacherResponse.confirmsPatientUnderstandingOfSituation} / 1`;
                scoringWindow.document.getElementById('securesPatientConsentAndProvidesSummary').textContent = `${teacherResponse.securesPatientConsentAndProvidesSummary} / 2`;
                scoringWindow.document.getElementById('inquiresIfPatientWantsCompanionPresent').textContent = `${teacherResponse.inquiresIfPatientWantsCompanionPresent} / 1`;
                scoringWindow.document.getElementById('providesWarningBeforeSharingBadNews').textContent = `${teacherResponse.providesWarningBeforeSharingBadNews} / 1`;
                scoringWindow.document.getElementById('effectivelyCommunicatesBadNews').textContent = `${teacherResponse.effectivelyCommunicatesBadNews} / 2`;
                scoringWindow.document.getElementById('usesPausesToAllowProcessing').textContent = `${teacherResponse.usesPausesToAllowProcessing} / 2`;

                // Section 4: Communication Techniques
                scoringWindow.document.getElementById('addressesPatientEmotionsAndConcerns').textContent = `${teacherResponse.addressesPatientEmotionsAndConcerns} / 2`;
                scoringWindow.document.getElementById('determinesPatientWantsToKnow').textContent = `${teacherResponse.determinesPatientWantsToKnow} / 2`;
                scoringWindow.document.getElementById('offersReassuranceWithoutFalseHope').textContent = `${teacherResponse.offersReassuranceWithoutFalseHope} / 2`;
                scoringWindow.document.getElementById('demonstratesEmpathy').textContent = `${teacherResponse.demonstratesEmpathy} / 1`;
                scoringWindow.document.getElementById('avoidsMedicalJargon').textContent = `${teacherResponse.avoidsMedicalJargon} / 1`;
                scoringWindow.document.getElementById('activelyListensAndRespondsToCues').textContent = `${teacherResponse.activelyListensAndRespondsToCues} / 1`;

                // Section 5: Information Delivery
                scoringWindow.document.getElementById('clearlyExplainsDiagnosisAndPrognosis').textContent = `${teacherResponse.clearlyExplainsDiagnosisAndPrognosis} / 2`;
                scoringWindow.document.getElementById('discussesReferralAndFurtherInvestigations').textContent = `${teacherResponse.discussesReferralAndFurtherInvestigations} / 2`;
                scoringWindow.document.getElementById('outlinesTreatmentOptions').textContent = `${teacherResponse.outlinesTreatmentOptions} / 2`;
                scoringWindow.document.getElementById('encouragesDiscussionWithFamily').textContent = `${teacherResponse.encouragesDiscussionWithFamily} / 1`;
                scoringWindow.document.getElementById('offersCounselingAsAppropriate').textContent = `${teacherResponse.offersCounselingAsAppropriate} / 1`;
                scoringWindow.document.getElementById('schedulesFollowUpAndProvidesInfo').textContent = `${teacherResponse.schedulesFollowUpAndProvidesInfo} / 2`;

                // Section 6: Concluding the Consultation
                scoringWindow.document.getElementById('invitesQuestionsAndProvidesContact').textContent = `${teacherResponse.invitesQuestionsAndProvidesContact} / 2`;
                scoringWindow.document.getElementById('summarizesDiscussionAndConfirmsUnderstanding').textContent = `${teacherResponse.summarizesDiscussionAndConfirmsUnderstanding} / 2`;
                scoringWindow.document.getElementById('ensuresPatientIsReadyAndSafeToLeave').textContent = `${teacherResponse.ensuresPatientIsReadyAndSafeToLeave} / 1`;
                scoringWindow.document.getElementById('maintainsStructuredApproach').textContent = `${teacherResponse.maintainsStructuredApproach} / 1`;
            };
        }
    });

    // Microphone button logic
    const micBtn = document.getElementById('mic-btn');
    const micIcon = micBtn.querySelector('i');
    let isRecording = false;
    let recognition;
    
    // Function to start speech recognition
    async function startSpeechRecognition() {
        try {
            // Initialize the SpeechRecognition object
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognition = new SpeechRecognition();
    
            // Set recognition parameters
            recognition.continuous = true; // Keep recognizing speech continuously
            recognition.interimResults = false; // Return only final results
            recognition.lang = 'en-US'; // Set the language
    
            recognition.onstart = () => {
                micIcon.classList.remove('fa-microphone');
                micIcon.classList.add('fa-times');
                micBtn.classList.add('active');
                isRecording = true;
                console.log('Speech recognition started.');
            };
    
            recognition.onresult = (event) => {
                const transcript = Array.from(event.results)
                    .map(result => result[0].transcript)
                    .join('');
                console.log('Recognized text:', transcript);

                if (transcript != '') {
                    // Append the user's message to the chat
                    appendMessage('human', transcript);

                    // Initialize conversation history as a JSON array if not already done
                    if (!conversationHistory) {
                        conversationHistory = [];
                    }
                    
                    // Add the user's message to the conversation history
                    conversationHistory.push({ author: 'human', content: transcript });

                    // Send the message and conversation history to the back-end
                    getResponse({
                        message: transcript,
                        history: JSON.stringify(conversationHistory),  // Convert conversationHistory to a JSON string
                        selectedScenario: selectedScenario,
                        mode: modeSelector.value
                    });
                }
            };
    
            recognition.onerror = (event) => {
                console.error('Speech recognition error:', event.error);
            };
    
            recognition.onend = () => {
                micIcon.classList.remove('fa-times');
                micIcon.classList.add('fa-microphone');
                micBtn.classList.remove('active');
                isRecording = false;
                console.log('Speech recognition stopped.');
            };
    
            // Start recognition
            recognition.start();
    
        } catch (error) {
            console.error('Error starting speech recognition:', error);
        }
    }
    
    // Function to stop speech recognition
    function stopSpeechRecognition() {
        if (recognition && isRecording) {
            recognition.stop();
            micIcon.classList.remove('fa-times');
            micIcon.classList.add('fa-microphone');
            micBtn.classList.remove('active');
            isRecording = false;
            console.log('Speech recognition stopped.');
        }
    }

    // Toggle recording on button click
    micBtn.addEventListener('click', () => {
        if (isRecording) {
            stopSpeechRecognition();
        } else {
            startSpeechRecognition();
        }
    });

});