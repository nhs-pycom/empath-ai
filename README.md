<p align="center">
  <img src="static/images/logo.png" alt="empath.ai">
</p>

Empath.ai is a patient conversation simulator developed to train clinicians in delivering difficult news to patients. Using large language models (LLMs), it creates diverse patient scenarios and integrates the UCLMS Mark Scheme for Breaking Bad News to provide constructive feedback. This approach helps students identify areas for improvement and enhance their communication skills in sensitive situations.

# Getting Started
Empath.ai uses the [Cloud Vertex APIs](https://cloud.google.com/vertex-ai) for language-model calls and the [Cloud Text-To-Speech API](https://cloud.google.com/text-to-speech) to generate audio from the language model text output.
To get started with empath.ai, follow these steps:

**Cloud set-up:**

There are a few steps to this and some familiarity with Google Cloud is assumed:

1. Create a [Google Cloud project](https://cloud.google.com/cloud-console) and enable the [Vertex AI APIs](https://cloud.google.com/vertex-ai/generative-ai/docs/start/quickstarts/quickstart-multimodal). 

2. Enable the Cloud Text-to-Speech API and the Vertex AI API within your created project.

3. Follow [Google Cloud Installation Instructions](https://cloud.google.com/sdk/docs/install) to install the Google Cloud CLI.

4. Follow [Set up Application Default Credentials](https://cloud.google.com/docs/authentication/provide-credentials-adc) to set up application default credentials.

On the command line, configure your local GCloud environment and log-in with application default credentials so your local server can access the APIs in your Cloud project:

```sh
gcloud auth login
gcloud config set project [PROJECT_ID]
gcloud auth application-default login
gcloud auth application-default set-quota-project [PROJECT_ID]
```

**Clone the repository:**
```
git clone https://github.com/nhs-pycom/empath-ai.git
cd empath-ai
```

**Clone the repository:**
```
conda create --name empath-ai python=3.9
conda activate empath-ai
```

**Install dependencies**
```
pip install -r requirements.txt
```

# Usage
To launch Empath.ai, execute the following command:
```
python app.py
```

# License
This project is licensed under the MIT License. See the *LICENSE* file for details.
