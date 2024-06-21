<p align="center">
  <img src="images/logo.png" alt="empath.ai">
</p>

Empath.ai is a patient conversation simulator developed to train clinicians in delivering difficult news to patients. Using large language models (LLMs), it creates diverse patient scenarios and integrates the UCLMS Mark Scheme for Breaking Bad News to provide constructive feedback. This approach helps students identify areas for improvement and enhance their communication skills in sensitive situations.

# Getting Started
To get started with empath.ai, follow these steps:

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

**Set up Vertex AI Studio**

Empath.ai is built using Vertex AI. You can modify which models are used in *utils.py*, but to use the code out of the box, follow these steps*:
1. Sign up for a Google Cloud account if you don't already have one.
2. Create a new project in the Google Cloud Console.
3. Enable the Vertex AI API for your project.
4. Set up the required service accounts and credentials.

*Full instructions for setting up the Vertex AI API are found here: https://cloud.google.com/vertex-ai/docs/start/cloud-environment

**Set up HuggingFace Access Token**

Empath.ai makes use of open-source models hosted on HuggingFace. To use these you will need an access token. To create you access token follow these steps:

1. Go to the HuggingFace website.
2. Sign up for a free account or log in if you already have one.
3. Navigate to your account settings and find the "Access Tokens" section.
4. Create a new token and copy it.

**Configure the *.env* file**

Create a .env file in the root directory of your project. The *.env* file should be structured like so:
```
GCLOUD_PROJECT_ID = <Your Google Cloud ID>
HUGGINGFACE_API_KEY = <Your HuggingFace Access Token>
```

# Usage
To launch Empath.ai, execute the following command:
```
streamlit run main.py
```

# License
This project is licensed under the MIT License. See the *LICENSE* file for details.
