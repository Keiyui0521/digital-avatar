Digital Avatar 'Ku': An End-to-End Finetuning Project

This repository documents the creation of 'Ku,' a personalized AI digital avatar, built by finetuning the Qwen3-14B large language model and deploying it as an interactive, voice-enabled web application.

This project was built for the HUDT (Humanities and Digital Technologies) program and serves as a portfolio piece demonstrating a full-cycle MLOps workflow.

🚀 Live Demo (GIF)

(Please record a short 10-15 second GIF of your Streamlit app working and add it here. Show yourself typing a message, 'Ku' responding, and the TTS voice playing. This is the single most effective way to show a recruiter what you built.)

📖 Project Journal & Documentation

For a complete, in-depth breakdown of the project's entire lifecycle, please read the full project journal:

➡️ Read the Full Project Journal (HUDT_Project_Journal.pdf)

The journal covers:

Core Design: The decision between RAG (Retrieval Augmented Generation) vs. Finetuning.

Technique: Why LoRA (Low-Rank Adaptation) was chosen.

Platform Evaluation: A real-world comparison of SiliconFlow, Huawei Cloud, and the final choice, Xunfei Xingchen MaaS.

Data Engineering: The complete ETL (Extract, Transform, Load) pipeline for processing personal chat logs into a usable training dataset.

Model Finetuning: The full hyperparameter configuration used for the LoRA finetuning job.

Deployment: Building the Streamlit UI, integrating the MaaS API, and adding voice synthesis (TTS).

Challenges: Analysis of failures, including platform issues and local deployment VRAM limitations.

✨ Key Features

Personalized Persona: Finetuned on custom chat data to emulate a specific communication style.

LoRA Finetuning: Uses LoRA to efficiently adapt the Qwen3-1A4B base model.

Interactive Web UI: A user-friendly chat interface built with Streamlit.

Voice Synthesis: Integrated Coqui-AI's TTS model (xtts_v2) for voice-cloned, spoken responses.

Model Versioning: The app can switch between different finetuned 'Ku' model versions (e.g., v1.0, v5.0).

🛠 Tech Stack

LLM: Qwen3-14B

Finetuning: LoRA on Xunfei Xingchen MaaS

Data Processing: Python, Pandas

Web Application: Streamlit

Voice: Coqui-TTS

API: OpenAI Python Client (for MaaS endpoint)

📁 Repository Structure

/
├── README.md                # You are here
├── HUDT_Project_Journal.pdf   # The complete project write-up
├── app.py                     # The main Streamlit chat application
├── requirements.txt         # Python dependencies
│
├── assets/                    # Static files (images, voice samples)
├── data_processing/         # Python scripts for data cleaning & formatting
├── sample_data/             # Anonymized sample data to show format
└── scripts/                 # Utility scripts (e.g., API connection test)


⚙️ How to Run Locally

Clone the repository:

git clone [https://github.com/YOUR_USERNAME/ku_digital_avatar_project.git](https://github.com/YOUR_USERNAME/ku_digital_avatar_project.git)
cd ku_digital_avatar_project


Create and activate a virtual environment:

python3 -m venv .venv
source .venv/bin/activate


Install dependencies:

pip install -r requirements.txt


Set up your secrets:

IMPORTANT: This code uses hard-coded API keys for simplicity in this public repository. For production, use environment variables. Update the keys directly in app.py and scripts/connect_finetuned_model.py.

Run the Streamlit app:

streamlit run app.py


Open your browser to the local URL provided by Streamlit.