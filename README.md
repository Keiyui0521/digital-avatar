# Project Log: The Making of 'Ku', My Digital Avatar

This repository documents the end-to-end development of 'Ku', a personalized AI built by finetuning the Qwen3-14B Large Language Model. The project's goal was to create a "digital avatar" that emulates my personal communication style.

This project was developed for the HUDT (Humanities and Digital Technologies) program and serves as a portfolio piece demonstrating a full-cycle MLOps workflow.

## Final Application Demo

The final output is an interactive web application allowing real-time, voice-enabled conversation with the finetuned 'Ku' model.

## Development Log

This project was executed in three distinct phases, from initial strategy to final deployment.

### Phase 1: Strategy and Technical Selection

**Objective:** Define the core technical approach for creating a personalized AI persona.

* **Methodology:** Finetuning was selected over Retrieval-Augmented Generation (RAG). The goal was to deeply embed a communication style and persona into the model's parameters, rather than simply enabling it to retrieve facts about me.
* **Technique:** Low-Rank Adaptation (LoRA) was chosen as the finetuning technique. This allowed for efficient training on the 14-billion parameter base model without requiring prohibitive computational resources.

### Phase 2: Data ETL (Extract, Transform, Load) Pipeline

**Objective:** Process raw, unstructured chat logs into a clean, structured dataset suitable for training.

* **Data Cleaning:** A Python script (`delete_small_xlsx.py`) was written to automatically parse a directory of Excel files and remove conversations with fewer than seven turns, ensuring sufficient conversational context.
* **Data Transformation:** The core script (`xlsx_to_alpaca.py`) was developed to convert the cleaned chat logs into the strict Alpaca JSON format. This script handles multi-turn dialogues and structures them into the required "instruction-input-output" schema for the finetuning process.

### Phase 3: Model Training and Application Deployment

**Objective:** Train the custom LoRA adapter and build a user-facing application for interaction.

* **Model Training:** A LoRA finetuning job was configured and executed on the Xunfei Xingchen MaaS platform, using the dataset created in Phase 2.
* **Application Interface:** A web-based chat interface was built using Streamlit (`maas_chat_interface.py`).
* **Voice Synthesis:** The application was enhanced with a Text-to-Speech (TTS) feature by integrating Coqui-AI's `xtts_v2` model, enabling 'Ku' to speak its responses in a cloned voice.

## Technical Stack

* **LLM:** Qwen3-14B
* **Finetuning Method:** LoRA
* **Data Processing:** Python, Pandas
* **Web Application:** Streamlit
* **Voice Synthesis:** Coqui-TTS

## Full Project Documentation

This README provides a high-level summary. For a comprehensive breakdown of the technical details, platform comparisons, challenges, and hyperparameter configurations, please refer to the full project journal.

➡️ [Click here to read the full HUDT Project Journal](https://github.com/Keiyui0521/digital-avatar)

## How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Keiyui0521/digital-avatar.git](https://github.com/Keiyui0521/digital-avatar.git)
    cd digital-avatar
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure API Keys:**
    *Note: For this portfolio, API keys are hard-coded. For production, always use environment variables. Update keys in `data processing/maas_chat_interface.py` and `data processing/connect_finetuned_model.py`.*

5.  **Run the Streamlit App:**
    ```bash
    streamlit run "data processing/maas_chat_interface.py"
    ```
