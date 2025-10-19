from openai import OpenAI
import streamlit as st
import json
import os # Added for TTS
import tempfile # Added for TTS
from TTS.api import TTS # Added for TTS
import torch # Added for PyTorch 2.6+ TTS model loading fix
from TTS.tts.configs import xtts_config # Changed import for PyTorch 2.6+ fix
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs # Added XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig # Added for the new class
import base64 # Added for base64 encoding
import re # Added for regex pattern matching
from dotenv import load_dotenv # Added for loading environment variables

# Load environment variables from .env file
load_dotenv()

# --- Configuration START (Using environment variables for security) ---
# Load configuration from environment variables
api_key = os.getenv("MASS_API_KEY")
api_base = os.getenv("MASS_API_BASE", "https://maas-api.cn-huabei-1.xf-yun.com/v1")

# Validate required environment variables
if not api_key:
    st.error("Error: MASS_API_KEY environment variable is not set. Please set it in your .env file or environment.")
    st.stop()

# Model configurations
model_configs = {
    "KU1.0": {
        "model_id": os.getenv("KU1_MODEL_ID", "xop3qwen14b"),
        "lora_resource_id": os.getenv("KU1_LORA_RESOURCE_ID", "1922568028878811136")
    },
    "KU5.0": {
        "model_id": os.getenv("KU5_MODEL_ID", "xop3qwen14b"),
        "lora_resource_id": os.getenv("KU5_LORA_RESOURCE_ID", "1922568028878811136")
    }
}

# Default system prompt to be added to all conversations
DEFAULT_SYSTEM_PROMPT = "你是ku。请根据提供的对话上下文和用户最新的发言，以ku的身份和风格进行回应。"
# --- Configuration END ---

client = OpenAI(api_key=api_key, base_url=api_base)

# --- TTS Initialization ---
SPEAKER_WAV_PATH = "recording_sample.WAV" # Relative to this script

# Add XttsConfig to safe globals for PyTorch 2.6+
# This needs to be done before TTS model loading if PyTorch version is >= 2.6
# and the model uses this config class.
torch.serialization.add_safe_globals([xtts_config.XttsConfig]) # Use direct module reference

@st.cache_resource # Cache the TTS model for performance
def load_tts_model():
    # Consider gpu=True if on a compatible CUDA environment and want faster synthesis
    # For M1 Mac, PyTorch MPS can sometimes be used if TTS/PyTorch versions support it well.
    # Sticking to gpu=False (CPU) for broader compatibility initially.
    # Model will be downloaded on first run if not already cached by TTS library
    try:
        # Use safe_globals context manager directly around the TTS model instantiation
        with torch.serialization.safe_globals([
            xtts_config.XttsConfig, 
            XttsAudioConfig, 
            BaseDatasetConfig,
            XttsArgs # Added new class
        ]):
            model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
        return model
    except Exception as e:
        st.error(f"Failed to load TTS model: {e}")
        return None

# tts_model = load_tts_model() # Moved down

# --- Streamlit App ---

st.set_page_config(page_title="KU's digital avatar") # Optional: Set browser tab title

tts_model = load_tts_model() # Moved here, after set_page_config

# Custom CSS for better chat UI
st.markdown("""
<style>
.user-message {
    background-color: #F0F2F6;
    border-radius: 10px;
    padding: 10px 15px;
    margin: 5px 0;
    text-align: right;
    max-width: 80%;
    margin-left: auto;
    color: #000;  /* Explicitly set text color to black */
    font-weight: 500;  /* Slightly bold for better readability */
}
.assistant-message {
    background-color: #E1E6EB;
    border-radius: 10px;
    padding: 10px 15px;
    margin: 5px 0;
    text-align: left;
    max-width: 80%;
    color: #000;  /* Explicitly set text color to black */
    font-weight: 500;  /* Slightly bold for better readability */
}
.thinking-message {
    background-color: #E1E6EB;
    border-radius: 10px;
    padding: 10px 15px;
    margin: 5px 0;
    text-align: left;
    max-width: 80%;
    color: #666;
    font-style: italic;
}
.main-container {
    display: flex;
    flex-direction: column;
    height: 100vh;
}
.chat-area {
    flex-grow: 1;
    overflow-y: auto;
    padding-bottom: 80px;
}
.hidden-audio {
    display: none;  /* Hide the audio player completely */
}
.model-toggle {
    text-align: center;
    margin-bottom: 10px;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    font-weight: bold;
}
.model-selector {
    margin: 15px auto;
    max-width: 300px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Display image at the top, without caption
st.image("avatar_figure.png")

st.title("KU's digital avatar") # Title below the image

# Initialize chat history in session state if it doesn\'t exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Session state to track if we're waiting for a response
if "waiting_for_response" not in st.session_state:
    st.session_state.waiting_for_response = False

# Session state to track if audio should be played and prevent early page refresh
if "play_audio" not in st.session_state:
    st.session_state.play_audio = False
    st.session_state.audio_data = None

# Session state to track which model version is selected
if "model_version" not in st.session_state:
    st.session_state.model_version = "KU1.0"  # Default to KU1.0

# Model version change handler
def on_model_change():
    # Clear chat history when switching models
    st.session_state.messages = []

# Create main layout with two main sections
main_container = st.container()
with main_container:
    # Model version selection
    with st.container():
        st.markdown('<div class="model-selector">', unsafe_allow_html=True)
        st.selectbox(
            "Select Model Version:",
            options=list(model_configs.keys()),
            index=list(model_configs.keys()).index(st.session_state.model_version),
            key="model_version",
            on_change=on_model_change
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Top section for chat history (grows to fill space)
    chat_area = st.container()
    
    # Display all past messages
    with chat_area:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-message">{message["content"]}</div>', unsafe_allow_html=True)
        
        # Show "Thinking..." message if we're waiting for a response
        if st.session_state.waiting_for_response:
            st.markdown(f'<div class="thinking-message">Thinking...</div>', unsafe_allow_html=True)
    
    # Display audio if needed
    if st.session_state.play_audio and st.session_state.audio_data:
        st.markdown(
            f'<audio autoplay class="hidden-audio">'+
            f'<source src="data:audio/wav;base64,{st.session_state.audio_data}" type="audio/wav">'+
            f'</audio>',
            unsafe_allow_html=True
        )
        # Reset after playing to avoid replaying on every rerun
        st.session_state.play_audio = False
    
    # Bottom section always remains at the bottom for input
    with st.container():
        with st.form(key="chat_form", clear_on_submit=True):
            prompt = st.text_input("", placeholder="What would you like to say?", key="user_prompt_input_text")
            submit_button = st.form_submit_button("Send")
    
# Function to call the MaaS API
def get_model_response(current_conversation_history):
    try:
        # Get current model configuration based on selected version
        model_config = model_configs[st.session_state.model_version]
        model_id = model_config["model_id"]
        lora_resource_id = model_config["lora_resource_id"]

        # Add system prompt to the beginning of the conversation
        conversation_with_system = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
        conversation_with_system.extend(current_conversation_history)

        chat_completion = client.chat.completions.create(
            model=model_id,
            messages=conversation_with_system,
            temperature=0.7,  # Adjust as needed
            max_tokens=4096,  # Added from connect_finetuned_model.py
            extra_headers={"lora_id": lora_resource_id}, 
            stream_options={"include_usage": True},
            extra_body={"search_disable": False, "show_ref_label": True}
        )
        
        assistant_response = chat_completion.choices[0].message.content
        return assistant_response.strip()

    except Exception as e:
        st.error(f"Error communicating with the model: {e}")
        return None

# Handle form submission
if submit_button and prompt and not st.session_state.waiting_for_response:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Set flag to show "Thinking..." message
    st.session_state.waiting_for_response = True
    # Force a rerun to display the thinking message
    st.experimental_rerun()

# Process response if we're waiting
if st.session_state.waiting_for_response:
    # Prepare conversation history for the API call
    api_conversation_history = []
    for msg in st.session_state.messages:
        api_conversation_history.append({"role": msg["role"], "content": msg["content"]})
    
    # Get model response (no spinner needed as we have the "Thinking..." message)
    assistant_response = get_model_response(api_conversation_history)

    if assistant_response:
        # Clear waiting flag
        st.session_state.waiting_for_response = False
        
        # Clean up special tokens or unwanted text
        assistant_response = assistant_response.replace("<end>", "")

        # Replace placeholders with emojis (order can matter if tags are substrings of others, but not an issue here)
        assistant_response = assistant_response.replace("[炸弹]", "💣")
        assistant_response = assistant_response.replace("[流泪]", "😭")
        assistant_response = assistant_response.replace("[大哭]", "😭") # Synonym for 流泪
        assistant_response = assistant_response.replace("[微笑]", "😊")
        assistant_response = assistant_response.replace("[调皮]", "😝")
        assistant_response = assistant_response.replace("[呲牙]", "😁")
        assistant_response = assistant_response.replace("[可爱]", "🥰")
        assistant_response = assistant_response.replace("[爱心]", "❤️")
        assistant_response = assistant_response.replace("[偷笑]", "😂")
        assistant_response = assistant_response.replace("[再见]", "👋")
        assistant_response = assistant_response.replace("[发呆]", "🤔")
        assistant_response = assistant_response.replace("[疑问]", "🤔")
        assistant_response = assistant_response.replace("[傲慢]", "😒") # Arrogant/Haughty
        assistant_response = assistant_response.replace("[撇嘴]", "😒")
        assistant_response = assistant_response.replace("[鼓掌]", "👏")
        assistant_response = assistant_response.replace("[尴尬]", "😅")
        assistant_response = assistant_response.replace("[发怒]", "😠")
        assistant_response = assistant_response.replace("[奋斗]", "💪")
        assistant_response = assistant_response.replace("[惊恐]", "😱")
        assistant_response = assistant_response.replace("[恐惧]", "😱") # Added new emoji
        assistant_response = assistant_response.replace("[惊讶]", "😮")
        assistant_response = assistant_response.replace("[酷]", "😎")
        assistant_response = assistant_response.replace("[愉快]", "😄")
        assistant_response = assistant_response.replace("[委屈]", "🥺")
        assistant_response = assistant_response.replace("[阴险]", "😏")
        assistant_response = assistant_response.replace("[赞]", "👍")
        assistant_response = assistant_response.replace("[猪头]", "🐷")
        assistant_response = assistant_response.replace("[抱拳]", "🙏")
        assistant_response = assistant_response.replace("[握手]", "🤝")
        # Add more specific replacements above this line if needed

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        
        # Function to check if text contains only punctuation and symbols
        def is_only_punctuation(text):
            # Remove spaces and check if only punctuation/symbols remain
            text = text.strip()
            if not text:
                return True
            
            # Chinese/English punctuation and symbols
            punctuation_pattern = r'^[\s\.\,\，\。\!\?\？\！\:\;\：\；\"\'\"\"\'\'\(\)\（\）\[\]\【\】\{\}\<\>\《\》\-\_\=\+\~\～\@\#\$\%\^\&\*\°\…\′\″\‖\|\·\・\～]*$'
            return bool(re.match(punctuation_pattern, text))

        # Generate TTS audio - only if conditions are met:
        # 1. Text doesn't contain emoji placeholders
        # 2. Text isn't only punctuation or symbols
        emoji_patterns = [
            "[炸弹]", "[流泪]", "[大哭]", "[微笑]", "[调皮]", "[呲牙]", "[可爱]", "[爱心]", 
            "[偷笑]", "[再见]", "[发呆]", "[疑问]", "[傲慢]", "[撇嘴]", "[鼓掌]", "[尴尬]", 
            "[发怒]", "[奋斗]", "[惊恐]", "[恐惧]", "[惊讶]", "[酷]", "[愉快]", "[委屈]", 
            "[阴险]", "[赞]", "[猪头]", "[抱拳]", "[握手]"
        ]

        contains_emoji = any(pattern in assistant_response for pattern in emoji_patterns)
        contains_only_punctuation = is_only_punctuation(assistant_response)
        original_response = assistant_response # Store for checking

        if tts_model and assistant_response and not contains_emoji and not contains_only_punctuation:
            try:
                # Language setting for TTS
                tts_language = "zh-cn"
                
                # Check if speaker WAV exists
                if not os.path.exists(SPEAKER_WAV_PATH):
                    st.warning(f"Speaker WAV file not found at {SPEAKER_WAV_PATH}. TTS will use a default voice.")
                    speaker_arg = None
                else:
                    speaker_arg = SPEAKER_WAV_PATH

                # Create a temporary file for the audio
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio_file:
                    output_audio_path = tmp_audio_file.name
                
                # Generate TTS audio
                if speaker_arg:
                    tts_model.tts_to_file(
                        text=assistant_response,
                        speaker_wav=speaker_arg,
                        language=tts_language,
                        file_path=output_audio_path
                    )
                else:
                    # Fallback if speaker_wav is missing
                    tts_model.tts_to_file(
                        text=assistant_response,
                        language=tts_language,
                        file_path=output_audio_path
                    )

                # Read audio file and convert to base64
                with open(output_audio_path, "rb") as audio_file:
                    audio_bytes = audio_file.read()
                    audio_base64 = base64.b64encode(audio_bytes).decode()
                
                # Set the audio data in session state to be played in the next render
                st.session_state.audio_data = audio_base64
                st.session_state.play_audio = True
                
                # Clean up temporary file
                os.remove(output_audio_path)

            except Exception as e:
                st.error(f"Error generating audio: {e}")
                # Clean up temp file if it exists
                if 'output_audio_path' in locals() and os.path.exists(output_audio_path):
                    try:
                        os.remove(output_audio_path)
                    except Exception:
                        pass
        
        # Rerun to update the UI with new message
        st.experimental_rerun()

# To run this app:
# 1. Make sure you are in the \'digital avatar\' directory in your terminal.
# 2. Make sure your virtual environment is activated: source .venv/bin/activate
# 3. Run: streamlit run maas_chat_interface.py
# A new tab should open in your web browser with the chat interface. 
