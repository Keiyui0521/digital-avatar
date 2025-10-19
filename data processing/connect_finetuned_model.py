from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration START (Using environment variables for security) ---
# Load configuration from environment variables
api_key = os.getenv("MASS_API_KEY")
api_base = os.getenv("MASS_API_BASE", "https://maas-api.cn-huabei-1.xf-yun.com/v1")
model_id = os.getenv("MASS_MODEL_ID", "xop3qwen14b")
lora_resource_id = os.getenv("MASS_LORA_RESOURCE_ID", "1922568028878811136")

# Validate required environment variables
if not api_key:
    print("Error: MASS_API_KEY environment variable is not set. Please set it in your .env file or environment.")
    exit(1)
# --- Configuration END ---

client = OpenAI(api_key=api_key, base_url=api_base)

try:
    print(f"Sending request to model: {model_id} with LoRA: {lora_resource_id}")
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": "你好, 你是谁？"}], # Example prompt
        stream=False,
        temperature=0.7,
        max_tokens=4096,
        extra_headers={"lora_id": lora_resource_id},
        stream_options={"include_usage": True}, # As per documentation
        # extra_body is for features like search or thinking mode.
        # search_disable and show_ref_label are for models supporting search.
        # enable_thinking might be relevant for Qwen models like yours.
        # Defaulting to example values, adjust as needed for your model's capabilities.
        extra_body={"search_disable": False, "show_ref_label": True}
    )

    print("\n--- Response ---")
    if response.choices:
        message = response.choices[0].message
        
        # Print standard content
        if message.content:
            print(f"Content: {message.content}")
        else:
            print("Content: No content received.")

        # Safely print reasoning_content if it exists (for models supporting deep thinking)
        if hasattr(message, 'reasoning_content') and message.reasoning_content:
            print(f"Reasoning Content: {message.reasoning_content}")

        # Safely print plugins_content if it exists (for models supporting web search)
        if hasattr(message, 'plugins_content') and message.plugins_content:
            print(f"Plugins Content: {message.plugins_content}")
        
        print(f"Finish Reason: {response.choices[0].finish_reason}")

    if response.usage:
        print("\n--- Usage ---")
        print(f"Prompt Tokens: {response.usage.prompt_tokens}")
        print(f"Completion Tokens: {response.usage.completion_tokens}")
        print(f"Total Tokens: {response.usage.total_tokens}")
        
    # print("\n--- Full Response Object ---")
    # print(response)

except Exception as e:
    print(f"An error occurred: {e}") 
