import os
import pandas as pd
import json
from pathlib import Path

def convert_xlsx_to_sharegpt(input_dir, output_file):
    all_formatted_conversations = []
    directory = Path(input_dir)

    for xlsx_file in directory.glob('*.xlsx'):
        contact_name = xlsx_file.stem  # Use filename (without extension) as contact name
        conversation_id = f"{contact_name}_chat_log"
        
        try:
            df = pd.read_excel(xlsx_file)
            
            # Adjust column names based on the provided image
            sender_col = '发送人'
            message_col = '内容'

            if sender_col not in df.columns or message_col not in df.columns:
                print(f"Skipping {xlsx_file.name}: Missing '{sender_col}' or '{message_col}' column.")
                continue

            messages = []
            # Add the initial system prompt
            messages.append({
                "from": "system",
                "value": "你是ku。请根据对话内容自然回应。"
            })

            for index, row in df.iterrows():
                # Handle potentially missing sender value (NaN for 'ku')
                sender_val = row[sender_col]
                if pd.isna(sender_val) or str(sender_val).strip() == "":
                    sender = "ku"
                else:
                    sender = str(sender_val).strip()
                
                message_text = str(row[message_col]).strip()

                if not message_text:  # Skip empty messages
                    continue

                if sender.lower() == "ku":
                    messages.append({"from": "ku", "value": message_text})
                else:
                    # All other senders are treated as the 'user' for this contact
                    # The actual sender name from the column is used for logging/ID but 'user' for ShareGPT
                    messages.append({"from": "user", "value": message_text})
            
            if len(messages) > 1: # Only add if there are actual messages beyond system prompt
                all_formatted_conversations.append({
                    "id": conversation_id,
                    "conversations": messages
                })
            else:
                print(f"Skipping {xlsx_file.name}: No actual messages found after system prompt.")

        except Exception as e:
            print(f"Error processing {xlsx_file.name}: {e}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_formatted_conversations, f, ensure_ascii=False, indent=2)
    
    print(f"\nSuccessfully converted {len(all_formatted_conversations)} conversations to {output_file}")

if __name__ == "__main__":
    input_directory = "avatar training data"
    output_json_file = "all_conversations_sharegpt.json"
    
    # Ensure the input directory exists
    if not Path(input_directory).is_dir():
        print(f"Error: Input directory '{input_directory}' not found.")
    else:
        convert_xlsx_to_sharegpt(input_directory, output_json_file) 
