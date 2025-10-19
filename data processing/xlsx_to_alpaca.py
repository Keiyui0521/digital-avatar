import pandas as pd
import json
import os

def convert_xlsx_to_alpaca(xlsx_dir, output_json_file):
    all_alpaca_data = []
    instruction_text = "你是ku。请根据提供的对话上下文和用户最新的发言，以ku的身份和风格进行回应。"

    for filename in os.listdir(xlsx_dir):
        if filename.endswith(".xlsx"):
            filepath = os.path.join(xlsx_dir, filename)
            try:
                xls = pd.ExcelFile(filepath)
            except Exception as e:
                print(f"Error reading Excel file {filepath}: {e}")
                continue

            for sheet_name in xls.sheet_names:
                try:
                    df = xls.parse(sheet_name)
                except Exception as e:
                    print(f"Error parsing sheet {sheet_name} in file {filepath}: {e}")
                    continue

                # Assuming columns are '时间', '发言人', '内容'
                # Adjust column names if they are different in your files
                # From the screenshot, it seems the relevant columns are B ('发言人') and C ('内容')
                # The first column A looks like a timestamp ('时间')

                # Try to find the correct columns, allowing for some flexibility
                speaker_col = None
                content_col = None

                # Use the exact column names provided by the user
                if '发送人' in df.columns and '内容' in df.columns:
                    speaker_col = '发送人'
                    content_col = '内容'
                # Add a check for English default names if Chinese ones are not found, as a fallback
                elif 'Speaker' in df.columns and 'Content' in df.columns:
                    speaker_col = 'Speaker'
                    content_col = 'Content'
                # Fallback to original assumption if new ones are not found - useful if user runs on old data
                elif '发言人' in df.columns and '内容' in df.columns:
                    print(f"Warning: Using '发言人' and '内容' for columns in {filepath}, sheet {sheet_name} as '发送人' was not found.")
                    speaker_col = '发言人'
                    content_col = '内容'
                else:
                    # If specific names are not found, attempt to use column indices B and C (1 and 2)
                    # This is a less reliable fallback
                    if len(df.columns) >= 3:
                        # Check if column B (index 1) appears to be a speaker column (heuristic: many unique values or NaNs)
                        # Check if column C (index 2) appears to be a content column (heuristic: mostly non-empty text)
                        # For simplicity, we'll directly try to use them and let errors downstream indicate issues if this guess is wrong.
                        # A more robust heuristic might involve checking data types or patterns.
                        print(f"Warning: Columns '发送人' and '内容' not found in {filepath}, sheet {sheet_name}. Falling back to column indices 1 (for speaker) and 2 (for content). Available columns: {df.columns.tolist()}")
                        try:
                            # Ensure the columns chosen by index are not all NaN, which would indicate they are likely not the correct ones
                            if not df.iloc[:, 1].isnull().all() and not df.iloc[:, 2].isnull().all():
                                speaker_col = df.columns[1]
                                content_col = df.columns[2]
                            else:
                                print(f"Fallback to column indices 1 and 2 failed for {filepath}, sheet {sheet_name} as they appear empty. Skipping sheet.")
                        except IndexError:
                             print(f"Fallback to column indices failed for {filepath}, sheet {sheet_name} due to insufficient columns. Skipping sheet.")
                    else:
                        print(f"Could not find required columns ('发送人', '内容', or fallbacks) in {filepath}, sheet {sheet_name}. Skipping sheet.")
                        print(f"Available columns: {df.columns.tolist()}")
                        continue


                if not speaker_col or not content_col:
                    print(f"Could not find speaker or content columns in {filepath}, sheet {sheet_name}. Skipping sheet.")
                    print(f"Available columns: {df.columns.tolist()}")
                    continue

                current_conversation = []
                user_input_buffer = []

                for _, row in df.iterrows():
                    speaker = row[speaker_col]
                    message = str(row[content_col]).strip() if pd.notna(row[content_col]) else ""

                    if not message: # Skip empty messages
                        continue

                    if pd.notna(speaker) and str(speaker).strip(): # User's message
                        user_input_buffer.append(message)
                    else: # Ku's message (speaker is NaN or empty)
                        if user_input_buffer: # If there's pending user input
                            alpaca_item = {
                                "instruction": instruction_text,
                                "input": "\n".join(user_input_buffer),
                                "output": message,
                                "system": "", # As per Alpaca format, system can be optional or empty
                                "history": [] # History is more complex, for now an empty list
                            }
                            # Potentially add previous turns to history if needed for multi-turn
                            # For this version, we'll treat each user-ku exchange as a separate item

                            # Example for simple history (last exchange):
                            # if all_alpaca_data:
                            #     last_item = all_alpaca_data[-1]
                            #     alpaca_item["history"] = [[last_item["input"], last_item["output"]]]

                            all_alpaca_data.append(alpaca_item)
                            user_input_buffer = [] # Clear buffer after Ku's response

                # If file ends with user messages without a final Ku response,
                # decide how to handle it. For now, we are only creating pairs where Ku responds.

    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(all_alpaca_data, f, ensure_ascii=False, indent=2)
    print(f"Successfully converted {len(all_alpaca_data)} entries to {output_json_file}")

if __name__ == "__main__":
    # The script expects 'avatar training data' to be in the same directory as the script,
    # or provide an absolute path.
    # For workspace, it will be relative to workspace root.
    source_directory = "avatar training data"
    output_file = "alpaca_formatted_data.json"
    
    # Check if the directory exists
    if not os.path.isdir(source_directory):
        print(f"Error: Directory '{source_directory}' not found. Make sure it's in the correct path.")
        # Attempt to find it in parent directory if script is in a subfolder - this is a guess
        # More robust: expect it to be in workspace root or CWD
        if os.path.isdir(os.path.join("..", source_directory)):
             source_directory = os.path.join("..", source_directory)
             print(f"Found '{source_directory}' in parent directory. Proceeding.")
        else:
            exit(1)

    convert_xlsx_to_alpaca(source_directory, output_file) 
