import os
import pandas as pd
from pathlib import Path

def delete_small_xlsx_files(directory_path, min_lines=7):
    # Convert to Path object for better path handling
    directory = Path(directory_path)
    
    # Counter for deleted files
    deleted_count = 0
    
    # Iterate through all xlsx files in the directory
    for xlsx_file in directory.glob('*.xlsx'):
        try:
            # Read the Excel file
            df = pd.read_excel(xlsx_file)
            
            # Get number of rows
            num_rows = len(df)
            
            # If file has fewer than min_lines rows, delete it
            if num_rows < min_lines:
                print(f"Deleting {xlsx_file.name} - {num_rows} rows")
                os.remove(xlsx_file)
                deleted_count += 1
            else:
                print(f"Keeping {xlsx_file.name} - {num_rows} rows")
                
        except Exception as e:
            print(f"Error processing {xlsx_file.name}: {str(e)}")
    
    print(f"\nTotal files deleted: {deleted_count}")

if __name__ == "__main__":
    directory_path = "avatar training data"
    delete_small_xlsx_files(directory_path) 
