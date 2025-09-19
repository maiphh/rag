"""
Example usage of DocumentLoader to get all files in root directory.
"""
from src.document_loader import DocumentLoader
import os

def main():
    # Create document loader instance
    loader = DocumentLoader()
    
    # Example 1: Get all files in default directory
    print("=== All files in default directory ===")
    all_files = loader.load()
    for file in all_files:
        print(file)
    
    # Example 2: Get all files in specific directory
    print("\n=== All files in specific directory ===")
    specific_dir = "./data"  # Change this to your target directory
    if os.path.exists(specific_dir):
        files = loader.get_all_files(specific_dir)
        for file in files:
            print(file)
    else:
        print(f"Directory {specific_dir} does not exist")
    
    # Example 3: Get only PDF files
    print("\n=== PDF files only ===")
    pdf_files = loader.get_pdf_files("./data")
    for file in pdf_files:
        print(file)
    
    # Example 4: Get files with specific extensions
    print("\n=== Files with specific extensions (.pdf, .txt, .docx) ===")
    filtered_files = loader.get_files_by_extension("./data", ['.pdf', '.txt', '.docx'])
    for file in filtered_files:
        print(file)
    
    # Example 5: Get files in current working directory
    print("\n=== Files in current working directory ===")
    current_dir_files = loader.get_all_files(".")
    for file in current_dir_files[:10]:  # Show only first 10 files
        print(file)
    if len(current_dir_files) > 10:
        print(f"... and {len(current_dir_files) - 10} more files")

if __name__ == "__main__":
    main()