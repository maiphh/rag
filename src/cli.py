from dotenv import load_dotenv
from config import settings
from db import chromaDb
from document_loader import document_loader
from dotenv import load_dotenv
import os
from test import test_tracing
from chain import chain

# Configuration for all RAG chains - ADD NEW CHAINS HERE ONLY
RAG_CHAINS = {
    1: {
        "name": "Simple RAG Chain",
        "description": "Basic retrieval-augmented generation",
        "function": lambda llm: chain.simple_rag_chain(llm)
    },
    2: {
        "name": "Multi-Query RAG Chain",
        "description": "Generates multiple queries for better retrieval",
        "function": lambda llm: chain.multi_query_chain(llm)
    },

    3: {
        "name" : "RAG Fusion",
        "description": "Combines results from multiple queries using Reciprocal Rank Fusion",
        "function": lambda llm: chain.rag_fusion_chain(llm)
    }

    # Add new chains here:
    # 3: {
    #     "name": "Your New Chain",
    #     "description": "Description of your new chain",
    #     "function": lambda llm: chain.your_new_chain(llm)
    # }
}











def display_menu():
    """Display the available RAG chain options dynamically"""
    print("\n" + "="*60)
    print("RAG Chain Selection Menu")
    print("="*60)
    
    # Dynamically generate menu from RAG_CHAINS config
    for key, chain_info in RAG_CHAINS.items():
        print(f"{key}. {chain_info['name']}")
        print(f"   └─ {chain_info['description']}")
    
    print(f"{len(RAG_CHAINS) + 1}. Exit")
    print("="*60)

def get_user_choice():
    """Get and validate user's menu choice dynamically"""
    valid_choices = list(RAG_CHAINS.keys()) + [len(RAG_CHAINS) + 1]  # Include exit option
    
    while True:
        try:
            choice = int(input(f"Please select an option (1-{len(RAG_CHAINS) + 1}): "))
            if choice in valid_choices:
                return choice
            else:
                print(f"Invalid choice. Please select from {min(valid_choices)}-{max(valid_choices)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def get_user_query():
    """Get the query from user"""
    while True:
        query = input("\nEnter your query (or 'back' to return to menu): ").strip()
        if query:
            return query
        print("Query cannot be empty. Please try again.")

def run_rag_chain(chain_type, llm, query):
    """Execute the selected RAG chain with the user's query"""
    try:
        print(f"\nProcessing query: '{query}'")
        print("Loading...")
        
        # Get chain info from configuration
        chain_info = RAG_CHAINS[chain_type]
        print(f"Using {chain_info['name']}...")
        
        # Execute the chain function
        rag_chain = chain_info['function'](llm)
        result = rag_chain.invoke(query)
        
        print("\n" + "-"*50)
        print("RESULT:")
        print("-"*50)
        print(result)
        print("-"*50)
        
    except Exception as e:
        print(f"Error processing query: {str(e)}")

def main():
    # Load environment variables
    if not load_dotenv():
        print(".env file not found")
        return

    print("Initializing RAG system...")
    
    # Initialize components
    try:
        llm = settings.get_llm()
        embed = settings.get_embed()
        
        # Load and process documents
        print("Loading documents...")
        docs = document_loader.load_documents()
        chunks = document_loader.split_documents(docs)
        chromaDb.add_to_db(chunks)
        
        print("RAG system initialized successfully!")
        
    except Exception as e:
        print(f"Error initializing RAG system: {str(e)}")
        return

    # Main CLI loop
    exit_option = len(RAG_CHAINS) + 1
    
    while True:
        display_menu()
        choice = get_user_choice()
        
        if choice == exit_option:
            print("Thank you for using the RAG system. Goodbye!")
            break
        
        query = get_user_query()
        
        if query.lower() == 'back':
            continue
            
        run_rag_chain(choice, llm, query)
        
        # Ask if user wants to continue
        while True:
            continue_choice = input("\nWould you like to make another query? (y/n): ").lower().strip()
            if continue_choice in ['y', 'yes']:
                break
            elif continue_choice in ['n', 'no']:
                print("Thank you for using the RAG system. Goodbye!")
                return
            else:
                print("Please enter 'y' for yes or 'n' for no.")

if __name__ == "__main__":
    main()