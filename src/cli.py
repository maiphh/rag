from dotenv import load_dotenv
import sys
import traceback
import os

from rag import Rag, LLM, RagType
from chain import simple_rag_chain, multi_query_chain, rag_fusion_chain

# Map RagType to corresponding chain builder

MENU_OPTIONS = {
    "1": "Adjust LLM",
    "2": "Adjust RAG Type",
    "3": "Query",
    "4": "Clear DB",
    "5": "Exit"
}

def format_llm_name(llm_enum: LLM) -> str:
    return llm_enum.value

def format_rag_type(rag_type: RagType) -> str:
    return rag_type.name.title().replace("_", " ")

def print_header(rag: Rag):
    print("\n" + "=" * 60)
    print("RAG Interactive CLI")
    print("=" * 60)
    # Active LLM
    active_llm_value = None
    for enum_item in LLM:
        if enum_item.value == rag.get_llm().model:
            active_llm_value = enum_item
            break
    llm_display = active_llm_value.value if active_llm_value else getattr(rag.get_llm(), "model", "Unknown")
    print(f"Active LLM    : {llm_display}")
    print(f"Active ragType: {format_rag_type(rag.get_rag_type())}")
    print("-" * 60)
    for key, label in MENU_OPTIONS.items():
        print(f"{key}. {label}")
    print("-" * 60)

def adjust_llm(rag: Rag):
    print("\nAvailable LLMs:")
    llm_enums = list(LLM)
    for idx, enum_item in enumerate(llm_enums, start=1):
        print(f"{idx}. {enum_item.value}")
    while True:
        choice = input("Select LLM number (or 'c' to cancel): ").strip()
        if choice.lower() == 'c':
            return
        if choice.isdigit() and 1 <= int(choice) <= len(llm_enums):
            target_enum = llm_enums[int(choice) - 1]
            try:
                rag.set_llm(target_enum)
                print(f"LLM set to {target_enum.value}")
            except Exception as e:
                print(f"Failed to set LLM: {e}")
            return
        print("Invalid selection.")

def adjust_rag_type(rag: Rag):
    print("\nAvailable RAG Types:")
    rag_types = list(RagType)
    for idx, rt in enumerate(rag_types, start=1):
        print(f"{idx}. {format_rag_type(rt)}")
    while True:
        choice = input("Select ragType number (or 'c' to cancel): ").strip()
        if choice.lower() == 'c':
            return
        if choice.isdigit() and 1 <= int(choice) <= len(rag_types):
            rag.set_rag_type(rag_types[int(choice) - 1])
            print(f"ragType set to {format_rag_type(rag.get_rag_type())}")
            return
        print("Invalid selection.")

def run_query_session(rag: Rag):

    print(f"\nEntering query mode with {format_rag_type(rag.get_rag_type())}. Type 'back' to return.")
    while True:
        q = input("\nQuery: ").strip()
        if not q:
            print("Empty query. Try again.")
            continue
        if q.lower() == 'back':
            return
        try:
            print("Processing...")
            result, docs = rag.invoke(q)
            print("\n" + "-" * 50)
            print("RESULT")
            print("-" * 50)
            print(result)
            print("-" * 50)
        except KeyboardInterrupt:
            print("\nCancelled.")
        except Exception as e:
            print(f"Error: {e}")
            if os.getenv("DEBUG"):
                traceback.print_exc()

def clear_db(rag: Rag):
    confirm = input("Type 'yes' to confirm clearing database: ").strip().lower()
    if confirm == 'yes':
        try:
            rag.clear_db()
            print("Database cleared.")
        except Exception as e:
            print(f"Failed to clear DB: {e}")
    else:
        print("Cancelled.")

def main():
    load_dotenv()
    print("Initializing Rag...")
    try:
        rag = Rag()
    except Exception as e:
        print(f"Failed to initialize Rag: {e}")
        sys.exit(1)

    while True:
        print_header(rag)
        choice = input("Select option: ").strip()
        if choice == "1":
            adjust_llm(rag)
        elif choice == "2":
            adjust_rag_type(rag)
        elif choice == "3":
            run_query_session(rag)
        elif choice == "4":
            clear_db(rag)
        elif choice == "5":
            print("Goodbye.")
            break
        else:
            print("Invalid option.")

if __name__ == "__main__":
    main()