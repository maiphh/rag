from dotenv import load_dotenv
import sys
import traceback
import os

from enum_manager import *
from rag import Rag

# Re-ordered and expanded menu:
MENU_OPTIONS = {
    "1": "Query",
    "2": "Adjust LLM",
    "3": "Adjust RAG Type",
    "4": "Adjust Domain",
    "5": "Adjust Threshold",
    "6": "Load New Docs",
    "7": "Clear Cache",
    "8": "Clear DB",
    "9": "Exit"
}

def format_llm_name(llm_enum: LLM) -> str:
    return llm_enum.value

def format_rag_type(rag_type: RagType) -> str:
    return rag_type.name.title().replace("_", " ")

def format_domain(domain_enum: DOMAIN) -> str:
    return domain_enum.name.title().replace("_", " ")

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

    # Active Domain
    active_domain_enum = None
    current_domain_value = rag.get_domain() if hasattr(rag, "get_domain") else getattr(rag.domain_router, "domain", "unknown")
    for d in DOMAIN:
        if d.value == current_domain_value:
            active_domain_enum = d
            break
    domain_display = format_domain(active_domain_enum) if active_domain_enum else current_domain_value

    print(f"Active LLM     : {llm_display}")
    print(f"Active RAG Type: {format_rag_type(rag.get_rag_type())}")
    print(f"Active Domain  : {domain_display}")
    print(f"Threshold      : {rag.get_threshold()}")
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
        choice = input("Select RAG Type number (or 'c' to cancel): ").strip()
        if choice.lower() == 'c':
            return
        if choice.isdigit() and 1 <= int(choice) <= len(rag_types):
            rag.set_rag_type(rag_types[int(choice) - 1])
            print(f"RAG Type set to {format_rag_type(rag.get_rag_type())}")
            return
        print("Invalid selection.")

def adjust_domain(rag: Rag):
    print("\nAvailable Domains:")
    domains = list(DOMAIN)
    for idx, d in enumerate(domains, start=1):
        print(f"{idx}. {format_domain(d)} ({d.value})")
    while True:
        choice = input("Select Domain number (or 'c' to cancel): ").strip()
        if choice.lower() == 'c':
            return
        if choice.isdigit() and 1 <= int(choice) <= len(domains):
            selected = domains[int(choice) - 1]
            try:
                rag.set_domain(selected)
                print(f"Domain set to {format_domain(selected)}")
            except Exception as e:
                print(f"Failed to set domain: {e}")
            return
        print("Invalid selection.")

def adjust_threshold(rag: Rag):
    print(f"\nCurrent threshold: {rag.get_threshold()}")
    while True:
        val = input("Enter new threshold (0.0 - 1.0) or 'c' to cancel: ").strip()
        if val.lower() == 'c':
            return
        try:
            f = float(val)
            if 0.0 <= f <= 1.0:
                rag.set_threshold(f)
                print(f"Threshold set to {f}")
                return
            else:
                print("Value out of range.")
        except ValueError:
            print("Invalid number.")

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
            response = rag.invoke(q)
            answer = response.get("answer", "No answer.")
            docs = response.get("docs", [])
            print("\n" + "-" * 50)
            print("RESULT")
            print("-" * 50)
            print(answer)
            print("-" * 50)
        except KeyboardInterrupt:
            print("\nCancelled.")
        except Exception as e:
            print(f"Error: {e}")
            if os.getenv("DEBUG"):
                traceback.print_exc()

def load_new_docs(rag:Rag):
    rag.load_documents()

def clear_cache(rag: Rag):
    """
    Clear any in-memory or persisted cache. Expects Rag.clear_cache().
    """
    rag.clear_cache()

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
            run_query_session(rag)
        elif choice == "2":
            adjust_llm(rag)
        elif choice == "3":
            adjust_rag_type(rag)
        elif choice == "4":
            adjust_domain(rag)
        elif choice == "5":
            adjust_threshold(rag)
        elif choice == "6":
            load_new_docs(rag)
        elif choice == "7":
            clear_cache(rag)
        elif choice == "8":
            clear_db(rag)
        elif choice == "9":
            print("Goodbye.")
            break
        else:
            print("Invalid option.")

if __name__ == "__main__":
    main()