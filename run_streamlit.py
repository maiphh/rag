#!/usr/bin/env python3
"""
Simple script to run the Streamlit RAG application.
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit app."""
    
    # Change to the src directory
    src_dir = os.path.join(os.path.dirname(__file__), 'src')
    
    print("🚀 Starting RAG Streamlit App...")
    print(f"📁 Working directory: {src_dir}")
    print("🌐 The app will open in your browser at http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Run streamlit from the src directory
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            os.path.join(src_dir, "streamlit.py"),
            "--server.port", "8501",
            "--server.address", "localhost"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Shutting down the app...")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running the app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()