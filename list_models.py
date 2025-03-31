import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # Load from .env

if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found. Cannot list models.")
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        print("Available models supporting 'generateContent':")
        for m in genai.list_models():
          if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
    except Exception as e:
        print(f"An error occurred: {e}")
