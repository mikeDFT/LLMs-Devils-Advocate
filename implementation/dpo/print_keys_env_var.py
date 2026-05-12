import os
from dotenv import load_dotenv

# for these API keys you need to create a .env file in the root of the project with:
# GROQ_API_KEY=your_groq_api_key_here
# GEMINI_API_KEY=your_gemini_api_key_here

def main():
    load_dotenv()
    
    groq_key = os.environ.get("GROQ_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY")
    
    print(f"GROQ_API_KEY: {groq_key}")
    print(f"GEMINI_API_KEY: {gemini_key}")

if __name__ == "__main__":
    main()
