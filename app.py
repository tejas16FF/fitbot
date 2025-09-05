import os
from dotenv import load_dotenv

load_dotenv()  # this loads .env into environment

# Check if key is loaded (optional debug)
print("API key loaded:", os.getenv("OPENAI_API_KEY") is not None)
