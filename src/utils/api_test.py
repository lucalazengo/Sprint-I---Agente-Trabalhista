# In your terminal, first run:
# pip install xai-sdk python-dotenv

import sys
from pathlib import Path

# Add the utils directory to the path so we can import config
utils_dir = Path(__file__).parent
if str(utils_dir) not in sys.path:
    sys.path.insert(0, str(utils_dir))

from xai_sdk import Client
from xai_sdk.chat import user, system
from config import get_xai_api_key

# Get API key using the centralized config utility
api_key = get_xai_api_key()

client = Client(
    api_key=api_key,
    timeout=3600, # Override default timeout with longer timeout for reasoning models
)

# Use a valid model name - check xAI documentation for latest model names
# Common models: "grok-beta", "grok-2", etc.
chat = client.chat.create(model="grok-beta")
chat.append(system("You are Grok, a highly intelligent, helpful AI assistant."))
chat.append(user("What is the meaning of life, the universe, and everything?"))

response = chat.sample()
print(response.content)