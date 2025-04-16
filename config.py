# config.py

import os
import logging
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Openrouter
OPENROUTER_API_COMPLETIONS = "https://api.openrouter.ai/v1/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Configure logging
LOG_FILE = "log.log"
LOG_LEVEL = logging.DEBUG
LOG_FORMAT = "%(asctime)s - %(filename)s - %(levelname)s - %(message)s"
logging.basicConfig(
    filename=LOG_FILE,
    level=LOG_LEVEL,
    format=LOG_FORMAT
)

# Models
# 
models = ["openai/chatgpt-4o-latest",
          "anthropic/claude-3.7-sonnet", 
          "google/gemini-2.0-flash-001", 
          "meta-llama/llama-3.3-70b-instruct",
          "deepseek/deepseek-r1"]

# Number of attempts to retrieve correct response
num_attempts = 3 # set to 5 for 'normal' execution

# Timers
execution_timeout = 3600
dryrun_timeout = 600

