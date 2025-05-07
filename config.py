# config.py

import os
import logging
from dotenv import load_dotenv
from challenges.FOURTOPS.fourtops import fourtop_challenge

# Load environment variables from the .env file
load_dotenv()

# Openrouter
OPENROUTER_API_COMPLETIONS = "https://openrouter.ai/api/v1/completions"
OPENROUTER_API_MODELS = "https://openrouter.ai/api/v1/models"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Configure logging
LOG_FILE = "log.log"
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(filename)s - %(levelname)s - %(message)s"
logging.basicConfig(
    filename=LOG_FILE,
    level=LOG_LEVEL,
    format=LOG_FORMAT
)

# Models
models = ["openai/gpt-4.1",
          "anthropic/claude-3.7-sonnet",
          "google/gemini-2.5-pro-preview-03-25", 
          "x-ai/grok-3-beta",
          "deepseek/deepseek-chat-v3-0324"]


# Number of attempts to retrieve correct response
num_attempts = 5 # set to 5 for 'normal' execution

# Timers
execution_timeout = 3600
dryrun_timeout = 600

# All the challenges in the pipeline
challenges = [fourtop_challenge]

# Docker Image
DOCKER_IMAGE = "llm-script-sandbox:latest"

