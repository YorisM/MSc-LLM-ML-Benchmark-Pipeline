# config.py

import os
import logging
from dotenv import load_dotenv
from challenges.FOURTOPS.fourtops import fourtop_challenge
from challenges.TRACKFORMERS.trackformers import trackformers_challenge

# Load environment variables from the .env file
load_dotenv()

# Openrouter
OPENROUTER_API_COMPLETIONS = "https://openrouter.ai/api/v1/completions"
OPENROUTER_API_CHAT = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_MODELS = "https://openrouter.ai/api/v1/models"
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

# Models - Needs at least two models 
models = [
          "openai/o4-mini-high-2025-04-16",
          "google/gemini-2.5-pro"
          ] 

"""
models = ["openai/o4-mini-high-2025-04-16",
          "openai/o3-pro-2025-06-10",
          "anthropic/claude-4-sonnet-20250522",
          "google/gemini-2.5-pro",
          "x-ai/grok-3-beta",
          "deepseek/deepseek-chat-v3-0324"
          ] 
"""

# Number of attempts to retrieve correct response
num_attempts = 5

# Max Tokens
MAX_TOKENS = 16*4096
REASONING_MAX_TOKENS = 2*4096

# Timeouts in seconds
DRYRUN_TIMEOUT_S = 3600
TRAIN_TIMEOUT_S  = 99999999
EVAL_TIMEOUT_S   = 3600

# Resources - set to 0 to remove resource constraints
CPU_LIMIT       = 0     #4 
MEMORY_LIMIT_GB = 0     #32
PIDS_LIMIT      = 0     #1024

# All the challenges in the pipeline
challenges = [trackformers_challenge] #[fourtop_challenge]

# Docker Image
DOCKER_IMAGE = "llm-script-sandbox:latest"
DOCKER_TRAINING_IMAGE = "llm-training-sandbox:latest"
DOCKER_EVAL_IMAGE = "llm-evaluation-sandbox:latest"

