import os
from openai import OpenAI
from dotenv import load_dotenv

######################################## OpenAI ########################################
# Load environment variables
load_dotenv(override=True)

# Get OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found in environment variables")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


######################################## variables ########################################
# Path to the trained model
MODEL_PATH = "../models/best_model.pkl"

# Encoder mapping for 'Environment'
ENVIRONMENT_MAPPING = {"Arctic": 0, "Desert": 1, "Onshore": 2, "Offshore": 3}