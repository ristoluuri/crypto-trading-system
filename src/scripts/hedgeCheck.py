import sys, os
from binance.client import Client
from binance.enums import *
from dotenv import load_dotenv

base_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(base_dir, ".."))
sys.path.append(PROJECT_ROOT)

env_path = os.path.join(PROJECT_ROOT, ".env")
load_dotenv(env_path)

# Replace with your actual API keys
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")
client = Client(api_key, api_secret)

# Check position mode
position_mode = client.futures_get_position_mode()

if position_mode['dualSidePosition']:
    print("You are in Hedge Mode (dualSidePosition = True)")
else:
    print("You are in One-Way Mode (dualSidePosition = False)")