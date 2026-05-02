import os
import requests
import json

from dotenv import load_dotenv


load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Set OPENROUTER_API_KEY (or OPENAI_API_KEY) in backend/.env")

response = requests.get(
  url="https://openrouter.ai/api/v1/key",
  headers={
    "Authorization": f"Bearer {api_key}"
  }
)

print(json.dumps(response.json(), indent=2))
