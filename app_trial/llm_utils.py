import os
import re
import json
import httpx
from dotenv import load_dotenv
from google import genai

load_dotenv()

MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "gemini").lower()  # gemini / openai / aipipe
GENAI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")

genai.configure(api_key=GENAI_API_KEY)


def extract_json_from_response(response_text: str):
    match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    else:
        json_str = response_text.strip()
    return json.loads(json_str)


async def call_llm(prompt: str, provider: str = MODEL_PROVIDER, model: str = None) -> str:
    try:
        if provider == "gemini":
            model = model or "models/gemini-pro"
            chat = genai.GenerativeModel(model).start_chat()
            response = chat.send_message(prompt)
            return response.text

        elif provider == "openai":
            model = model or "gpt-4o"
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}]
            }
            async with httpx.AsyncClient() as client:
                res = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                res.raise_for_status()
                return res.json()['choices'][0]['message']['content']

        elif provider == "aipipe":
            model = model or "gpt-4o-mini"
            headers = {
                "Authorization": f"Bearer {AIPIPE_TOKEN}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}]
            }
            async with httpx.AsyncClient(verify=False) as client:
                res = await client.post("https://aipipe.org/openai/v1/chat/completions", headers=headers, json=payload)
                res.raise_for_status()
                return res.json()['choices'][0]['message']['content']

        else:
            raise ValueError("Invalid MODEL_PROVIDER. Choose from gemini, openai, aipipe.")

    except Exception as e:
        print(f"LLM call failed: {e}")
        raise
