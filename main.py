import os
import requests
from fastapi import FastAPI, Request

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT")

if not TELEGRAM_TOKEN or not GEMINI_API_KEY or not SYSTEM_PROMPT:
    raise ValueError("Missing environment variables")

TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.0-flash:generateContent"
)

app = FastAPI()
user_memory = {}

def ask_gemini(user_id: int, message: str) -> str:
    history = user_memory.get(user_id, [])
    history.append({"role": "user", "parts": [{"text": message}]})

    contents = [{"role": "user", "parts": [{"text": SYSTEM_PROMPT}]}] + history

    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0.5,
            "maxOutputTokens": 500
        }
    }

    response = requests.post(
        f"{GEMINI_URL}?key={GEMINI_API_KEY}",
        json=payload,
        timeout=20
    )

    reply = response.json()["candidates"][0]["content"]["parts"][0]["text"]
    history.append({"role": "model", "parts": [{"text": reply}]})
    user_memory[user_id] = history[-10:]  # keep memory short

    return reply

@app.post("/webhook")
async def telegram_webhook(req: Request):
    data = await req.json()

    if "message" not in data:
        return {"ok": True}

    chat_id = data["message"]["chat"]["id"]
    text = data["message"].get("text", "")
    user_id = data["message"]["from"]["id"]

    reply = ask_gemini(user_id, text)

    requests.post(
        f"{TELEGRAM_API}/sendMessage",
        json={"chat_id": chat_id, "text": reply}
    )

    return {"ok": True}
