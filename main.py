import os
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ========================
# Environment Variables
# ========================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a professional product manager. Answer in a calm, structured, and actionable style.")

if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN not set")

# ========================
# Load DeepSeek-R1 / Distill LLaMA Model Locally
# ========================
MODEL_NAME = "DeepSeek-R1"  # Or local path if uploaded

print("Loading DeepSeek-R1 model locally. This may take a minute...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print("Model loaded successfully!")

# ========================
# Conversation Memory
# ========================
user_histories = {}

# ========================
# Local Model Function
# ========================
def ask_local(user_id: int, user_input: str) -> str:
    if user_id not in user_histories:
        user_histories[user_id] = [SYSTEM_PROMPT]

    # Add user message
    user_histories[user_id].append(user_input)

    # Prepare prompt
    prompt = "\n".join(user_histories[user_id]) + "\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7)

    reply = tokenizer.decode(output[0], skip_special_tokens=True)
    # Keep only assistant reply
    reply = reply.replace(prompt, "").strip()

    # Save assistant reply
    user_histories[user_id].append(reply)
    return reply

# ========================
# Telegram Handlers
# ========================
def start(update: Update, context: CallbackContext):
    update.message.reply_text(
        "Hello! I'm June, your PM assistant. Ask me anything about product strategy, roadmaps, or execution."
    )

def handle_message(update: Update, context: CallbackContext):
    user_id = update.message.from_user.id
    user_input = update.message.text

    reply = ask_local(user_id, user_input)
    update.message.reply_text(reply)

# ========================
# Main Function
# ========================
def main():
    updater = Updater(TELEGRAM_TOKEN)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

    print("June Telegram Bot is running locally on DeepSeek-R1...")
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
