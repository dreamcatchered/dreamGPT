# dreamGPT

AI assistant — Telegram bot + web interface. Supports local models via Ollama and external AI APIs.

Chat with AI, process documents, images and voice messages from Telegram or a browser.

## Features

- Telegram bot with AI chat
- Voice message transcription
- Image understanding (vision)
- Document and PDF processing
- Web chat interface (Flask)
- Ollama (local LLMs) support
- External AI API support
- DreamID SSO authentication

## Structure

```
dreamGPT/
├── bot.py          # Telegram bot
├── ollama/         # Web UI with Ollama backend
└── web/            # Web UI with external API backend
```

## Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=flat&logo=sqlite&logoColor=white)

## Setup

```bash
pip install -r requirements.txt
```

Set environment variables:
```env
BOT_TOKEN=your_bot_token
API_TOKEN=your_ai_api_token
SSO_CLIENT_SECRET=your_sso_secret
```

```bash
# Telegram bot
python bot.py
# Web UI (Ollama)
python ollama/app.py
# Web UI (external API)
python web/app.py
```

## Contact

Telegram: [@dreamcatch_r](https://t.me/dreamcatch_r)
