# utils/telegram_notifier.py
import os
import httpx
from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from utils.logger import logger

TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

def send_telegram_message(text: str):
    """发送 Telegram 消息（异步非阻塞）"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return  # 未配置则静默跳过

    try:
        with httpx.Client(timeout=10) as client:
            client.post(TELEGRAM_API, data={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": f"🤖 AI Butterfly\n\n{text}",
                "parse_mode": "HTML"
            })
        logger.info("✅ Telegram 消息已发送")
    except Exception as e:
        logger.warning(f"⚠️ Telegram 发送失败: {e}")