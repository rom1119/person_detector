from dotenv import load_dotenv
import os


load_dotenv()

RTSP_URL = os.getenv("RTSP_URL")

REOLINK_IP = os.getenv("REOLINK_IP")
REOLINK_USER = os.getenv("REOLINK_USER")
REOLINK_PASSWORD = os.getenv("REOLINK_PASSWORD")

BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
