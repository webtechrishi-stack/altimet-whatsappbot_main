"""
Configuration & Utilities for ARIS Real Estate AI & WhatsApp Platform.
Loads environment variables and validates mandatory credentials on startup.
"""

import os
import sys
from datetime import datetime, timezone
from typing import Any
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


class Config:
    """
    Application Configuration
    """
    PORT = int(os.getenv("PORT", 5000))
    API_VERSION = os.getenv("API_VERSION", "v25.0").strip()
    ACCESS_TOKEN = os.getenv("ACCESS_TOKEN", "").strip()
    PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID", "").strip()
    VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "").strip()

    # Secret key for session signing
    SECRET_KEY = os.getenv("SECRET_KEY", "aris-secret-development-key-2026").strip()

    # Default Admin Credentials
    ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@aris.ai").strip().lower()
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "Admin@Aris2026").strip()

    # AI / LLM Configuration
    GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")).strip()
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip()

    # MongoDB Configuration
    MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017").strip()
    DATABASE_NAME = os.getenv("DATABASE_NAME", "aris_db").strip()

    # WhatsApp Outbound — Test Send Rate Limiting
    WHATSAPP_TEST_SEND_RATE_LIMIT = int(os.getenv("WHATSAPP_TEST_SEND_RATE_LIMIT", "10"))
    WHATSAPP_TEST_SEND_RATE_WINDOW_SECONDS = int(os.getenv("WHATSAPP_TEST_SEND_RATE_WINDOW_SECONDS", "3600"))

    # WhatsApp Outbound — Campaign Throttle
    WHATSAPP_CAMPAIGN_MESSAGES_PER_SECOND = float(os.getenv("WHATSAPP_CAMPAIGN_MESSAGES_PER_SECOND", "1.0"))
    WHATSAPP_CAMPAIGN_BATCH_SIZE = int(os.getenv("WHATSAPP_CAMPAIGN_BATCH_SIZE", "10"))
    WHATSAPP_MAX_RETRY_ATTEMPTS = int(os.getenv("WHATSAPP_MAX_RETRY_ATTEMPTS", "3"))
    CAMPAIGN_SCHEDULER_INTERVAL = int(os.getenv("CAMPAIGN_SCHEDULER_INTERVAL", "30"))
    CAMPAIGN_DELIVERY_WINDOW_START = os.getenv("CAMPAIGN_DELIVERY_WINDOW_START", "09:00").strip()
    CAMPAIGN_DELIVERY_WINDOW_END = os.getenv("CAMPAIGN_DELIVERY_WINDOW_END", "21:00").strip()
    CAMPAIGN_MAX_RETRY_DELAY = int(os.getenv("CAMPAIGN_MAX_RETRY_DELAY", "300"))

    # Redis (optional)
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0").strip()

    # Vector DB / Qdrant Configuration
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost").strip()
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "aris_real_estate_knowledge").strip()
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip()

    # Real Estate Consultant & Cab Service
    CAB_SERVICE_ENABLED = os.getenv("CAB_SERVICE_ENABLED", "true").lower() in ("true", "1", "yes")
    AGENCY_NAME = os.getenv("AGENCY_NAME", "ARIS Real Estate Advisory").strip()
    CONVERSATION_RETENTION_DAYS = int(os.getenv("CONVERSATION_RETENTION_DAYS", "365"))

    # Smart Follow-Up Automation Settings (6-12 hr drip, max 2 nudges, quiet hours)
    FOLLOWUP_MIN_HOURS = float(os.getenv("FOLLOWUP_MIN_HOURS", "6.0"))
    FOLLOWUP_MAX_HOURS = float(os.getenv("FOLLOWUP_MAX_HOURS", "12.0"))
    FOLLOWUP_MAX_COUNT = int(os.getenv("FOLLOWUP_MAX_COUNT", "2"))
    FOLLOWUP_COOLDOWN_HOURS = float(os.getenv("FOLLOWUP_COOLDOWN_HOURS", "24.0"))
    FOLLOWUP_ENABLED = os.getenv("FOLLOWUP_ENABLED", "true").lower() in ("true", "1", "yes")

    @classmethod
    def validate(cls):
        """
        Validates environment variables on startup.
        Exits immediately if any required variable is missing.
        """
        required = {
            "ACCESS_TOKEN": cls.ACCESS_TOKEN,
            "PHONE_NUMBER_ID": cls.PHONE_NUMBER_ID,
            "VERIFY_TOKEN": cls.VERIFY_TOKEN,
        }

        missing = [
            key for key, val in required.items()
            if not val or val.startswith("your_")
        ]

        if missing:
            print("------------------------------------------------")
            print("FATAL ERROR: Missing required environment variables!")
            print(f"Missing: {', '.join(missing)}")
            print("Please configure your .env file properly.")
            print("------------------------------------------------")
            sys.exit(1)


def log_banner(title: str):
    """Prints a styled section separator."""
    print(f"\n========================================")
    print(f" {title}")
    print(f"========================================")


def log_event(event_type: str, details: Any):
    """Logs structured events to console."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{timestamp}] [{event_type}] {details}")
