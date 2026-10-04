"""
ARIS Safety Guard, Rate Limiter, and Inbound/Outbound Message Gatekeeper.
Prevents infinite bot-to-bot loops, cuts off dummy/fake numbers from outbound sends,
enforces registered-leads-only outbound, filters inbound messages, and provides
global and per-number toggles for automated AI replies.
"""

import os
import re
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Tuple, Dict, Any, Optional, Set

logger = logging.getLogger(__name__)

# =============================================================================
# 1. OFFICIAL CONTACT INFO & PLATFORM METADATA
# =============================================================================
ALTIMET_SUPPORT_EMAIL = "hello@altimetai.com"
ALTIMET_SUPPORT_WHATSAPP = "8600079496"
ALTIMET_SUPPORT_WHATSAPP_INTL = "+91 8600079496"

# Owner / Admin phone numbers that always have full authorization
OWNER_PHONE_NUMBERS: Set[str] = {
    "918600079496",
    "8600079496",
}

ARIS_PLATFORM_SUMMARY = (
    "ARIS is an AI-powered WhatsApp Real Estate Advisory & Automation Platform "
    "developed by Altimet AI. It assists homebuyers 24/7 with verified property discovery, "
    "floor plans, amenities, and site visit scheduling in Nagpur and Pune."
)

ARIS_CONTACT_INFO_TEXT = (
    f"For detailed information about ARIS or custom AI bot solutions for your business, "
    f"you can connect with our team directly:\n"
    f"📧 *Email:* {ALTIMET_SUPPORT_EMAIL}\n"
    f"💬 *WhatsApp / Call:* {ALTIMET_SUPPORT_WHATSAPP_INTL} ({ALTIMET_SUPPORT_WHATSAPP})"
)


# =============================================================================
# 2. HARDCODED BLACKLIST & DUMMY NUMBER PATTERNS
# =============================================================================
KNOWN_BLACKLISTED_NUMBERS: Set[str] = {
    # Domino's bot that caused the 64,000 loop
    "919160001286",
    # Test numbers found across test files
    "919833392058",
    "919876543210",
    "919876543211",
    "919876543299",
    "919876543255",
    "919876543266",
    "919876543277",
    "919000000001",
    "919000000002",
    "919999988888",
    "9199990b9500",
    "919822334455",
    "919766554433",
}

DUMMY_PATTERNS = [
    r"^(?:91)?0{8,}$",                # All zeros: 0000000000
    r"^(?:91)?1{8,}$",                # All ones: 1111111111
    r"^(?:91)?9{8,}$",                # All nines: 9999999999
    r"^(?:91)?1234567890$",           # Sequential
    r"^(?:91)?9876543210$",           # Sequential
    r"^(?:91)?98765432\d{2}$",        # Test batch 98765432xx
    r"^(?:91)?90000000\d{2}$",        # Test batch 90000000xx
]


def normalize_clean_phone(raw: str) -> str:
    """Strips all non-digit characters and standardizes country code."""
    if not raw:
        return ""
    digits = re.sub(r"\D", "", str(raw))
    if digits.startswith("0") and len(digits) == 11:
        digits = "91" + digits[1:]
    elif len(digits) == 10 and digits[0] in "6789":
        digits = "91" + digits
    return digits


def is_dummy_or_fake_number(phone: str) -> Tuple[bool, str]:
    """
    Returns (True, reason) if the number is a fake, test, dummy, or blacklisted phone.
    Owner numbers are always valid.
    """
    clean = normalize_clean_phone(phone)
    if not clean:
        return True, "EMPTY_PHONE"

    if clean in OWNER_PHONE_NUMBERS:
        return False, "OWNER_PHONE"

    if clean in KNOWN_BLACKLISTED_NUMBERS:
        return True, f"BLACKLISTED_NUMBER ({clean})"

    if len(clean) < 10 or len(clean) > 15:
        return True, f"INVALID_PHONE_LENGTH ({len(clean)} digits)"

    for pat in DUMMY_PATTERNS:
        if re.search(pat, clean):
            return True, f"DUMMY_PATTERN_MATCH ({clean})"

    if clean.startswith("91") and len(clean) == 12:
        lead_digit = clean[2]
        if lead_digit not in "6789":
            return True, f"INVALID_INDIAN_MOBILE_PREFIX (starts with {lead_digit})"

    return False, "VALID"


# =============================================================================
# 3. GLOBAL & PER-NUMBER AI AUTO-REPLY CONTROLS
# =============================================================================
_GLOBAL_AI_AUTOREPLY: bool = os.environ.get("ENABLE_AI_AUTOREPLY", "true").lower() in ("true", "1")
STRICT_REGISTERED_LEADS_ONLY: bool = os.environ.get("STRICT_REGISTERED_LEADS_ONLY", "true").lower() in ("true", "1")


def is_global_ai_autoreply_enabled() -> bool:
    """Returns True if AI is permitted to send automated replies across the platform."""
    global _GLOBAL_AI_AUTOREPLY
    try:
        from database import DB
        if DB.webhooks._db.is_connected() and DB.webhooks._db.db is not None:
            doc = DB.webhooks._db.db["system_settings"].find_one({"key": "global_ai_autoreply"})
            if doc and "enabled" in doc:
                _GLOBAL_AI_AUTOREPLY = bool(doc["enabled"])
    except Exception:
        pass
    return _GLOBAL_AI_AUTOREPLY


def set_global_ai_autoreply_enabled(enabled: bool) -> bool:
    """Sets global automated AI reply toggle in memory and MongoDB."""
    global _GLOBAL_AI_AUTOREPLY
    _GLOBAL_AI_AUTOREPLY = bool(enabled)
    try:
        from database import DB
        if DB.webhooks._db.is_connected() and DB.webhooks._db.db is not None:
            DB.webhooks._db.db["system_settings"].update_one(
                {"key": "global_ai_autoreply"},
                {"$set": {"key": "global_ai_autoreply", "enabled": _GLOBAL_AI_AUTOREPLY, "updated_at": datetime.now(timezone.utc)}},
                upsert=True
            )
    except Exception as ex:
        logger.warning(f"[SETTINGS ERROR] Could not save global_ai_autoreply: {ex}")
    return _GLOBAL_AI_AUTOREPLY


def is_ai_enabled_for_number(phone: str) -> bool:
    """
    Checks if AI auto-reply is active for a specific phone number.
    Returns False if global AI is off or if the conversation/lead has ai_disabled / human_takeover set.
    """
    if not is_global_ai_autoreply_enabled():
        return False

    clean = normalize_clean_phone(phone)
    try:
        from database import DB
        if DB.conversations._db.is_connected():
            conv = DB.conversations._db.conversations.find_one({
                "$or": [{"wa_id": clean}, {"wa_id": clean[-10:]}]
            })
            if conv:
                if conv.get("human_takeover") or conv.get("ai_disabled") or conv.get("opted_out") or conv.get("ai_reply_enabled") is False:
                    return False

            lead = DB.leads._db.leads.find_one({
                "$or": [{"wa_id": clean}, {"phone": clean}, {"phone": clean[-10:]}]
            })
            if lead and (lead.get("ai_disabled") or lead.get("opted_out") or lead.get("ai_reply_enabled") is False):
                return False
    except Exception:
        pass

    return True


def set_ai_enabled_for_number(phone: str, enabled: bool) -> bool:
    """
    Enables or disables automated AI replies for a specific phone number.
    Updates conversation and lead records in MongoDB.
    """
    clean = normalize_clean_phone(phone)
    if not clean:
        return False
    try:
        from database import DB
        conv_updates = {
            "ai_disabled": not enabled,
            "ai_reply_enabled": enabled,
            "human_takeover": not enabled,
            "updated_at": datetime.now(timezone.utc)
        }
        lead_updates = {
            "ai_disabled": not enabled,
            "ai_reply_enabled": enabled,
            "updated_at": datetime.now(timezone.utc)
        }
        if DB.conversations._db.is_connected():
            DB.conversations._db.conversations.update_many(
                {"$or": [{"wa_id": clean}, {"wa_id": clean[-10:]}]},
                {"$set": conv_updates}
            )
            DB.leads._db.leads.update_many(
                {"$or": [{"wa_id": clean}, {"phone": clean}, {"phone": clean[-10:]}]},
                {"$set": lead_updates}
            )
        return True
    except Exception as ex:
        logger.error(f"[SET AI NUMBER ERROR] {ex}")
        return False


def is_registered_lead(phone: str) -> bool:
    """
    Verifies that a phone number is an authorized contact in the system:
    - Owner/Admin number
    - Added manually via CRM / API (`source="MANUAL"`, `"UI_ADMIN"`, etc.)
    - Uploaded via Excel Lead Import (`source="CAMPAIGN_IMPORT"`, `"EXCEL_IMPORT"`)
    - Existing campaign recipient
    """
    clean = normalize_clean_phone(phone)
    if not clean:
        return False

    if clean in OWNER_PHONE_NUMBERS:
        return True

    if is_dummy_or_fake_number(clean)[0]:
        return False

    try:
        from database import DB
        if DB.leads._db.is_connected():
            lead = DB.leads._db.leads.find_one({
                "$or": [{"wa_id": clean}, {"phone": clean}, {"phone": clean[-10:]}]
            })
            if lead:
                return True

            recip = DB.campaign_recipients._db.campaign_recipients.find_one({
                "$or": [{"phone": clean}, {"phone": clean[-10:]}]
            })
            if recip:
                return True
    except Exception as ex:
        logger.warning(f"[REGISTERED CHECK ERROR] {ex}")

    return False


# =============================================================================
# 4. RATE LIMITER & CIRCUIT BREAKER
# =============================================================================
class SafetyRateLimiter:
    """
    Multi-tier rate limiter protecting both Meta Cloud API costs and server stability:
    1. Debounce: Minimum 4 seconds between replies to the same contact (stops rapid ping-pong).
    2. Hourly Cap: Max 10 automated replies per phone number per hour.
    3. Daily Cap: Max 25 automated replies per phone number per 24 hours.
    4. Global Outbound Circuit Breaker: Max 60 outbound calls per minute across the entire server.
    """
    def __init__(
        self,
        min_debounce_sec: float = 4.0,
        max_user_hourly: int = 10,
        max_user_daily: int = 25,
        max_global_per_minute: int = 60
    ):
        self.min_debounce_sec = min_debounce_sec
        self.max_user_hourly = max_user_hourly
        self.max_user_daily = max_user_daily
        self.max_global_per_minute = max_global_per_minute

        self._last_outbound_time: Dict[str, float] = {}
        self._user_hourly_history: Dict[str, list] = {}
        self._user_daily_history: Dict[str, list] = {}
        self._global_outbound_history: list = []

    def check_outbound(self, recipient: str) -> Tuple[bool, str]:
        clean = normalize_clean_phone(recipient)
        is_dummy, dummy_reason = is_dummy_or_fake_number(clean)
        if is_dummy:
            return False, f"CUTOFF_DUMMY_NUMBER: {dummy_reason}"

        now = time.time()

        cutoff_min = now - 60.0
        self._global_outbound_history = [t for t in self._global_outbound_history if t > cutoff_min]
        if len(self._global_outbound_history) >= self.max_global_per_minute:
            return False, f"GLOBAL_RATE_LIMIT_EXCEEDED ({len(self._global_outbound_history)}/min)"

        last_time = self._last_outbound_time.get(clean, 0.0)
        if (now - last_time) < self.min_debounce_sec:
            return False, f"DEBOUNCE_ACTIVE (wait {self.min_debounce_sec - (now - last_time):.1f}s)"

        cutoff_hour = now - 3600.0
        user_h = [t for t in self._user_hourly_history.get(clean, []) if t > cutoff_hour]
        self._user_hourly_history[clean] = user_h
        if len(user_h) >= self.max_user_hourly:
            return False, f"USER_HOURLY_LIMIT_EXCEEDED ({len(user_h)}/{self.max_user_hourly})"

        cutoff_day = now - 86400.0
        user_d = [t for t in self._user_daily_history.get(clean, []) if t > cutoff_day]
        self._user_daily_history[clean] = user_d
        if len(user_d) >= self.max_user_daily:
            return False, f"USER_DAILY_LIMIT_EXCEEDED ({len(user_d)}/{self.max_user_daily})"

        return True, "OK"

    def record_outbound(self, recipient: str):
        clean = normalize_clean_phone(recipient)
        now = time.time()
        self._last_outbound_time[clean] = now
        self._global_outbound_history.append(now)

        self._user_hourly_history.setdefault(clean, []).append(now)
        self._user_daily_history.setdefault(clean, []).append(now)


SAFETY_RATE_LIMITER = SafetyRateLimiter()


# =============================================================================
# 5. INBOUND RELEVANCE & BOT-LOOP FILTER
# =============================================================================
AUTOMATED_BOT_PATTERNS = [
    r"\b(?:welcome to domino|domino'?s|pizza|hut|burger king|mcdonald'?s)\b",
    r"\b(?:swiggy|zomato|blinkit|zepto|instamart|dunzo|deliver(?:y|ed)|track your order|order id)\b",
    r"\b(?:uber|ola|rapido|driver is on|arriving in \d+ min)\b",
    r"\b(?:otp\b|one time password|verification code|security code|do not share this otp)\b",
    r"\b(?:automated (?:message|response|reply)|do not reply to this (?:message|number))\b",
    r"\b(?:press \d to|type \d to|reply with \d to|select an option below)\b",
    r"\b(?:reply stop to unsubscribe|unsubscribe from these alerts|terms and conditions apply)\b",
    r"\b(?:feeling hungry\?|order now|cashback|flat \d+% off on food)\b",
]

SPAM_PATTERNS = [
    r"\b(?:crypto|bitcoin|forex|earn \d+|earn money|from home click|lucky draw|lottery|casino|betting|win free)\b",
    r"\b(?:bit\.ly|tinyurl|t\.me|telegram channel|whatsapp group link)\b",
]

REAL_ESTATE_PATTERNS = [
    r"\b(?:flat|flats|property|properties|apartment|apartments|villa|villas|plot|plots|house|houses|bungalow)\b",
    r"\b(?:dream home|new home|residential property|buy home|gated community)\b",
    r"\b(?:1\s*bhk|2\s*bhk|3\s*bhk|4\s*bhk|studio|penthouse|row house|duplex)\b",
    r"\b(?:buy|buying|purchase|rent|renting|lease|sell|selling|invest|investment|commercial|residential)\b",
    r"\b(?:budget|price|pricing|cost|rate|lakh|lakhs|lac|lacs|cr|crore|crores|emi|loan)\b",
    r"\b(?:nagpur|pune|besa|mihan|manish nagar|wardha road|dharampeth|ramdaspeth|trimurti nagar)\b",
    r"\b(?:kharadi|baner|wakad|hinjewadi|viman nagar|hadapsar|kothrud|bavdhan|ravet|punawale)\b",
    r"\b(?:visit|site visit|see the flat|dekhna hai|location|address|map|direction|metro)\b",
    r"\b(?:brochure|floor plan|amenities|layout|parking|swimming pool|gym|clubhouse|possession|rera)\b",
    r"\b(?:ready to move|under construction|builder|developer|aris|aris premier)\b",
    r"\b(?:ghar|makaan|kholya|flat chahiye|kahan hai|rate kya hai|kitne me)\b",
]

BOT_SERVICE_PATTERNS = [
    r"\b(?:what is aris|who is aris|what does aris do|who are you|who built you|who made you)\b",
    r"\b(?:altimet|altimet ai|altimetai)\b",
    r"\b(?:whatsapp bot|ai bot|ai agent|chatbot|automation service|bot service|saas|api|ai service|ai services)\b",
    r"\b(?:contact (?:us|you|team)|email|phone number|call me|support|customer care)\b",
    r"\b(?:how (?:does this|do you) work|what do you do|services offered|what services|services you give|services u gave|services u give|what you offer)\b",
    r"\b(?:aap kaun ho|kon ho|kya karte ho|kya service hai)\b",
]

NATURAL_GREETINGS = [
    r"^(?:hi+|hello+|hey+|hii+|namaste|good\s*(?:morning|afternoon|evening)|hola|salaam)\b"
]

OPT_OUT_PATTERNS = [
    r"\b(?:don'?t (?:message|text|contact|call|msg)|stop|unsubscribe|leave me alone)\b",
    r"\b(?:msg mat karo|message mat bhejo|mat bhejo|nahi chahiye|ab message mat karna)\b",
]


def classify_inbound_message(message_text: str, sender: str) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Evaluates an incoming message to determine if it is genuine and relevant.
    Returns:
        (should_reply: bool, reason_code: str, metadata: Dict[str, Any])
    """
    clean_sender = normalize_clean_phone(sender)

    # 1. Check if AI is disabled globally or for this specific number
    if not is_global_ai_autoreply_enabled():
        return False, "GLOBAL_AI_AUTOREPLY_DISABLED", {"category": "GLOBAL_OFF"}

    if not is_ai_enabled_for_number(clean_sender):
        return False, "AI_DISABLED_FOR_RECIPIENT", {"category": "USER_OFF"}

    # 2. Blacklist & Dummy check
    if clean_sender in KNOWN_BLACKLISTED_NUMBERS:
        return False, "BLACKLISTED_SENDER", {"category": "BLACKLISTED"}

    is_dummy, dummy_reason = is_dummy_or_fake_number(clean_sender)
    if is_dummy:
        return False, f"DUMMY_SENDER: {dummy_reason}", {"category": "DUMMY"}

    text = (message_text or "").strip()
    if not text:
        return False, "EMPTY_MESSAGE", {"category": "EMPTY"}

    lower = text.lower()

    # 3. Check for Automated Bot Signatures (Domino's, OTPs, delivery alerts)
    for pat in AUTOMATED_BOT_PATTERNS:
        if re.search(pat, lower):
            KNOWN_BLACKLISTED_NUMBERS.add(clean_sender)
            return False, "AUTOMATED_BOT_DETECTED", {"category": "BOT_PING_PONG", "pattern": pat}

    # 4. Check for unsolicited spam / scam links
    for pat in SPAM_PATTERNS:
        if re.search(pat, lower):
            KNOWN_BLACKLISTED_NUMBERS.add(clean_sender)
            return False, "SPAM_DETECTED", {"category": "SPAM", "pattern": pat}

    # 5. Always permit Opt-Out messages
    for pat in OPT_OUT_PATTERNS:
        if re.search(pat, lower):
            return True, "OPT_OUT", {"category": "OPT_OUT"}

    # 6. Check for ARIS / Bot Service inquiries
    is_bot_service = any(re.search(pat, lower) for pat in BOT_SERVICE_PATTERNS)
    if is_bot_service:
        wants_contact = any(k in lower for k in [
            "contact", "email", "phone", "reach", "number", "talk to human", "sales", "details",
            "business", "pricing", "cost", "partnership", "hire"
        ])
        return True, "BOT_SERVICE_INQUIRY", {
            "category": "BOT_SERVICE",
            "wants_contact": wants_contact,
            "give_contact_info": wants_contact
        }

    # 7. Check for Real Estate inquiries
    is_real_estate = any(re.search(pat, lower) for pat in REAL_ESTATE_PATTERNS)
    if is_real_estate:
        return True, "REAL_ESTATE_INQUIRY", {"category": "REAL_ESTATE"}

    # 8. Check for natural human greetings
    is_greeting = any(re.search(pat, lower) for pat in NATURAL_GREETINGS)
    if is_greeting:
        return True, "HUMAN_GREETING", {"category": "GREETING"}

    # 9. All other genuine customer messages: ALWAYS ALLOW & PROCESS!
    # Real customer messages should NEVER be silently ignored.
    return True, "GENUINE_USER_MESSAGE", {"category": "GENUINE"}


# =============================================================================
# 6. OUTBOUND GATEWAY FILTER
# =============================================================================
def is_outbound_allowed(recipient: str, context: Optional[str] = None) -> Tuple[bool, str]:
    """
    Final authorization gate before sending any WhatsApp API request.
    Strictly prevents outbound to dummy/fake numbers and enforces that outbound
    is only permitted to numbers explicitly registered via manual lead or Excel import.
    """
    clean = normalize_clean_phone(recipient)
    if not clean:
        return False, "EMPTY_RECIPIENT"

    # Owner / Admin number is always authorized
    if clean in OWNER_PHONE_NUMBERS:
        return True, "OWNER_PHONE"

    # Strict check on dummy/fake numbers
    is_dummy, reason = is_dummy_or_fake_number(clean)
    if is_dummy:
        logger.warning(f"[SAFETY_GUARD BLOCKED] Outbound to dummy number blocked: {clean} ({reason})")
        return False, f"BLOCKED: {reason}"

    # Strict Registered Leads Check: Outbound only allowed to manually added or Excel imported numbers
    if STRICT_REGISTERED_LEADS_ONLY and context != "test_suite":
        if not is_registered_lead(clean):
            logger.warning(f"[SAFETY_GUARD BLOCKED] Unregistered recipient blocked from outbound: {clean}")
            return False, "UNREGISTERED_RECIPIENT: Outbound only permitted to numbers added via Manual Lead or Excel import."

    # Rate limiting check
    allowed, rate_reason = SAFETY_RATE_LIMITER.check_outbound(clean)
    if not allowed:
        logger.warning(f"[SAFETY_GUARD BLOCKED] Outbound rate limit hit for {clean}: {rate_reason}")
        return False, f"RATE_LIMITED: {rate_reason}"

    return True, "OK"


def record_outbound_send(recipient: str):
    """Call after an outbound message is successfully dispatched."""
    SAFETY_RATE_LIMITER.record_outbound(recipient)
