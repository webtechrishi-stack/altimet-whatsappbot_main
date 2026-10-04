"""
ARIS WhatsApp Engine — Unified Meta Cloud API (v25.0) Client, Outbound Service & Campaign Worker.
Consolidates:
- WhatsAppClient with typed exception hierarchy & payload builders
- Phone normalization (E.164) & phone masking
- WhatsApp Template Registry (approved Meta templates)
- Rate Limiter for test sends (MongoDB-backed window)
- OutboundService (single unified send service)
- CampaignWorker (async rate-limited campaign execution)
- Legacy WhatsAppService wrapper for backward compatibility
"""

import os
import sys
import re
import time
import json
import uuid
import logging
import requests
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple

from config import Config


def safe_terminal_log(message: str):
    """
    Safely prints structured logs to stdout and logging system,
    preventing charmap/UnicodeEncodeError on Windows console.
    """
    try:
        encoding = getattr(sys.stdout, "encoding", "utf-8") or "utf-8"
        clean = message.encode(encoding, errors="replace").decode(encoding)
        print(clean, flush=True)
    except Exception:
        print(re.sub(r"[^\x00-\x7F]+", " ", message), flush=True)
    try:
        logger.info(re.sub(r"[^\x00-\x7F]+", " ", message))
    except Exception:
        pass
from database import (
    DB,
    get_db,
    OutboundMessage,
    OutboundStatus,
    OutboundSource,
    OutboundMessageRepository,
    Campaign,
    CampaignStatus,
    CampaignRepository,
    CampaignRecipient,
    CampaignRecipientRepository,
    RecipientStatus,
    LeadRepository,
    EventRepository,
)

logger = logging.getLogger(__name__)


# =====================================================================
# 1. Phone Normalization Service
# =====================================================================

_INDIA_PREFIX = "91"
_DEFAULT_COUNTRY = "IN"


def _strip_to_digits(raw: str) -> str:
    return re.sub(r"\D", "", str(raw or ""))


def normalize_phone(raw_phone: str, default_country: str = "IN") -> Tuple[str, str, str]:
    """
    Normalizes a phone number to Meta WhatsApp API format (digits only, with country code).
    Returns (raw_phone, normalized_phone, country_code).
    """
    raw = str(raw_phone or "").strip()
    if not raw:
        return ("", "", "")

    digits = _strip_to_digits(raw)
    if not digits:
        raise ValueError(f"No digits found in phone number: {raw!r}")

    # Country prefix logic
    if raw.startswith("+"):
        normalized = digits
    elif digits.startswith("0") and len(digits) == 11:
        normalized = _INDIA_PREFIX + digits[1:]
    elif len(digits) == 10 and default_country == "IN" and digits.startswith(("6", "7", "8", "9")):
        normalized = _INDIA_PREFIX + digits
    else:
        normalized = digits

    if len(normalized) < 7 or len(normalized) > 15:
        raise ValueError(
            f"Normalized phone number has invalid length ({len(normalized)} digits): {normalized!r}. Expected 7-15 digits."
        )

    # Infer country
    if normalized.startswith("91") and len(normalized) == 12:
        country_code = "IN"
    elif normalized.startswith("1") and len(normalized) == 11:
        country_code = "US"
    elif normalized.startswith("44") and len(normalized) == 12:
        country_code = "GB"
    elif normalized.startswith("971"):
        country_code = "AE"
    elif normalized.startswith("65"):
        country_code = "SG"
    else:
        country_code = "UNKNOWN"

    return raw, normalized, country_code


def mask_phone(normalized: str) -> str:
    """Masks phone number for safe logging: 919876543210 -> +91 9876***210"""
    if not normalized or len(normalized) < 8:
        return "***"
    return f"+{normalized[:2]} {normalized[2:6]}***{normalized[-3:]}"


class NormalizedPhone(dict):
    def __str__(self) -> str:
        return self.get("normalized_phone", "")

    def __repr__(self) -> str:
        return self.get("normalized_phone", "")

    def __eq__(self, other):
        if isinstance(other, str):
            return self.get("normalized_phone", "") == other
        return super().__eq__(other)


class PhoneService:
    def __init__(self, default_country: str = "IN"):
        self.default_country = default_country

    def normalize(self, raw_phone: str) -> NormalizedPhone:
        raw, normalized, country_code = normalize_phone(raw_phone, self.default_country)
        return NormalizedPhone({
            "raw_phone": raw,
            "normalized_phone": normalized,
            "country_code": country_code,
            "masked": mask_phone(normalized) if normalized else "",
            "is_valid": bool(normalized),
        })

    def is_valid(self, raw_phone: str) -> bool:
        try:
            res = self.normalize(raw_phone)
            return bool(res.get("normalized_phone"))
        except Exception:
            return False

    def validate(self, raw_phone: str) -> bool:
        return self.is_valid(raw_phone)


# =====================================================================
# 2. Typed WhatsApp Exceptions
# =====================================================================

class WhatsAppBaseError(Exception):
    code: str = "WHATSAPP_ERROR"
    http_status: int = 500
    safe_message: str = "WhatsApp service encountered an error."

    def __init__(self, message: str = "", meta_code: int = 0, meta_subcode: int = 0):
        super().__init__(message)
        self.internal_message = message
        self.meta_code = meta_code
        self.meta_subcode = meta_subcode

    def to_api_response(self) -> dict:
        return {"code": self.code, "message": self.safe_message}


class WhatsAppAuthenticationError(WhatsAppBaseError):
    code = "WHATSAPP_AUTH_ERROR"
    http_status = 401
    safe_message = "WhatsApp API authentication failed. Check server configuration."


class WhatsAppRateLimitError(WhatsAppBaseError):
    code = "WHATSAPP_RATE_LIMIT"
    http_status = 429
    safe_message = "WhatsApp message rate limit reached. Please try again later."


class WhatsAppTemplateError(WhatsAppBaseError):
    code = "WHATSAPP_TEMPLATE_ERROR"
    http_status = 422
    safe_message = "WhatsApp template configuration error. Verify template name and parameters."

    def __init__(self, message: str = "", meta_code: int = 0, meta_subcode: int = 0):
        super().__init__(message, meta_code, meta_subcode)
        if message:
            self.safe_message = f"Meta Template Error ({meta_code}): {message}"


class WhatsAppRecipientError(WhatsAppBaseError):
    code = "WHATSAPP_RECIPIENT_ERROR"
    http_status = 422
    safe_message = "Recipient phone number is invalid or not registered on WhatsApp."


class WhatsAppTransientError(WhatsAppBaseError):
    code = "WHATSAPP_TRANSIENT_ERROR"
    http_status = 503
    safe_message = "WhatsApp service temporarily unavailable. Retry scheduled."


class WhatsAppSendError(WhatsAppBaseError):
    code = "WHATSAPP_SEND_FAILED"
    http_status = 500
    safe_message = "Unable to send WhatsApp message. Contact support if this persists."


class WhatsAppConfigError(WhatsAppBaseError):
    code = "WHATSAPP_CONFIG_ERROR"
    http_status = 500
    safe_message = "WhatsApp API is not configured on this server."


def map_meta_error(
    http_status: int,
    error_code: int = 0,
    error_subcode: int = 0,
    error_message: str = ""
) -> WhatsAppBaseError:
    if http_status == 401 or error_code in (190, 102, 10):
        return WhatsAppAuthenticationError(error_message, error_code, error_subcode)
    if http_status == 429 or error_code in (80007, 131056, 130429):
        return WhatsAppRateLimitError(error_message, error_code, error_subcode)
    if error_code in (132000, 132001, 132005, 132007, 132012, 132015, 132016):
        return WhatsAppTemplateError(error_message, error_code, error_subcode)
    if error_code in (131030, 131026, 100) and "phone" in error_message.lower():
        return WhatsAppRecipientError(error_message, error_code, error_subcode)
    if error_code in (131026,):
        return WhatsAppRecipientError(error_message, error_code, error_subcode)
    if http_status in (500, 502, 503, 504) or error_code in (131042, 131045, 1):
        return WhatsAppTransientError(error_message, error_code, error_subcode)
    return WhatsAppSendError(error_message, error_code, error_subcode)


# =====================================================================
# 3. Payload Construction Helpers
# =====================================================================

def sanitize_recipient(recipient: str) -> str:
    return re.sub(r"\D", "", str(recipient or ""))


def build_text_payload(recipient: str, body: str, preview_url: bool = False) -> Dict[str, Any]:
    return {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": sanitize_recipient(recipient),
        "type": "text",
        "text": {"preview_url": preview_url, "body": body}
    }


def build_buttons_payload(
    recipient: str,
    body_text: str,
    buttons: List[Dict[str, str]],
    header_text: Optional[str] = None,
    footer_text: Optional[str] = None
) -> Dict[str, Any]:
    action_buttons = []
    for btn in buttons[:3]:
        action_buttons.append({
            "type": "reply",
            "reply": {
                "id": btn.get("id", f"btn_{len(action_buttons)}"),
                "title": btn.get("title", "Select")[:20]
            }
        })

    interactive: Dict[str, Any] = {
        "type": "button",
        "body": {"text": body_text},
        "action": {"buttons": action_buttons}
    }
    if header_text:
        interactive["header"] = {"type": "text", "text": header_text}
    if footer_text:
        interactive["footer"] = {"text": footer_text}

    return {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": sanitize_recipient(recipient),
        "type": "interactive",
        "interactive": interactive
    }


def build_list_payload(
    recipient: str,
    body_text: str,
    button_label: str,
    sections: List[Dict[str, Any]],
    title: Optional[str] = None,
    footer_text: Optional[str] = None
) -> Dict[str, Any]:
    interactive: Dict[str, Any] = {
        "type": "list",
        "body": {"text": body_text},
        "action": {"button": button_label[:20], "sections": sections}
    }
    if title:
        interactive["header"] = {"type": "text", "text": title}
    if footer_text:
        interactive["footer"] = {"text": footer_text}

    return {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": sanitize_recipient(recipient),
        "type": "interactive",
        "interactive": interactive
    }


def build_document_payload(
    recipient: str,
    document_url: str,
    filename: str,
    caption: Optional[str] = None
) -> Dict[str, Any]:
    doc: Dict[str, Any] = {"link": document_url, "filename": filename}
    if caption:
        doc["caption"] = caption
    return {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": sanitize_recipient(recipient),
        "type": "document",
        "document": doc
    }


def build_image_payload(recipient: str, image_url: str, caption: Optional[str] = None) -> Dict[str, Any]:
    img: Dict[str, Any] = {"link": image_url}
    if caption:
        img["caption"] = caption
    return {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": sanitize_recipient(recipient),
        "type": "image",
        "image": img
    }


def build_location_payload(
    recipient: str,
    latitude: float,
    longitude: float,
    name: str,
    address: str
) -> Dict[str, Any]:
    return {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": sanitize_recipient(recipient),
        "type": "location",
        "location": {
            "latitude": latitude,
            "longitude": longitude,
            "name": name,
            "address": address
        }
    }


def build_template_payload(
    recipient: str,
    template_name: str,
    language_code: str = "en",
    components: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    template: Dict[str, Any] = {
        "name": template_name,
        "language": {"code": language_code}
    }
    if components:
        template["components"] = components

    return {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": sanitize_recipient(recipient),
        "type": "template",
        "template": template
    }


# =====================================================================
# 4. Meta WhatsApp Cloud API Client
# =====================================================================

@dataclass
class WhatsAppSendResult:
    success: bool
    meta_message_id: str = ""
    messaging_product: str = "whatsapp"
    raw_response: Dict[str, Any] = field(default_factory=dict)


class WhatsAppClient:
    """
    Unified client for Meta WhatsApp Cloud API (v25.0+).
    Raises typed exceptions on failure. Never leaks access tokens to logs.
    """

    def __init__(
        self,
        access_token: Optional[str] = None,
        phone_number_id: Optional[str] = None,
        api_version: Optional[str] = None
    ):
        self.access_token = (access_token or Config.ACCESS_TOKEN or "").strip()
        self.phone_number_id = (phone_number_id or Config.PHONE_NUMBER_ID or "").strip()
        self.api_version = (api_version or Config.API_VERSION or "v25.0").strip()
        self.base_url = "https://graph.facebook.com"
        self.url = f"{self.base_url}/{self.api_version}/{self.phone_number_id}/messages"

    @property
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }

    def _validate_config(self):
        if not self.access_token or not self.phone_number_id:
            raise WhatsAppConfigError(
                "ACCESS_TOKEN or PHONE_NUMBER_ID missing from environment configuration."
            )

    def _dispatch(self, payload: Dict[str, Any]) -> WhatsAppSendResult:
        self._validate_config()
        to_number = payload.get("to", "UNKNOWN")

        logger.info(
            "[WA_CLIENT] Dispatching %s message to %s..%s via phone_id %s",
            payload.get("type", "unknown"),
            to_number[:4] if len(to_number) > 4 else "****",
            to_number[-3:] if len(to_number) > 3 else "***",
            self.phone_number_id,
        )

        # Safety Guard Cutoff: Block dummy/fake numbers & enforce rate limits
        try:
            from safety_guard import is_outbound_allowed, record_outbound_send
            allowed, block_reason = is_outbound_allowed(to_number)
            if not allowed:
                logger.warning(
                    "[OUTBOUND CUTOFF BLOCKED] Suppressed Meta API call to %s: %s",
                    to_number, block_reason
                )
                return WhatsAppSendResult(
                    success=False,
                    meta_message_id=None,
                    messaging_product="whatsapp",
                    raw_response={"status": "blocked", "reason": block_reason}
                )
        except ImportError:
            pass

        start_ms = time.time() * 1000
        try:
            response = requests.post(
                self.url,
                headers=self._headers,
                json=payload,
                timeout=15
            )
            latency = round(time.time() * 1000 - start_ms, 1)

            try:
                res_data = response.json()
            except Exception:
                res_data = {"raw": response.text}

            logger.info(
                "[WA_CLIENT] Meta API responded in %.1f ms — HTTP %d",
                latency,
                response.status_code,
            )

            if response.status_code >= 400 or "error" in res_data:
                error_obj = res_data.get("error", {})
                error_code = int(error_obj.get("code", 0))
                error_subcode = int(error_obj.get("error_subcode", 0))
                error_message = str(error_obj.get("message", "Unknown Meta API error"))

                logger.error(
                    "[WA_CLIENT] Meta API error — HTTP %d, code=%d, subcode=%d: %s",
                    response.status_code, error_code, error_subcode, error_message
                )

                raise map_meta_error(
                    http_status=response.status_code,
                    error_code=error_code,
                    error_subcode=error_subcode,
                    error_message=error_message
                )

            messages = res_data.get("messages", [{}])
            meta_message_id = messages[0].get("id", "") if messages else ""

            try:
                from safety_guard import record_outbound_send
                record_outbound_send(to_number)
            except Exception:
                pass

            return WhatsAppSendResult(
                success=True,
                meta_message_id=meta_message_id,
                messaging_product=res_data.get("messaging_product", "whatsapp"),
                raw_response=res_data,
            )

        except (WhatsAppAuthenticationError, WhatsAppRateLimitError, WhatsAppTemplateError,
                WhatsAppRecipientError, WhatsAppTransientError, WhatsAppSendError,
                WhatsAppConfigError):
            raise
        except requests.exceptions.Timeout:
            raise WhatsAppTransientError("Meta API request timed out.")
        except requests.exceptions.ConnectionError:
            raise WhatsAppTransientError("Failed to connect to Meta API.")
        except requests.exceptions.RequestException as ex:
            raise WhatsAppSendError(f"Request failed: {type(ex).__name__}")

    def send_text(self, recipient: str, message: str) -> WhatsAppSendResult:
        clean = sanitize_recipient(recipient)
        if not clean:
            raise WhatsAppRecipientError(f"Invalid recipient: {recipient!r}")
        return self._dispatch(build_text_payload(clean, message))

    def send_interactive(
        self,
        recipient: str,
        body_text: str,
        buttons: List[Dict[str, str]],
        header_text: Optional[str] = None,
        footer_text: Optional[str] = None
    ) -> WhatsAppSendResult:
        clean = sanitize_recipient(recipient)
        if not clean:
            raise WhatsAppRecipientError(f"Invalid recipient: {recipient!r}")
        return self._dispatch(build_buttons_payload(clean, body_text, buttons, header_text, footer_text))

    def send_buttons(self, *args, **kwargs) -> WhatsAppSendResult:
        return self.send_interactive(*args, **kwargs)

    def send_list(
        self,
        recipient: str,
        body_text: str,
        button_label: str,
        sections: List[Dict[str, Any]],
        title: Optional[str] = None,
        footer_text: Optional[str] = None
    ) -> WhatsAppSendResult:
        clean = sanitize_recipient(recipient)
        if not clean:
            raise WhatsAppRecipientError(f"Invalid recipient: {recipient!r}")
        return self._dispatch(build_list_payload(clean, body_text, button_label, sections, title, footer_text))

    def send_document(
        self,
        recipient: str,
        document_url: str,
        filename: str,
        caption: Optional[str] = None
    ) -> WhatsAppSendResult:
        clean = sanitize_recipient(recipient)
        if not clean:
            raise WhatsAppRecipientError(f"Invalid recipient: {recipient!r}")
        return self._dispatch(build_document_payload(clean, document_url, filename, caption))

    def send_image(self, recipient: str, image_url: str, caption: Optional[str] = None) -> WhatsAppSendResult:
        clean = sanitize_recipient(recipient)
        if not clean:
            raise WhatsAppRecipientError(f"Invalid recipient: {recipient!r}")
        return self._dispatch(build_image_payload(clean, image_url, caption))

    def send_location(
        self, recipient: str, latitude: float, longitude: float, name: str, address: str
    ) -> WhatsAppSendResult:
        clean = sanitize_recipient(recipient)
        if not clean:
            raise WhatsAppRecipientError(f"Invalid recipient: {recipient!r}")
        return self._dispatch(build_location_payload(clean, latitude, longitude, name, address))

    def send_template(
        self,
        recipient: str,
        template_name: str,
        language_code: str = "en",
        components: Optional[List[Dict[str, Any]]] = None
    ) -> WhatsAppSendResult:
        clean = sanitize_recipient(recipient)
        if not clean:
            raise WhatsAppRecipientError(f"Invalid recipient: {recipient!r}")
        return self._dispatch(build_template_payload(clean, template_name, language_code, components))

    def fetch_meta_templates(self) -> Dict[str, Any]:
        self._validate_config()
        phone_url = f"{self.base_url}/{self.api_version}/{self.phone_number_id}?fields=whatsapp_business_account"
        try:
            r = requests.get(phone_url, headers=self._headers, timeout=10)
            phone_data = r.json()
            waba_id = phone_data.get("whatsapp_business_account", {}).get("id")
            if not waba_id:
                return {"error": "Could not locate WABA ID for phone number", "details": phone_data}
            tmpl_url = f"{self.base_url}/{self.api_version}/{waba_id}/message_templates?limit=100"
            tr = requests.get(tmpl_url, headers=self._headers, timeout=10)
            return tr.json()
        except Exception as ex:
            logger.error(f"[WA_CLIENT] fetch_meta_templates error: {ex}")
            return {"error": str(ex)}

    def mark_read(self, message_id: str) -> bool:
        if not message_id:
            return False
        payload = {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": message_id
        }
        try:
            self._dispatch(payload)
            return True
        except Exception:
            return False


# =====================================================================
# 5. Template Registry
# =====================================================================

@dataclass
class WhatsAppTemplate:
    name: str
    language_code: str = "en"
    category: str = "MARKETING"
    display_name: str = ""
    description: str = ""
    body_preview: str = ""
    status: str = "ACTIVE"
    quality_status: str = "QUALITY_PENDING"
    has_header: bool = False
    has_footer: bool = False
    header_type: str = ""
    header_params: List[str] = field(default_factory=list)
    body_params: List[str] = field(default_factory=list)
    button_params: List[str] = field(default_factory=list)
    enabled: bool = True

    @property
    def body_text(self) -> str:
        return self.body_preview

    @property
    def quality_score(self) -> str:
        return self.quality_status

    @property
    def is_active(self) -> bool:
        return self.enabled and self.status == "ACTIVE"

    def is_sendable(self) -> bool:
        return self.enabled and self.status in ("ACTIVE", "QUALITY_PENDING")

    def requires_components(self) -> bool:
        return bool(self.header_params or self.body_params or self.button_params)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def ui_status_badge(self) -> str:
        if self.quality_status == "QUALITY_APPROVED":
            return "Active · Approved"
        elif self.quality_status == "QUALITY_PENDING":
            return "Active – Quality pending"
        elif self.quality_status == "QUALITY_REJECTED":
            return "Rejected"
        elif self.status == "PAUSED":
            return "Paused"
        return self.status


TEMPLATES: Dict[str, WhatsAppTemplate] = {
    "aris_lead_outreach": WhatsAppTemplate(
        name="aris_lead_outreach",
        language_code="en",
        category="MARKETING",
        display_name="ARIS Lead Outreach",
        description="Initial outreach to new real estate leads.",
        body_preview="HelloHello! Thank you for your interest in ARIS Real Estate. How can we assist you today?",
        status="ACTIVE",
        quality_status="QUALITY_PENDING",
        has_header=False,
        enabled=True,
    ),
    "brochure_altimetai": WhatsAppTemplate(
        name="brochure_altimetai",
        language_code="en",
        category="MARKETING",
        display_name="Brochure – Altimetai",
        description="Property brochure outreach for Nagpur investment corridor.",
        body_preview="Namaste! 🙏 Looking to invest in Nagpur's fastest-growing real estate corridors?",
        status="ACTIVE",
        quality_status="QUALITY_PENDING",
        has_header=True,
        header_type="DOCUMENT",
        enabled=True,
    ),
}


class TemplateRegistry:
    @staticmethod
    def get(name: str) -> Optional[WhatsAppTemplate]:
        return TEMPLATES.get(name.strip().lower())

    @staticmethod
    def register(template: WhatsAppTemplate):
        TEMPLATES[template.name.strip().lower()] = template

    @staticmethod
    def get_all() -> List[WhatsAppTemplate]:
        return list(TEMPLATES.values())

    @staticmethod
    def list_all() -> List[WhatsAppTemplate]:
        return list(TEMPLATES.values())

    @staticmethod
    def get_enabled() -> List[WhatsAppTemplate]:
        return [t for t in TEMPLATES.values() if t.is_sendable()]

    @staticmethod
    def validate(name: str) -> Optional[str]:
        tpl = TEMPLATES.get(name.strip().lower())
        if not tpl:
            return f"Template '{name}' is not registered in the system."
        if not tpl.enabled:
            return f"Template '{name}' is disabled."
        if not tpl.is_sendable():
            return f"Template '{name}' has status '{tpl.status}' and cannot be sent."
        return None


# =====================================================================
# 6. Rate Limiter (MongoDB-backed)
# =====================================================================

RATE_LIMIT_COLLECTION = "rate_limit_events"


class RateLimiter:
    """
    Server-side rate limiter for test send operations using MongoDB time-window.
    """

    def __init__(self):
        self._db = get_db()
        self._limit = Config.WHATSAPP_TEST_SEND_RATE_LIMIT
        self._window_seconds = Config.WHATSAPP_TEST_SEND_RATE_WINDOW_SECONDS
        self._ensure_index()

    def _ensure_index(self):
        try:
            col = self._db.db[RATE_LIMIT_COLLECTION]
            col.create_index(
                [("created_at", 1)],
                expireAfterSeconds=self._window_seconds,
                name="rate_limit_ttl"
            )
        except Exception:
            pass

    def _col(self):
        return self._db.db[RATE_LIMIT_COLLECTION]

    def check_and_record(self, user_identifier: str, action: str = "TEST_SEND") -> Tuple[bool, int, int]:
        window_start = datetime.now(timezone.utc) - timedelta(seconds=self._window_seconds)
        try:
            count = self._col().count_documents({
                "user_id": user_identifier,
                "action": action,
                "created_at": {"$gte": window_start}
            })

            if count >= self._limit:
                logger.warning(
                    "[RATE_LIMITER] Rate limit exceeded for '%s' on '%s': %d/%d",
                    user_identifier, action, count, self._limit
                )
                return False, count, self._limit

            self._col().insert_one({
                "user_id": user_identifier,
                "action": action,
                "created_at": datetime.now(timezone.utc),
            })
            return True, count + 1, self._limit
        except Exception as ex:
            logger.warning("[RATE_LIMITER] Rate check exception (allowing request): %s", ex)
            return True, 0, self._limit

    def get_current_count(self, user_identifier: str, action: str = "TEST_SEND") -> int:
        window_start = datetime.now(timezone.utc) - timedelta(seconds=self._window_seconds)
        try:
            return self._col().count_documents({
                "user_id": user_identifier,
                "action": action,
                "created_at": {"$gte": window_start}
            })
        except Exception:
            return 0


# =====================================================================
# 7. Outbound Service (Single Reusable Send Service)
# =====================================================================

class SendResult:
    def __init__(
        self,
        success: bool,
        outbound_id: str = "",
        meta_message_id: str = "",
        status: str = "",
        error_code: str = "",
        safe_error_message: str = "",
    ):
        self.success = success
        self.outbound_id = outbound_id
        self.meta_message_id = meta_message_id
        self.status = status
        self.error_code = error_code
        self.safe_error_message = safe_error_message

    def to_api_dict(self) -> Dict[str, Any]:
        if self.success:
            return {
                "success": True,
                "outbound_id": self.outbound_id,
                "message_id": self.meta_message_id,
                "status": self.status,
            }
        return {
            "success": False,
            "error": {
                "code": self.error_code,
                "message": self.safe_error_message,
            }
        }


class OutboundService:
    """
    Unified outbound template message sender used by both Test Send and Campaign Worker.
    """

    def __init__(self):
        self.client = WhatsAppClient()
        self.repo = DB.outbound_messages
        self.phone_svc = PhoneService()

    def send_template(
        self,
        phone_number: str,
        template_name: str,
        language_code: str = "en",
        components: Optional[List[Dict[str, Any]]] = None,
        source: str = OutboundSource.TEST_SEND,
        campaign_id: Optional[str] = None,
        lead_id: Optional[str] = None,
        created_by: str = "system",
        idempotency_key: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        # Step 1: Validate template
        template_error = TemplateRegistry.validate(template_name)
        if template_error:
            logger.warning("[OUTBOUND] Template validation failed: %s", template_error)
            return SendResult(
                success=False,
                error_code="TEMPLATE_VALIDATION_FAILED",
                safe_error_message=template_error,
            )

        template = TemplateRegistry.get(template_name)

        # Step 2: Normalize phone
        try:
            phone_info = self.phone_svc.normalize(phone_number)
        except ValueError as ex:
            return SendResult(
                success=False,
                error_code="INVALID_PHONE_NUMBER",
                safe_error_message=f"Invalid phone number format: {str(ex)}",
            )

        normalized = phone_info["normalized_phone"]
        masked = phone_info["masked"]

        # Step 3: Idempotency check
        idem_key = idempotency_key or f"{source}_{normalized}_{template_name}_{uuid.uuid4().hex[:8]}"
        existing = self.repo.check_idempotency(idem_key)
        if existing:
            return SendResult(
                success=True,
                outbound_id=existing.get("outbound_id", ""),
                meta_message_id=existing.get("meta_message_id", ""),
                status=existing.get("status", OutboundStatus.ACCEPTED),
            )

        # Step 4: Record outbound message
        msg = OutboundMessage(
            phone_number=phone_info["raw_phone"],
            normalized_phone=normalized,
            country_code=phone_info["country_code"],
            lead_id=lead_id,
            campaign_id=campaign_id,
            created_by=created_by,
            template_name=template_name,
            template_language=language_code or template.language_code,
            template_components=components or [],
            idempotency_key=idem_key,
            source=source,
            status=OutboundStatus.SENDING,
            metadata=metadata or {},
        )
        self.repo.save(msg)

        # Step 5: Attach document component if brochure template requires it
        send_components = components
        if not send_components and template and getattr(template, "header_type", "") == "DOCUMENT":
            send_components = [
                {
                    "type": "header",
                    "parameters": [
                        {
                            "type": "document",
                            "document": {
                                "link": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
                                "filename": "ARIS_Brochure.pdf"
                            }
                        }
                    ]
                }
            ]

        # Step 6: Dispatch via Meta WhatsApp Cloud API
        try:
            result = self.client.send_template(
                recipient=normalized,
                template_name=template_name,
                language_code=language_code or (template.language_code if template else "en"),
                components=send_components,
            )

            self.repo.update_status(
                outbound_id=msg.outbound_id,
                status=OutboundStatus.ACCEPTED,
                meta_message_id=result.meta_message_id,
            )

            logger.info(
                "[OUTBOUND] Meta accepted template '%s' -> wamid: %s (to: %s)",
                template_name,
                result.meta_message_id[:20] + "..." if result.meta_message_id else "UNKNOWN",
                masked,
            )

            return SendResult(
                success=True,
                outbound_id=msg.outbound_id,
                meta_message_id=result.meta_message_id,
                status=OutboundStatus.ACCEPTED,
            )

        except WhatsAppBaseError as ex:
            self.repo.update_status(
                outbound_id=msg.outbound_id,
                status=OutboundStatus.FAILED,
                error_code=ex.code,
                error_message=ex.internal_message,
            )
            return SendResult(
                success=False,
                outbound_id=msg.outbound_id,
                error_code=ex.code,
                safe_error_message=ex.safe_message,
            )

        except Exception as ex:
            self.repo.update_status(
                outbound_id=msg.outbound_id,
                status=OutboundStatus.FAILED,
                error_code="UNEXPECTED_ERROR",
                error_message=str(type(ex).__name__),
            )
            return SendResult(
                success=False,
                outbound_id=msg.outbound_id,
                error_code="WHATSAPP_SEND_FAILED",
                safe_error_message="An unexpected error occurred. Contact system administrator.",
            )


# =====================================================================
# 8. Campaign Worker (Asynchronous Durable Queue Processing)
# =====================================================================

_job_queue: List[Dict[str, Any]] = []
_queue_lock = threading.Lock()
_worker_threads: Dict[str, threading.Thread] = {}
_threads_lock = threading.Lock()


def _try_redis():
    try:
        import redis
        r = redis.from_url(Config.REDIS_URL, socket_connect_timeout=2)
        r.ping()
        return r
    except Exception:
        return None


def is_within_delivery_window(start_str: Optional[str], end_str: Optional[str]) -> bool:
    """Check if current time (IST / UTC+5:30) is within the delivery window 'HH:MM' - 'HH:MM'."""
    if not start_str or not end_str:
        return True
    try:
        ist = timezone(timedelta(hours=5, minutes=30))
        now_time = datetime.now(ist).time()
        start_parts = [int(p) for p in start_str.strip().split(":")]
        end_parts = [int(p) for p in end_str.strip().split(":")]
        from datetime import time as dt_time
        start_t = dt_time(start_parts[0], start_parts[1])
        end_t = dt_time(end_parts[0], end_parts[1])
        if start_t <= end_t:
            return start_t <= now_time <= end_t
        else:
            return now_time >= start_t or now_time <= end_t
    except Exception:
        return True


def categorize_error(error_code: Optional[Any], error_message: Optional[str]) -> str:
    """Categorize Meta WhatsApp error into normalized buckets."""
    code = str(error_code or "").upper()
    msg = str(error_message or "").upper()
    if "190" in code or "AUTH" in code or "AUTH" in msg or "TOKEN" in msg:
        return "AUTH"
    if "130429" in code or "131056" in code or "80007" in code or "RATE" in code or "RATE" in msg or "THROTTLED" in msg:
        return "RATE_LIMIT"
    if "132000" in code or "132001" in code or "132015" in code or "TEMPLATE" in code or "TEMPLATE" in msg:
        return "TEMPLATE_ERROR"
    if "131026" in code or "131047" in code or "131051" in code or "PHONE" in msg or "RECIPIENT" in msg:
        return "RECIPIENT_ERROR"
    if "TRANSIENT" in code or "TIMEOUT" in msg or "NETWORK" in msg or "TRANSIENT" in msg:
        return "TRANSIENT"
    return "OTHER"


class CampaignWorker:
    """
    Production-grade asynchronous campaign sender with MongoDB-backed recipient durability,
    delivery window enforcement, per-campaign rate throttling, and automatic process restart recovery.
    """

    def __init__(self):
        self.outbound_svc = OutboundService()
        self.campaign_repo = DB.campaigns
        self.recipient_repo = DB.campaign_recipients
        self.lead_repo = DB.leads
        self.event_repo = DB.events
        self._db = get_db()
        self._redis = _try_redis()
        self._rate = Config.WHATSAPP_CAMPAIGN_MESSAGES_PER_SECOND
        self._batch_size = Config.WHATSAPP_CAMPAIGN_BATCH_SIZE
        self._max_retries = Config.WHATSAPP_MAX_RETRY_ATTEMPTS

    def enqueue_campaign(
        self,
        campaign_id: str,
        leads: List[Dict[str, Any]],
        template_name: str,
        language_code: str = "en",
    ) -> int:
        """Populates campaign recipients into database and kicks off worker thread."""
        recipients = []
        for lead in leads:
            phone = lead.get("wa_id") or lead.get("phone", "")
            recipients.append({
                "lead_id": lead.get("lead_id", ""),
                "phone": phone,
                "name": lead.get("name") or "Valued Customer",
                "status": "PENDING"
            })

        count = self.recipient_repo.bulk_insert(campaign_id, recipients)
        self.campaign_repo.update_campaign(campaign_id, {"queued_count": count})
        self._start_worker_thread(campaign_id)
        return count

    def _start_worker_thread(self, campaign_id: str):
        global _worker_threads
        with _threads_lock:
            existing_thread = _worker_threads.get(campaign_id)
            if existing_thread and existing_thread.is_alive():
                return
            t = threading.Thread(
                target=self._process_queue,
                args=(campaign_id,),
                daemon=True,
                name=f"aris-camp-{campaign_id[:8]}"
            )
            _worker_threads[campaign_id] = t
            t.start()

    @staticmethod
    def resume_stalled_campaigns():
        """Resumes processing for any campaigns that were left RUNNING on server restart."""
        try:
            running = DB.campaigns.list_campaigns_filtered(status=CampaignStatus.RUNNING, limit=20)
            worker = CampaignWorker()
            for c in running:
                cid = c.get("campaign_id")
                safe_terminal_log(f"[CAMPAIGN WORKER] Resuming stalled campaign: {cid} ('{c.get('title')}')")
                worker._start_worker_thread(cid)
        except Exception as ex:
            safe_terminal_log(f"[CAMPAIGN WORKER] Stalled recovery error: {ex}")

    def _process_queue(self, campaign_id: str):
        campaign = self.campaign_repo.get_by_id(campaign_id) or {}
        camp_title = campaign.get("title", campaign_id)
        template = campaign.get("template_name", "outreach")
        lang_code = campaign.get("language_code", "en")
        speed = float(campaign.get("rate_limit_per_second") or self._rate or 1.0)
        deliv_start = campaign.get("delivery_window_start") or Config.CAMPAIGN_DELIVERY_WINDOW_START
        deliv_end = campaign.get("delivery_window_end") or Config.CAMPAIGN_DELIVERY_WINDOW_END

        stats = self.recipient_repo.get_stats(campaign_id)
        total_initial = stats.get("total", 0)

        safe_terminal_log("\n" + "=" * 70)
        safe_terminal_log(f" [CAMPAIGN WORKER] START DISPATCH: '{camp_title}' ({campaign_id})")
        safe_terminal_log(f" Total Audience: {total_initial} | Template: '{template}' | Speed: {speed} msg/sec")
        safe_terminal_log(f" Delivery Window: {deliv_start} - {deliv_end} IST")
        safe_terminal_log("=" * 70)

        processed = 0
        success_count = 0
        failed_count = 0
        skipped_count = 0
        retry_delays = [5, 30, 120]

        while True:
            try:
                # 1. Check if campaign was paused, cancelled, or deleted
                current_camp = self.campaign_repo.get_by_id(campaign_id)
                if not current_camp or current_camp.get("status") in (CampaignStatus.PAUSED, CampaignStatus.CANCELLED) or current_camp.get("is_deleted"):
                    status_name = current_camp.get("status") if current_camp else "REMOVED"
                    safe_terminal_log(f"\n[CAMPAIGN WORKER] Campaign {campaign_id} status is now {status_name}. Pausing worker loop.")
                    break

                # 2. Check delivery window
                if not is_within_delivery_window(deliv_start, deliv_end):
                    safe_terminal_log(f"[CAMPAIGN WORKER] Current time outside delivery window ({deliv_start} - {deliv_end} IST). Pausing for 60s...")
                    time.sleep(60)
                    continue

                # 3. Claim next batch atomically from MongoDB
                batch = self.recipient_repo.claim_next_batch(campaign_id, batch_size=self._batch_size)
                if not batch:
                    # Check if any pending or queued remain
                    current_stats = self.recipient_repo.get_stats(campaign_id)
                    if current_stats.get("pending", 0) == 0 and current_stats.get("queued", 0) == 0:
                        break
                    time.sleep(2)
                    continue

                # 4. Process each recipient in claimed batch
                for recipient in batch:
                    # Check campaign status again before each message
                    c_chk = self.campaign_repo.get_by_id(campaign_id)
                    if not c_chk or c_chk.get("status") in (CampaignStatus.PAUSED, CampaignStatus.CANCELLED) or c_chk.get("is_deleted"):
                        # Put back to PENDING so it can resume cleanly
                        self.recipient_repo.update_status(campaign_id, recipient.get("lead_id"), "PENDING")
                        break

                    processed += 1
                    phone = str(recipient.get("phone", "")).strip()
                    lead_id = str(recipient.get("lead_id", "")).strip()
                    attempt = int(recipient.get("retry_count", 0))

                    # Resolve lead details for name & consent check
                    lead = self.lead_repo.get_by_id(lead_id) if lead_id else None
                    if not lead and phone:
                        lead = self.lead_repo.get_by_wa_id(phone) or self.lead_repo.get_by_phone(phone)

                    raw_name = recipient.get("name") or (lead.get("name") if lead else "") or "Valued Client"
                    safe_lead_name = re.sub(r"[^\x00-\x7F]+", "", raw_name).strip() or "Valued Client"

                    safe_terminal_log(f"\n[CAMPAIGN WORKER] [{processed}/{total_initial}] Lead: +{phone} ({safe_lead_name})")

                    # Opt-in and consent check
                    if lead:
                        if lead.get("opted_out") is True or lead.get("marketing_opt_in") is False:
                            safe_terminal_log(f"  [SKIPPED] Lead +{phone} opted out or revoked consent.")
                            skipped_count += 1
                            self.recipient_repo.update_status(
                                campaign_id, lead_id, "SKIPPED",
                                error_code="CONSENT_REVOKED",
                                error_message="Lead opted out or revoked marketing consent",
                                error_category="OPT_OUT"
                            )
                            self.campaign_repo.increment_counter(campaign_id, "excluded_no_consent")
                            continue

                    # Build personalized components
                    lead_city = (lead.get("preferred_city") or lead.get("city") or "Nagpur") if lead else "Nagpur"
                    lead_bhk = (lead.get("bhk") or "2BHK") if lead else "2BHK"
                    if isinstance(lead_bhk, list) and lead_bhk:
                        lead_bhk = lead_bhk[0]

                    tpl_obj = TemplateRegistry.get(template)
                    components = None
                    if tpl_obj and getattr(tpl_obj, "body_params", None):
                        param_values = [str(safe_lead_name), str(lead_bhk), str(lead_city)]
                        components = [{
                            "type": "body",
                            "parameters": [{"type": "text", "text": val} for val in param_values[:len(tpl_obj.body_params)]]
                        }]

                    safe_terminal_log(f"  [SENDING] Meta WhatsApp Template '{template}' -> +{phone}...")

                    result = self.outbound_svc.send_template(
                        phone_number=phone,
                        template_name=template,
                        language_code=lang_code,
                        source=OutboundSource.CAMPAIGN,
                        campaign_id=campaign_id,
                        lead_id=lead_id or (lead.get("lead_id") if lead else ""),
                        created_by="campaign_worker",
                        idempotency_key=f"camp_{campaign_id}_{lead_id}_{template}",
                        components=components,
                    )

                    if result.success:
                        success_count += 1
                        self.recipient_repo.update_status(
                            campaign_id,
                            lead_id,
                            "SENT",
                            outbound_id=result.outbound_id,
                            meta_message_id=result.meta_message_id,
                        )
                        self.campaign_repo.increment_counter(campaign_id, "sent_count")
                        safe_terminal_log(f"  [SUCCESS] Accepted -> wamid: {result.meta_message_id}")
                    else:
                        cat = categorize_error(result.error_code, result.safe_error_message)
                        if cat == "TRANSIENT" and attempt < self._max_retries:
                            delay = retry_delays[min(attempt, len(retry_delays) - 1)]
                            safe_terminal_log(f"  [TRANSIENT] Retrying +{phone} in {delay}s (attempt {attempt+1}/{self._max_retries})")
                            time.sleep(delay)
                            self.recipient_repo.mark_failed_for_retry(campaign_id, lead_id)
                        else:
                            failed_count += 1
                            self.recipient_repo.update_status(
                                campaign_id,
                                lead_id,
                                "FAILED",
                                error_code=result.error_code,
                                error_message=result.safe_error_message,
                                error_category=cat,
                            )
                            self.campaign_repo.increment_counter(campaign_id, "failed_count")
                            safe_terminal_log(f"  [FAILED] Failed for +{phone}: [{result.error_code}] {result.safe_error_message}")

                    # Update queued count
                    rem = self.recipient_repo.count_recipients(campaign_id, status_filter="PENDING")
                    self.campaign_repo.update_campaign(campaign_id, {"queued_count": rem})

                    # Rate throttling
                    time.sleep(1.0 / max(0.5, speed))

            except Exception as ex:
                safe_terminal_log(f"  [WORKER EXCEPTION] {str(ex)}")
                time.sleep(2)

        # Final queue evaluation
        final_stats = self.recipient_repo.get_stats(campaign_id)
        rem = final_stats.get("pending", 0) + final_stats.get("queued", 0)
        err_breakdown = self.recipient_repo.get_error_breakdown(campaign_id)

        if rem == 0 and total_initial > 0:
            self.campaign_repo.update_campaign(campaign_id, {
                "status": CampaignStatus.COMPLETED,
                "completed_at": datetime.now(timezone.utc),
                "queued_count": 0,
                "error_summary": err_breakdown,
            })
            status_text = "COMPLETED"
        else:
            status_text = "PAUSED / HALTED"
            self.campaign_repo.update_campaign(campaign_id, {
                "error_summary": err_breakdown,
            })

        safe_terminal_log("\n" + "=" * 70)
        safe_terminal_log(f" [CAMPAIGN WORKER] DISPATCH {status_text}: '{camp_title}' ({campaign_id})")
        safe_terminal_log(f" Processed: {processed} | Sent: {success_count} | Failed: {failed_count} | Skipped: {skipped_count} | Remaining: {rem}")
        safe_terminal_log("=" * 70 + "\n")


# =====================================================================
# 9. Campaign Service (Marketing Lifecycle & Audience Segmentation)
# =====================================================================

class CampaignService:
    """
    Full lifecycle campaign management: create -> validate -> dry-run -> start -> pause -> resume -> cancel.
    """

    def __init__(self):
        self.campaign_repo = DB.campaigns
        self.recipient_repo = DB.campaign_recipients
        self.lead_repo = DB.leads
        self.event_repo = DB.events
        self.outbound_repo = DB.outbound_messages
        self.phone_svc = PhoneService()

    def create_campaign(
        self,
        title: str,
        template_name: str,
        target_city: Optional[str] = None,
        target_stage: Optional[str] = None,
        min_lead_score: int = 0,
        language_code: str = "en",
        description: str = "",
        created_by: str = "admin",
        bhk_filter: Optional[str] = None,
        budget_min: Optional[float] = None,
        budget_max: Optional[float] = None,
        property_id: Optional[str] = None,
        project_id: Optional[str] = None,
        audience_type: str = "FILTER",
        selected_lead_ids: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        priority: int = 2,
        delivery_window_start: Optional[str] = None,
        delivery_window_end: Optional[str] = None,
        rate_limit_per_second: Optional[int] = None,
        scheduled_at: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        if template_name.strip().lower() == "hello_world":
            raise ValueError("'hello_world' is not permitted for marketing campaigns.")

        error = TemplateRegistry.validate(template_name)
        if error:
            raise ValueError(error)

        from config import Config
        data = {
            "title": title,
            "description": description,
            "template_name": template_name,
            "language_code": language_code,
            "target_city": target_city,
            "target_stage": target_stage,
            "min_lead_score": min_lead_score,
            "bhk_filter": bhk_filter,
            "budget_min": budget_min,
            "budget_max": budget_max,
            "property_id": property_id,
            "project_id": project_id,
            "audience_type": audience_type,
            "selected_lead_ids": selected_lead_ids or [],
            "tags": tags or [],
            "priority": int(priority) if priority else 2,
            "delivery_window_start": delivery_window_start or Config.CAMPAIGN_DELIVERY_WINDOW_START,
            "delivery_window_end": delivery_window_end or Config.CAMPAIGN_DELIVERY_WINDOW_END,
            "rate_limit_per_second": rate_limit_per_second or int(Config.WHATSAPP_CAMPAIGN_MESSAGES_PER_SECOND),
            "scheduled_at": scheduled_at,
            "status": CampaignStatus.QUEUED if scheduled_at else CampaignStatus.DRAFT,
            "created_by": created_by,
        }
        campaign = self.campaign_repo.create_campaign(data)
        self._audit("CAMPAIGN_CREATED", campaign.get("campaign_id"), created_by)
        return campaign

    def update_campaign(
        self,
        campaign_id: str,
        updates: Dict[str, Any],
        updated_by: str = "admin"
    ) -> Dict[str, Any]:
        campaign = self.campaign_repo.get_by_id(campaign_id)
        if not campaign:
            raise ValueError(f"Campaign '{campaign_id}' not found.")

        current_status = campaign.get("status")
        # In WATI / Meta standards: Cannot modify audience targeting or template if campaign is actively RUNNING or COMPLETED
        restricted_fields = [
            "template_name", "audience_type", "selected_lead_ids",
            "target_city", "target_stage", "min_lead_score", "bhk_filter"
        ]
        if current_status in (CampaignStatus.RUNNING, CampaignStatus.COMPLETED):
            for field in restricted_fields:
                if field in updates and updates[field] != campaign.get(field):
                    raise ValueError(f"Cannot modify '{field}' while campaign is {current_status}. Please pause the campaign first.")

        # If template is modified, validate it
        if "template_name" in updates and updates["template_name"]:
            template_name = str(updates["template_name"]).strip()
            if template_name.lower() == "hello_world":
                raise ValueError("'hello_world' is not permitted for marketing campaigns.")
            err = TemplateRegistry.validate(template_name)
            if err:
                raise ValueError(err)

        allowed_fields = [
            "title", "description", "template_name", "language_code",
            "target_city", "target_stage", "min_lead_score", "bhk_filter",
            "budget_min", "budget_max", "property_id", "project_id",
            "audience_type", "selected_lead_ids", "scheduled_at",
            "tags", "priority", "delivery_window_start", "delivery_window_end",
            "rate_limit_per_second"
        ]
        clean_updates: Dict[str, Any] = {}
        for k in allowed_fields:
            if k in updates:
                clean_updates[k] = updates[k]

        clean_updates["updated_at"] = datetime.now(timezone.utc)
        clean_updates["updated_by"] = updated_by

        # Recalculate eligible count if audience targeting changed
        aud_type = clean_updates.get("audience_type", campaign.get("audience_type", "FILTER"))
        sel_leads = clean_updates.get("selected_lead_ids", campaign.get("selected_lead_ids", []))
        t_city = clean_updates.get("target_city", campaign.get("target_city"))
        t_stage = clean_updates.get("target_stage", campaign.get("target_stage"))
        m_score = clean_updates.get("min_lead_score", campaign.get("min_lead_score", 0))
        bhk_f = clean_updates.get("bhk_filter", campaign.get("bhk_filter"))

        eligible = self.get_eligible_audience(
            city=t_city,
            stage=t_stage,
            min_score=m_score,
            bhk_filter=bhk_f,
            audience_type=aud_type,
            selected_lead_ids=sel_leads,
        )
        clean_updates["eligible_count"] = len(eligible)
        if current_status in (CampaignStatus.DRAFT, CampaignStatus.READY):
            clean_updates["queued_count"] = len(eligible)

        updated_doc = self.campaign_repo.update_campaign(campaign_id, clean_updates)
        self._audit("CAMPAIGN_UPDATED", campaign_id, updated_by, {"modified_keys": list(clean_updates.keys())})
        return updated_doc or self.campaign_repo.get_by_id(campaign_id)

    def remove_lead_from_campaign(
        self,
        campaign_id: str,
        lead_id: str,
        removed_by: str = "admin"
    ) -> Dict[str, Any]:
        campaign = self.campaign_repo.get_by_id(campaign_id)
        if not campaign:
            raise ValueError(f"Campaign '{campaign_id}' not found.")

        current_status = campaign.get("status")
        if current_status not in (CampaignStatus.DRAFT, CampaignStatus.READY, CampaignStatus.PAUSED):
            raise ValueError(f"Cannot remove leads while campaign is {current_status}.")

        aud_type = campaign.get("audience_type", "FILTER")
        selected_leads = list(campaign.get("selected_lead_ids") or [])

        if aud_type in ("SELECTED_LEADS", "FILE_UPLOAD"):
            new_leads = [lid for lid in selected_leads if lid != lead_id]
            return self.update_campaign(campaign_id, {"selected_lead_ids": new_leads}, updated_by=removed_by)
        else:
            current_audience = self.get_eligible_audience(
                city=campaign.get("target_city"),
                stage=campaign.get("target_stage"),
                min_score=campaign.get("min_lead_score", 0),
                bhk_filter=campaign.get("bhk_filter"),
                audience_type="FILTER"
            )
            all_lids = [l.get("lead_id") for l in current_audience if l.get("lead_id") != lead_id]
            return self.update_campaign(
                campaign_id,
                {"audience_type": "SELECTED_LEADS", "selected_lead_ids": all_lids},
                updated_by=removed_by
            )

    def get_eligible_audience(
        self,
        city: Optional[str] = None,
        stage: Optional[str] = None,
        min_score: int = 0,
        bhk_filter: Optional[str] = None,
        audience_type: str = "FILTER",
        selected_lead_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        # If specific leads were chosen or imported
        if audience_type in ("SELECTED_LEADS", "FILE_UPLOAD") and selected_lead_ids:
            leads = []
            for lid in selected_lead_ids:
                lead = self.lead_repo.get_by_id(lid)
                if not lead:
                    lead = self.lead_repo.get_by_wa_id(lid)
                if lead and not lead.get("opted_out", False):
                    leads.append(lead)
            return leads

        filter_query: Dict[str, Any] = {
            "opted_out": {"$ne": True},
            "marketing_opt_in": {"$ne": False},
        }
        if city:
            filter_query["preferred_city"] = city
        if stage and stage != "ALL":
            filter_query["sales_stage"] = stage
        if min_score > 0:
            filter_query["lead_score"] = {"$gte": min_score}
        if bhk_filter:
            filter_query["bhk"] = {"$in": [bhk_filter]}

        return self.lead_repo.list_leads(filter_query=filter_query, limit=2000)

    def compute_audience_breakdown(
        self,
        city: Optional[str] = None,
        stage: Optional[str] = None,
        min_score: int = 0,
        bhk_filter: Optional[str] = None,
        audience_type: str = "FILTER",
        selected_lead_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if audience_type in ("SELECTED_LEADS", "FILE_UPLOAD") and selected_lead_ids:
            total = len(selected_lead_ids)
            eligible_leads = self.get_eligible_audience(
                city=city, stage=stage, min_score=min_score, bhk_filter=bhk_filter,
                audience_type=audience_type, selected_lead_ids=selected_lead_ids
            )
            opted_out = 0
            no_consent = 0
            for lid in selected_lead_ids:
                l = self.lead_repo.get_by_id(lid) or self.lead_repo.get_by_wa_id(lid) or {}
                if l.get("opted_out"):
                    opted_out += 1
                elif l.get("marketing_opt_in") is False:
                    no_consent += 1

            invalid_phone = sum(1 for l in eligible_leads if not (l.get("wa_id") or l.get("phone")))
            eligible = max(0, len(eligible_leads) - invalid_phone)
            return {
                "total": total,
                "eligible": eligible,
                "excluded_no_consent": no_consent,
                "excluded_opted_out": opted_out,
                "excluded_invalid_phone": invalid_phone,
            }

        all_filter: Dict[str, Any] = {}
        if city:
            all_filter["preferred_city"] = city
        if stage and stage != "ALL":
            all_filter["sales_stage"] = stage
        if min_score > 0:
            all_filter["lead_score"] = {"$gte": min_score}
        if bhk_filter:
            all_filter["bhk"] = {"$in": [bhk_filter]}

        total = self.lead_repo.count_leads(all_filter)
        opted_out = self.lead_repo.count_leads({**all_filter, "opted_out": True})
        no_consent = self.lead_repo.count_leads({
            **all_filter,
            "opted_out": {"$ne": True},
            "marketing_opt_in": False,
        })

        eligible_leads = self.get_eligible_audience(city, stage, min_score, bhk_filter)
        invalid_phone = sum(
            1 for l in eligible_leads
            if not (l.get("wa_id") or l.get("phone"))
        )
        eligible = max(0, len(eligible_leads) - invalid_phone)

        return {
            "total": total,
            "eligible": eligible,
            "excluded_no_consent": no_consent,
            "excluded_opted_out": opted_out,
            "excluded_invalid_phone": invalid_phone,
        }

    def validate_campaign(self, campaign_id: str) -> Dict[str, Any]:
        campaign = self.campaign_repo.get_by_id(campaign_id)
        if not campaign:
            return {"error": "Campaign not found"}

        breakdown = self.compute_audience_breakdown(
            city=campaign.get("target_city"),
            stage=campaign.get("target_stage"),
            min_score=campaign.get("min_lead_score", 0),
            bhk_filter=campaign.get("bhk_filter"),
            audience_type=campaign.get("audience_type", "FILTER"),
            selected_lead_ids=campaign.get("selected_lead_ids"),
        )

        eligible_leads = self.get_eligible_audience(
            city=campaign.get("target_city"),
            stage=campaign.get("target_stage"),
            min_score=campaign.get("min_lead_score", 0),
            bhk_filter=campaign.get("bhk_filter"),
            audience_type=campaign.get("audience_type", "FILTER"),
            selected_lead_ids=campaign.get("selected_lead_ids"),
        )

        # Snapshot audience into campaign_recipients
        now = datetime.now(timezone.utc)
        recipients = []
        for l in eligible_leads:
            phone = l.get("wa_id") or l.get("phone", "")
            if phone:
                recipients.append({
                    "lead_id": str(l.get("lead_id", "")),
                    "phone": str(phone),
                    "name": str(l.get("name") or "Valued Customer"),
                    "status": "PENDING"
                })

        inserted_count = self.recipient_repo.bulk_insert(campaign_id, recipients)

        self.campaign_repo.update_campaign(campaign_id, {
            "status": CampaignStatus.READY,
            "validated_at": now,
            "audience_snapshot_at": now,
            "audience_snapshot": recipients[:100],
            "total_recipients": breakdown["total"],
            "eligible_count": breakdown["eligible"],
            "queued_count": inserted_count,
            "excluded_no_consent": breakdown["excluded_no_consent"],
            "excluded_opted_out": breakdown["excluded_opted_out"],
            "excluded_invalid": breakdown["excluded_invalid_phone"],
        })

        return {"campaign_id": campaign_id, "breakdown": breakdown, "recipients_count": inserted_count}

    def dry_run(self, campaign_id: str) -> Dict[str, Any]:
        campaign = self.campaign_repo.get_by_id(campaign_id)
        if not campaign:
            return {"error": "Campaign not found"}

        template_error = TemplateRegistry.validate(campaign.get("template_name", ""))
        if template_error:
            return {"error": template_error, "dry_run": True}

        breakdown = self.compute_audience_breakdown(
            city=campaign.get("target_city"),
            stage=campaign.get("target_stage"),
            min_score=campaign.get("min_lead_score", 0),
            bhk_filter=campaign.get("bhk_filter"),
            audience_type=campaign.get("audience_type", "FILTER"),
            selected_lead_ids=campaign.get("selected_lead_ids"),
        )

        return {
            "dry_run": True,
            "campaign_id": campaign_id,
            "template": campaign.get("template_name"),
            "language": campaign.get("language_code", "en"),
            "audience": breakdown,
            "estimated_messages": breakdown["eligible"],
            "would_send": True,
            "meta_api_called": False,
        }

    def start_campaign(self, campaign_id: str, started_by: str = "admin") -> Dict[str, Any]:
        campaign = self.campaign_repo.get_by_id(campaign_id)
        if not campaign:
            return {"error": "Campaign not found"}

        if campaign.get("status") not in (CampaignStatus.DRAFT, CampaignStatus.READY, CampaignStatus.QUEUED, CampaignStatus.PAUSED, CampaignStatus.RUNNING):
            return {"error": f"Cannot start campaign in status: {campaign.get('status')}"}

        stats = self.recipient_repo.get_stats(campaign_id)
        if stats.get("total", 0) == 0:
            self.validate_campaign(campaign_id)
            stats = self.recipient_repo.get_stats(campaign_id)

        pending_count = stats.get("pending", 0) + stats.get("queued", 0)
        if pending_count == 0:
            self.campaign_repo.update_campaign(campaign_id, {
                "status": CampaignStatus.COMPLETED,
                "completed_at": datetime.now(timezone.utc),
                "queued_count": 0,
            })
            return {
                "success": True,
                "campaign_id": campaign_id,
                "status": CampaignStatus.COMPLETED,
                "enqueued_count": 0,
                "message": "All recipients have already been processed for this campaign."
            }

        self.campaign_repo.update_campaign(campaign_id, {
            "status": CampaignStatus.RUNNING,
            "started_at": datetime.now(timezone.utc),
            "queued_count": pending_count,
        })

        worker = CampaignWorker()
        worker._start_worker_thread(campaign_id)

        self._audit("CAMPAIGN_STARTED", campaign_id, started_by, {"pending_recipients": pending_count})

        return {
            "success": True,
            "campaign_id": campaign_id,
            "status": CampaignStatus.RUNNING,
            "enqueued_count": pending_count,
        }

    def pause_campaign(self, campaign_id: str, paused_by: str = "admin") -> Dict[str, Any]:
        campaign = self.campaign_repo.get_by_id(campaign_id)
        if not campaign:
            return {"error": "Campaign not found"}
        if campaign.get("status") != CampaignStatus.RUNNING:
            return {"error": "Only RUNNING campaigns can be paused."}

        self.campaign_repo.update_campaign(campaign_id, {
            "status": CampaignStatus.PAUSED,
            "paused_at": datetime.now(timezone.utc),
        })
        self._audit("CAMPAIGN_PAUSED", campaign_id, paused_by)
        return {"success": True, "campaign_id": campaign_id, "status": CampaignStatus.PAUSED}

    def resume_campaign(self, campaign_id: str, resumed_by: str = "admin") -> Dict[str, Any]:
        campaign = self.campaign_repo.get_by_id(campaign_id)
        if not campaign:
            return {"error": "Campaign not found"}
        if campaign.get("status") not in (CampaignStatus.PAUSED, CampaignStatus.RUNNING):
            return {"error": "Only PAUSED or RUNNING campaigns can be resumed."}

        return self.start_campaign(campaign_id, started_by=resumed_by)

    def cancel_campaign(self, campaign_id: str, cancelled_by: str = "admin") -> Dict[str, Any]:
        campaign = self.campaign_repo.get_by_id(campaign_id)
        if not campaign:
            return {"error": "Campaign not found"}
        if campaign.get("status") in (CampaignStatus.COMPLETED, CampaignStatus.CANCELLED):
            return {"error": f"Campaign already {campaign.get('status')}."}

        self.campaign_repo.update_campaign(campaign_id, {
            "status": CampaignStatus.CANCELLED,
            "cancelled_at": datetime.now(timezone.utc),
        })
        self._audit("CAMPAIGN_CANCELLED", campaign_id, cancelled_by)
        return {"success": True, "campaign_id": campaign_id, "status": CampaignStatus.CANCELLED}

    def get_analytics(self, campaign_id: str) -> Dict[str, Any]:
        stats = self.recipient_repo.get_stats(campaign_id)
        campaign = self.campaign_repo.get_by_id(campaign_id) or {}
        queued = max(0, campaign.get("queued_count", stats.get("pending", 0) + stats.get("queued", 0)))
        replied = campaign.get("replied_count", 0)
        delivered = stats.get("delivered", 0)
        reply_rate = round(replied / delivered * 100, 1) if delivered > 0 else 0.0

        return {
            "total": stats.get("total", 0),
            "queued": queued,
            "pending": stats.get("pending", 0),
            "sent": stats.get("sent", 0),
            "delivered": delivered,
            "read": stats.get("read", 0),
            "failed": stats.get("failed", 0),
            "skipped": stats.get("skipped", 0),
            "delivery_rate": stats.get("delivery_rate", 0.0),
            "read_rate": stats.get("read_rate", 0.0),
            "replied": replied,
            "reply_rate": reply_rate,
        }

    def get_recipient_list(
        self,
        campaign_id: str,
        status_filter: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 50,
        skip: int = 0
    ) -> List[Dict[str, Any]]:
        return self.recipient_repo.list_recipients(
            campaign_id=campaign_id,
            status_filter=status_filter,
            search=search,
            limit=limit,
            skip=skip
        )

    def get_recipient_count(
        self,
        campaign_id: str,
        status_filter: Optional[str] = None,
        search: Optional[str] = None
    ) -> int:
        return self.recipient_repo.count_recipients(
            campaign_id=campaign_id,
            status_filter=status_filter,
            search=search
        )

    def get_error_breakdown(self, campaign_id: str) -> Dict[str, Any]:
        return self.recipient_repo.get_error_breakdown(campaign_id)

    def clone_campaign(self, campaign_id: str, new_title: Optional[str] = None, cloned_by: str = "admin") -> Optional[Dict[str, Any]]:
        cloned = self.campaign_repo.clone_campaign(campaign_id, new_title=new_title, cloned_by=cloned_by)
        if cloned:
            self._audit("CAMPAIGN_CLONED", cloned["campaign_id"], cloned_by, {"cloned_from": campaign_id})
        return cloned

    def schedule_campaign(self, campaign_id: str, scheduled_at_iso: str, scheduled_by: str = "admin") -> Dict[str, Any]:
        campaign = self.campaign_repo.get_by_id(campaign_id)
        if not campaign:
            return {"error": "Campaign not found"}

        try:
            dt = datetime.fromisoformat(scheduled_at_iso.replace("Z", "+00:00"))
        except Exception as ex:
            return {"error": f"Invalid date/time format: {ex}"}

        self.validate_campaign(campaign_id)

        self.campaign_repo.update_campaign(campaign_id, {
            "status": CampaignStatus.QUEUED,
            "scheduled_at": dt,
            "updated_at": datetime.now(timezone.utc),
            "updated_by": scheduled_by
        })
        self._audit("CAMPAIGN_SCHEDULED", campaign_id, scheduled_by, {"scheduled_at": scheduled_at_iso})
        return {"success": True, "campaign_id": campaign_id, "status": CampaignStatus.QUEUED, "scheduled_at": dt.isoformat()}

    def retry_failed_recipients(self, campaign_id: str, retried_by: str = "admin") -> Dict[str, Any]:
        campaign = self.campaign_repo.get_by_id(campaign_id)
        if not campaign:
            return {"error": "Campaign not found"}

        retried_count = self.recipient_repo.mark_failed_for_retry(campaign_id)
        if retried_count > 0:
            self.campaign_repo.update_campaign(campaign_id, {
                "status": CampaignStatus.RUNNING,
                "queued_count": retried_count
            })
            worker = CampaignWorker()
            worker._start_worker_thread(campaign_id)
            self._audit("CAMPAIGN_RETRIED", campaign_id, retried_by, {"retried_count": retried_count})

        return {
            "success": True,
            "campaign_id": campaign_id,
            "retried_count": retried_count,
            "status": CampaignStatus.RUNNING if retried_count > 0 else campaign.get("status")
        }

    def get_campaign_timeline(self, campaign_id: str) -> List[Dict[str, Any]]:
        campaign = self.campaign_repo.get_by_id(campaign_id) or {}
        milestones = [
            ("Created", campaign.get("created_at"), campaign.get("created_by", "admin")),
            ("Validated", campaign.get("validated_at"), "system"),
            ("Scheduled", campaign.get("scheduled_at"), campaign.get("updated_by", "system")),
            ("Started", campaign.get("started_at"), "worker"),
            ("Paused", campaign.get("paused_at"), "admin"),
            ("Completed", campaign.get("completed_at"), "worker"),
            ("Cancelled", campaign.get("cancelled_at"), "admin"),
        ]
        timeline = []
        for name, ts, actor in milestones:
            if ts:
                timeline.append({
                    "event": name,
                    "timestamp": ts.isoformat() if isinstance(ts, datetime) else str(ts),
                    "actor": actor
                })
        timeline.sort(key=lambda x: x["timestamp"])
        return timeline

    def get_suppression_list(self, campaign_id: str) -> List[Dict[str, Any]]:
        return self.recipient_repo.list_recipients(campaign_id, status_filter="SKIPPED", limit=500)

    def _audit(
        self,
        action: str,
        campaign_id: str,
        user: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        try:
            self.event_repo.log_event(
                event_type=action.lower(),
                metadata={
                    "campaign_id": campaign_id,
                    "user": user,
                    **(metadata or {}),
                }
            )
        except Exception:
            pass


class CampaignScheduler:
    """
    Background daemon thread monitoring and auto-starting scheduled campaigns.
    """
    _running = False
    _thread = None

    @classmethod
    def start(cls):
        if cls._running:
            return
        cls._running = True
        cls._thread = threading.Thread(target=cls._run_loop, daemon=True, name="aris-campaign-scheduler")
        cls._thread.start()
        safe_terminal_log("[CAMPAIGN SCHEDULER] Background scheduler daemon initialized.")

    @classmethod
    def _run_loop(cls):
        interval = getattr(Config, "CAMPAIGN_SCHEDULER_INTERVAL", 30)
        camp_repo = DB.campaigns
        camp_svc = CampaignService()

        # Try recovering any previously running campaigns on startup
        try:
            CampaignWorker.resume_stalled_campaigns()
        except Exception as ex:
            safe_terminal_log(f"[CAMPAIGN SCHEDULER] Error resuming stalled campaigns: {ex}")

        while cls._running:
            try:
                now_utc = datetime.now(timezone.utc)
                queued_camps = camp_repo.list_campaigns_filtered(status=CampaignStatus.QUEUED, limit=50)
                for camp in queued_camps:
                    sched_at = camp.get("scheduled_at")
                    if not sched_at:
                        continue
                    if isinstance(sched_at, str):
                        try:
                            sched_at = datetime.fromisoformat(sched_at.replace("Z", "+00:00"))
                        except Exception:
                            continue
                    if sched_at.tzinfo is None:
                        sched_at = sched_at.replace(tzinfo=timezone.utc)
                    if sched_at <= now_utc:
                        cid = camp.get("campaign_id")
                        safe_terminal_log(f"[CAMPAIGN SCHEDULER] Auto-starting scheduled campaign '{camp.get('title')}' ({cid}).")
                        camp_svc.start_campaign(cid, started_by="scheduler")
            except Exception as ex:
                safe_terminal_log(f"[CAMPAIGN SCHEDULER] Error in scheduler loop: {ex}")

            time.sleep(interval)


class FollowUpScheduler:
    """
    Background daemon monitoring conversations and triggering smart follow-ups
    for leads inactive between 6 to 12 hours.
    """
    _running = False
    _thread = None

    @classmethod
    def start(cls):
        if cls._running:
            return
        cls._running = True
        cls._thread = threading.Thread(target=cls._run_loop, daemon=True, name="aris-followup-scheduler")
        cls._thread.start()
        safe_terminal_log("[FOLLOWUP SCHEDULER] Background smart follow-up daemon initialized.")

    @classmethod
    def _run_loop(cls):
        poll_interval = 60
        from ai_engine import FollowUpService
        fup_svc = FollowUpService()

        while cls._running:
            try:
                processed = fup_svc.process_all_eligible_followups()
                if processed > 0:
                    safe_terminal_log(f"[FOLLOWUP SCHEDULER] Dispatched {processed} automated smart follow-ups.")
            except Exception as ex:
                safe_terminal_log(f"[FOLLOWUP SCHEDULER] Error in follow-up loop: {ex}")

            time.sleep(poll_interval)


# =====================================================================
# 10. Legacy WhatsAppService Wrapper (Backward Compatibility)
# =====================================================================

class WhatsAppService:
    """
    Backward-compatible wrapper for legacy callers.
    """

    def __init__(self):
        self._client = WhatsAppClient()

    def send_text_message(self, recipient: str, message: str) -> dict:
        try:
            res = self._client.send_text(recipient, message)
            return res.raw_response or {"messages": [{"id": res.meta_message_id}]}
        except Exception as ex:
            return {"error": str(ex)}

    def mark_message_as_read(self, message_id: str) -> dict:
        success = self._client.mark_read(message_id)
        return {"success": success}

    def send_typing_indicator(self, message_id: str) -> dict:
        success = self._client.mark_read(message_id)
        return {"success": success}
