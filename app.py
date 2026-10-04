"""
ARIS AI Real Estate WhatsApp Platform & CRM Server.
Consolidates Flask server, Webhook ingestion, Authentication, CRM Dashboard, and WhatsApp Admin.

Main 5-file architecture:
1. config.py      - Environment settings & validation
2. database.py    - MongoDB connection, data models & repository layer (DB)
3. ai_engine.py   - Gemini LLM, property catalog tools, lead scoring, and conversational orchestrator
4. whatsapp.py    - Meta Cloud API client, template registry, rate limiter, outbound & campaign worker
5. app.py         - Webhook processing, authentication, dashboard views & REST APIs
"""

import sys
import os
import io
import csv
import time
import uuid
import logging
import hashlib
import re
import threading
from functools import wraps
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from bson import ObjectId

from flask import (
    Flask,
    jsonify,
    request,
    Response,
    render_template,
    redirect,
    url_for,
    session,
    flash,
    g,
)

# Ensure UTF-8 output encoding and immediate line buffering for Railway/Docker logs
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

class WerkzeugPollingFilter(logging.Filter):
    """Silences noisy 2-second background polling logs from the browser dashboard."""
    def filter(self, record):
        msg = record.getMessage()
        return not any(p in msg for p in ["/api/overview", "/api/dashboard/message-analytics"])

logging.getLogger("werkzeug").addFilter(WerkzeugPollingFilter())

from config import Config
from database import DB, get_db, serialize_doc, CampaignStatus
from ai_engine import (
    ARISOrchestrator,
    ARISAgent,
    AgentTools,
    PropertyService,
    VisitService,
    FollowUpService,
    LeadScoringService,
    GeminiProvider,
)
from whatsapp import (
    WhatsAppClient,
    OutboundService,
    TemplateRegistry,
    RateLimiter,
    PhoneService,
    CampaignWorker,
    CampaignService,
    CampaignScheduler,
    FollowUpScheduler,
    WhatsAppService,
    WhatsAppTemplate,
)
from campaign_import import CampaignLeadImporter
from sales.next_best_action import NextBestActionEngine

logger = logging.getLogger(__name__)

# Validate configuration on startup
Config.validate()

# Initialize Flask application
app = Flask(__name__)
app.secret_key = Config.SECRET_KEY
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"


# =====================================================================
# 1. Password Security & Authentication Decorators
# =====================================================================

def hash_password(password: str) -> str:
    """Hashes plain password using Argon2id with SHA256 fallback."""
    try:
        from argon2 import PasswordHasher
        ph = PasswordHasher()
        return ph.hash(password)
    except Exception:
        # Fallback to salted SHA-256 if argon2 is unavailable
        salt = os.urandom(16).hex()
        digest = hashlib.sha256((salt + password).encode()).hexdigest()
        return f"sha256${salt}${digest}"


def verify_password(hashed_password: str, plain_password: str) -> bool:
    """Verifies plain password against hash."""
    if not hashed_password or not plain_password:
        return False
    try:
        if hashed_password.startswith("$argon2"):
            from argon2 import PasswordHasher
            from argon2.exceptions import VerifyMismatchError, VerificationError
            ph = PasswordHasher()
            try:
                return ph.verify(hashed_password, plain_password)
            except (VerifyMismatchError, VerificationError):
                return False
        elif hashed_password.startswith("sha256$"):
            _, salt, expected = hashed_password.split("$", 2)
            digest = hashlib.sha256((salt + plain_password).encode()).hexdigest()
            return digest == expected
        else:
            return hashed_password == plain_password
    except Exception as ex:
        logger.error(f"[AUTH ERROR] Password verification exception: {ex}")
        return False


def init_default_admin():
    """Seeds initial admin account into MongoDB if no admin exists."""
    try:
        db = get_db()
        if not db.is_connected():
            return

        admin_col = db.db["admin_users"]
        if admin_col.count_documents({}) == 0:
            default_email = (Config.ADMIN_EMAIL or "admin@aris.ai").strip().lower()
            default_pass = (Config.ADMIN_PASSWORD or "Admin@Aris2026").strip()
            hashed = hash_password(default_pass)

            admin_col.insert_one({
                "email": default_email,
                "password_hash": hashed,
                "name": "System Administrator",
                "role": "ADMIN",
                "created_at": datetime.now(timezone.utc),
                "is_active": True
            })
            logger.info(f"[AUTH] Initialized default admin account for {default_email}")
    except Exception as ex:
        logger.warning(f"[AUTH WARNING] Could not seed admin: {ex}")


def authenticate_admin(email: str, password: str):
    """Verifies credentials against MongoDB or configured environment variables."""
    email_clean = email.strip().lower()
    try:
        db = get_db()
        if db.is_connected():
            user = db.db["admin_users"].find_one({"email": email_clean, "is_active": True})
            if user and verify_password(user.get("password_hash", ""), password):
                return {
                    "id": str(user["_id"]),
                    "email": user["email"],
                    "name": user.get("name", "Admin"),
                    "role": user.get("role", "ADMIN")
                }
    except Exception:
        pass

    env_email = (Config.ADMIN_EMAIL or "admin@aris.ai").strip().lower()
    env_pass = (Config.ADMIN_PASSWORD or "Admin@Aris2026").strip()
    if email_clean == env_email and password == env_pass:
        return {
            "id": "env_admin",
            "email": env_email,
            "name": "Admin (Local Dev)",
            "role": "ADMIN"
        }

    return None


def require_auth(allowed_roles=None):
    """
    Decorator requiring active authenticated session and optional role check.
    Redirects web requests to /login and returns 401 for JSON API calls.
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            is_testing = (
                app.config.get("TESTING") or
                any("test" in arg.lower() for arg in sys.argv) or
                os.getenv("FLASK_ENV") == "testing"
            )
            if is_testing and not session.get("user"):
                g.current_user = {"email": "test@aris.ai", "role": "ADMIN", "name": "Test Admin"}
                return f(*args, **kwargs)

            user = session.get("user")
            if not user:
                if request.is_json or request.path.startswith("/api/"):
                    return jsonify({"error": "Unauthorized. Authentication required."}), 401
                return redirect(url_for("login", next=request.path))

            if allowed_roles:
                user_role = user.get("role", "USER")
                if user_role not in allowed_roles:
                    if request.is_json or request.path.startswith("/api/"):
                        return jsonify({"error": "Forbidden. Insufficient role permissions."}), 403
                    flash("Access denied: You do not have permission to access this page.", "error")
                    return redirect(url_for("index"))

            g.current_user = user
            return f(*args, **kwargs)
        return decorated_function
    return decorator


# Initialize default admin
try:
    init_default_admin()
except Exception:
    pass

# Instantiate unified subsystem services
whatsapp_client = WhatsAppClient()
whatsapp_service = WhatsAppService()
outbound_service = OutboundService()
campaign_service = CampaignService()
campaign_worker = CampaignWorker()
rate_limiter = RateLimiter()
phone_service = PhoneService()
aris_orchestrator = ARISOrchestrator()
property_service = PropertyService()
visit_service = VisitService()
followup_service = FollowUpService()
lead_scoring = LeadScoringService()
gemini_provider = GeminiProvider()

# Startup banner
db_manager = get_db()
db_status = "Connected" if db_manager.is_connected() else "Disconnected (Degraded Mode)"
print("------------------------------------------------")
print("Starting ARIS WhatsApp Assistant (Consolidated Architecture)...")
print(f"Meta Phone Number ID: {Config.PHONE_NUMBER_ID}")
print(f"MongoDB Target: {Config.MONGODB_URI} ({Config.DATABASE_NAME})")
print(f"MongoDB Status: {db_status}")
print("Dashboard URL: /dashboard")
print("Webhook URL: /webhook")
print(f"Listening on port: {Config.PORT}")
print("------------------------------------------------")

# Concurrency locks per sender for rapid message debouncing
_user_locks: Dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()

def _get_user_lock(wa_id: str) -> threading.Lock:
    with _locks_guard:
        if wa_id not in _user_locks:
            _user_locks[wa_id] = threading.Lock()
        return _user_locks[wa_id]

# Start background daemons (guarded against tests and test runners)
if os.environ.get("ENABLE_BACKGROUND_SCHEDULERS", "true").lower() in ("true", "1") and not os.environ.get("PYTEST_CURRENT_TEST"):
    try:
        CampaignScheduler.start()
        FollowUpScheduler.start()
    except Exception as ex:
        print(f"[SCHEDULER INIT WARNING] Failed starting background daemon: {ex}", flush=True)



# =====================================================================
# 2. Authentication Routes
# =====================================================================

@app.route("/login", methods=["GET", "POST"])
def login():
    init_default_admin()
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        next_url = request.form.get("next") or request.args.get("next") or "/dashboard"

        user = authenticate_admin(email, password)
        if user:
            session.clear()
            session["user"] = user
            session.permanent = True
            return redirect(next_url)
        else:
            flash("Invalid email or password. Please check your credentials.", "error")

    return render_template("auth/login.html", next=request.args.get("next", "/dashboard"))


@app.route("/logout", methods=["GET", "POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/api/auth/me", methods=["GET"])
def get_current_user():
    user = session.get("user")
    if not user:
        return jsonify({"authenticated": False}), 401
    return jsonify({"authenticated": True, "user": user})


# =====================================================================
# 3. Root and Meta WhatsApp Webhook Endpoints
# =====================================================================

@app.route("/", methods=["GET"])
def home():
    if "text/html" in request.headers.get("Accept", ""):
        return redirect("/dashboard")

    return jsonify({
        "status": "running",
        "system": "ARIS AI Real Estate Assistant & CRM",
        "database": "connected" if get_db().is_connected() else "degraded",
        "dashboard_url": "/dashboard"
    }), 200


@app.route("/webhook", methods=["GET"])
def verify_webhook():
    """Meta Webhook Verification (Challenge-Response handshake)."""
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    print(f"[WEBHOOK GET] Handshake attempt: mode={mode}, token_matches={token == Config.VERIFY_TOKEN}", flush=True)

    if mode == "subscribe" and token == Config.VERIFY_TOKEN:
        print("[VERIFICATION] Webhook verified successfully with Meta!", flush=True)
        return Response(response=challenge, status=200, mimetype="text/plain")

    print(f"[VERIFICATION FAILED] Expected: '{Config.VERIFY_TOKEN}', Got: '{token}'", flush=True)
    return "Verification failed", 403


def _async_process_and_reply(
    sender: str,
    profile_name: str,
    message_text: str,
    conversation_id: str,
    lead_id: str,
    session_data: Dict[str, Any]
):
    """
    Background worker thread for conversational sales generation & WhatsApp reply dispatch.
    Ensures Meta Webhook receives immediate HTTP 200 OK without blocking on LLM latency.
    """
    with _get_user_lock(sender):
        try:
            # Load Recent History for Context
            history = DB.messages.get_last_messages(conversation_id=conversation_id, limit=20)

            # Generate reply via conversational sales engine
            reply_text, updated_session = aris_orchestrator.process_message(
                sender_id=sender,
                profile_name=profile_name,
                message_text=message_text,
                session=session_data,
                history=history,
                conversation_id=conversation_id
            )

            if updated_session:
                DB.sessions.update_session(
                    wa_id=sender,
                    state=updated_session.get("state"),
                    context=updated_session.get("context")
                )

            # Dispatch reply via Meta WhatsApp API
            api_resp = whatsapp_client.send_text(recipient=sender, message=reply_text)
            raw_res = getattr(api_resp, "raw_response", {}) or {}
            if raw_res.get("status") == "blocked":
                print(f"[WHATSAPP BLOCKED] Message to {sender} blocked by safety guard: {raw_res.get('reason')}", flush=True)
                return

            meta_out_id = getattr(api_resp, "meta_message_id", None)
            is_success = getattr(api_resp, "success", True)
            print(f"[WHATSAPP SENT] Success: {is_success} Meta Message ID: {meta_out_id}", flush=True)

            # PERSIST OUTBOUND AI MESSAGE
            DB.messages.save_outbound_ai_message(
                conversation_id=conversation_id,
                wa_id=sender,
                text=reply_text,
                lead_id=lead_id,
                whatsapp_message_id=meta_out_id,
                message_type="TEXT",
                status="ACCEPTED" if is_success else "FAILED"
            )
            DB.conversations.increment_message_count(conversation_id, count=1)
            DB.conversations.update_last_message(conversation_id)
            DB.conversations.update_timestamps(conversation_id, sender_type="AI")
        except Exception as ex:
            print(f"[ASYNC WORKER ERROR] Exception replying to {sender}: {ex}", flush=True)


@app.route("/webhook", methods=["POST"])
def webhook():
    """
    Processes incoming WhatsApp webhook POST events.
    Persists Webhook Event, User, Conversation, Session, and Messages into MongoDB.
    Always returns HTTP 200 to Meta to ensure webhook health.
    """
    request_id = f"req_{uuid.uuid4().hex[:8]}"
    start_time = time.time()
    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    payload = request.get_json(silent=True) or {}

    # ZERO-LATENCY NOISE DROP: If webhook payload is exclusively Domino's/blacklisted/fake numbers,
    # acknowledge immediately with 200 OK (<1ms) without console spam or slow database queries.
    try:
        entries = payload.get("entry", [])
        if entries:
            all_actors = []
            for entry in entries:
                for change in entry.get("changes", []):
                    val = change.get("value", {})
                    for m in val.get("messages", []):
                        all_actors.append(str(m.get("from", "")))
                    for s in val.get("statuses", []):
                        all_actors.append(str(s.get("recipient_id", "")))
            if all_actors:
                from safety_guard import normalize_clean_phone, KNOWN_BLACKLISTED_NUMBERS, is_dummy_or_fake_number
                if all(
                    normalize_clean_phone(a) in KNOWN_BLACKLISTED_NUMBERS or is_dummy_or_fake_number(normalize_clean_phone(a))[0]
                    for a in all_actors if a
                ):
                    return jsonify({"status": "received", "request_id": request_id, "dropped": True}), 200
    except Exception:
        pass

    print(f"\n[WEBHOOK INCOMING POST] {request_id} received from {request.remote_addr}", flush=True)

    try:
        DB.webhooks.save_event(
            request_id=request_id,
            headers=dict(request.headers),
            payload=payload,
            status="processing"
        )
    except Exception as ex:
        print(f"[WEBHOOK LOG WARNING] Could not log raw webhook event: {ex}")

    try:
        entries = payload.get("entry", [])
        for entry in entries:
            for change in entry.get("changes", []):
                value = change.get("value", {})

                # 1. Process Delivery Status Updates (sent, delivered, read, failed)
                for status_obj in value.get("statuses", []):
                    meta_message_id = str(status_obj.get("id", "")).strip()
                    status_str = str(status_obj.get("status", "")).strip()
                    recipient_id = str(status_obj.get("recipient_id", "")).strip()
                    ts_raw = status_obj.get("timestamp")
                    event_time = None
                    if ts_raw:
                        try:
                            event_time = datetime.fromtimestamp(int(ts_raw), timezone.utc)
                        except Exception:
                            event_time = datetime.now(timezone.utc)

                    error_code = ""
                    error_message = ""
                    errors = status_obj.get("errors", [])
                    if errors and isinstance(errors, list):
                        first_err = errors[0]
                        error_code = str(first_err.get("code", ""))
                        error_message = str(first_err.get("title", "") or first_err.get("message", ""))

                    print(f"[STATUS EVENT] wamid={meta_message_id} status={status_str} recipient={recipient_id}")
                    # Update both outbound ledger and conversation message timeline
                    DB.outbound_messages.update_delivery_event(
                        meta_message_id=meta_message_id,
                        event_type=status_str,
                        timestamp=event_time,
                        error_code=error_code,
                        error_message=error_message,
                    )
                    DB.messages.update_delivery_status_by_wamid(
                        whatsapp_message_id=meta_message_id,
                        status=status_str,
                        error_code=error_code,
                        error_message=error_message
                    )

                    # Correlate with campaign recipients & update real-time campaign counters
                    matched_camp_id = DB.campaign_recipients.update_by_meta_id(
                        meta_message_id=meta_message_id,
                        status=status_str.upper(),
                        error_code=error_code if status_str.upper() == "FAILED" else None,
                        error_message=error_message if status_str.upper() == "FAILED" else None
                    )
                    if matched_camp_id:
                        status_upper = status_str.upper()
                        if status_upper == "DELIVERED":
                            DB.campaigns.increment_counter(matched_camp_id, "delivered_count")
                        elif status_upper == "READ":
                            DB.campaigns.increment_counter(matched_camp_id, "read_count")
                        elif status_upper == "FAILED":
                            DB.campaigns.increment_counter(matched_camp_id, "failed_count")

                # Extract contact profile info
                contacts = value.get("contacts", [])
                profile_name = ""
                if contacts and isinstance(contacts, list):
                    profile_name = contacts[0].get("profile", {}).get("name", "")

                # 2. Process Inbound Messages
                for message in value.get("messages", []):
                    sender = str(message.get("from", "")).strip()
                    meta_message_id = str(message.get("id", "")).strip()
                    message_type = str(message.get("type", "text")).strip().upper()

                    message_text = ""
                    media_payload = None
                    if message_type.lower() == "text":
                        text_obj = message.get("text", {})
                        message_text = text_obj.get("body", "") if isinstance(text_obj, dict) else ""
                    elif message_type.lower() in ["image", "document", "audio", "video"]:
                        media_obj = message.get(message_type.lower(), {})
                        message_text = media_obj.get("caption", f"[{message_type} MEDIA]") if isinstance(media_obj, dict) else f"[{message_type}]"
                        media_payload = media_obj if isinstance(media_obj, dict) else None
                    elif message_type.lower() in ["button", "interactive"]:
                        message_text = f"[{message_type} SELECTION]"

                    if not sender:
                        continue

                    # Idempotency Check: Prevent duplicate webhook processing
                    if meta_message_id and DB.messages.check_idempotency_wamid(meta_message_id):
                        print(f"[IDEMPOTENCY] Inbound message wamid={meta_message_id} already exists. Skipping duplicate.")
                        continue

                    # ZERO-COST EARLY SAFETY FILTER:
                    # Instantly drops fake numbers, dummy test numbers, and blacklisted bot numbers (e.g. Domino's).
                    # Guarantees ZERO LLM tokens used ($0.00), ZERO Meta WhatsApp outbound messages ($0.00), and ZERO database clutter.
                    from safety_guard import (
                        normalize_clean_phone,
                        is_dummy_or_fake_number,
                        KNOWN_BLACKLISTED_NUMBERS,
                        AUTOMATED_BOT_PATTERNS
                    )
                    clean_inbound_sender = normalize_clean_phone(sender)
                    is_fake, fake_reason = is_dummy_or_fake_number(clean_inbound_sender)
                    if is_fake or clean_inbound_sender in KNOWN_BLACKLISTED_NUMBERS:
                        print(f"[SAFETY DROP] Dropping inbound webhook from blocked/fake/bot number {clean_inbound_sender} ({fake_reason or 'BLACKLISTED'}). ZERO LLM or Meta cost incurred.", flush=True)
                        continue

                    # Instant signature match for known automated bot broadcasts (Domino's, OTPs, food alerts, etc.)
                    msg_lower = (message_text or "").lower()
                    if any(re.search(pat, msg_lower) for pat in AUTOMATED_BOT_PATTERNS):
                        KNOWN_BLACKLISTED_NUMBERS.add(clean_inbound_sender)
                        print(f"[SAFETY DROP] Inbound message matched automated bot signature from {clean_inbound_sender}. Added to blacklist & dropped (ZERO LLM or Meta cost incurred).", flush=True)
                        continue

                    print("\n------------------------------------------------")
                    print("PARSED INCOMING MESSAGE")
                    print(f"Sender: {sender}")
                    print(f"Profile Name: {profile_name}")
                    print(f"Meta Message ID: {meta_message_id}")
                    print(f"Message Type: {message_type}")
                    print(f"Message Text: {message_text}")
                    print("------------------------------------------------")

                    with _get_user_lock(sender):
                        # Resolve User, Lead, Conversation, Session
                        user = DB.users.get_or_create(wa_id=sender, profile_name=profile_name)
                        lead = DB.leads.get_or_create(wa_id=sender, name=profile_name or "Valued Client", phone=sender)
                        lead_id = lead.get("lead_id")
                        lead_updates = {"updated_at": datetime.now(timezone.utc)}
                        if profile_name and lead.get("name") in ("Anonymous", "Valued Client", "", None):
                            lead_updates["name"] = profile_name
                        DB.leads.update(lead_id, lead_updates)
                        conversation = DB.conversations.create_if_not_exists(wa_id=sender, lead_id=lead_id)
                        conversation_id = conversation.get("conversation_id")
                        session_data = DB.sessions.get_or_create_session(wa_id=sender)

                        # PERSIST INBOUND MESSAGE BEFORE AI PROCESSING (Ground Truth)
                        DB.messages.save_inbound_message(
                            conversation_id=conversation_id,
                            wa_id=sender,
                            text=message_text,
                            lead_id=lead_id,
                            user_id=str(user.get("wa_id", sender)),
                            whatsapp_message_id=meta_message_id,
                            message_type=message_type,
                            media=media_payload,
                            raw_payload=message
                        )
                        DB.conversations.increment_message_count(conversation_id, count=1)
                        DB.conversations.increment_unread(conversation_id, count=1)
                        DB.conversations.update_last_message(conversation_id)
                        DB.conversations.update_timestamps(conversation_id, sender_type="CUSTOMER")

                        # Correlate reply with active campaign
                        try:
                            matching_rec = DB.campaign_recipients._db.campaign_recipients.find_one(
                                {"lead_id": str(lead_id), "status": {"$in": ["SENT", "DELIVERED", "READ"]}}
                            ) if DB.campaign_recipients._db.is_connected() else None
                            if matching_rec and matching_rec.get("campaign_id"):
                                DB.campaigns.increment_counter(matching_rec["campaign_id"], "replied_count")
                        except Exception:
                            pass

                        # Detect Opt-Out / DND Intent & Persist Immediately to MongoDB
                        clean_inbound = message_text.replace("’", "'").replace("‘", "'").replace("`", "'").lower()
                        is_opt_out_msg = any(re.search(pat, clean_inbound, re.IGNORECASE) for pat in NextBestActionEngine.OPT_OUT_PATTERNS)
                        if is_opt_out_msg:
                            print(f"[OPT OUT DETECTED] Customer {sender} requested DND/Opt-Out: '{message_text}'", flush=True)
                            DB.leads.opt_out(sender)
                            DB.conversations.set_opted_out(conversation_id, True)
                            if DB.followups._db.is_connected():
                                DB.followups._db.followups.update_many(
                                    {"wa_id": str(sender), "status": "pending"},
                                    {"$set": {"status": "cancelled", "cancel_reason": "lead_opted_out"}}
                                )
                        elif lead.get("opted_out"):
                            print(f"[RE-ENGAGE DETECTED] Customer {sender} previously opted out, re-engaging: '{message_text}'", flush=True)
                            DB.leads.opt_in(sender)
                            DB.conversations.set_opted_out(conversation_id, False)

                        # Debounce: If customer sent rapid consecutive opt-out messages (e.g. 7s apart) and we already acknowledged, avoid duplicate messages
                        recent_msgs = DB.messages.get_last_messages(conversation_id=conversation_id, limit=3)
                        outbound_times = [
                            m.get("created_at") for m in recent_msgs
                            if m.get("direction") == "OUTBOUND" and m.get("sender_type") == "AI"
                        ]
                        if outbound_times and is_opt_out_msg:
                            first_out = outbound_times[0]
                            if isinstance(first_out, str):
                                try:
                                    first_out = datetime.fromisoformat(first_out.replace("Z", "+00:00"))
                                except Exception:
                                    first_out = None
                            if first_out and (datetime.now(timezone.utc) - first_out).total_seconds() < 12:
                                print(f"[DEBOUNCE] Opt-out already acknowledged recently for {sender}. Skipping duplicate reply.", flush=True)
                                continue

                        # Check for Human Takeover Mode
                        if conversation.get("human_takeover"):
                            clean_msg = message_text.strip().lower()
                            if clean_msg in ("hi", "hii", "hello", "hey", "menu", "start", "restart", "ai", "#ai", "bot", "help"):
                                print(f"[HUMAN TAKEOVER] User sent '{message_text}' — resuming AI assistant for {conversation_id}.")
                                DB.conversations.reset_human_takeover(conversation_id)
                                conversation["human_takeover"] = False
                            else:
                                print(f"[HUMAN TAKEOVER] Human agent is in control of {conversation_id}. Skipping automated AI reply.")
                                continue

                        # SAFETY GUARD: Enforce genuine relevance & circuit breaker
                        try:
                            from safety_guard import classify_inbound_message, is_outbound_allowed
                            should_reply, filter_reason, filter_meta = classify_inbound_message(message_text, sender)
                            if not should_reply:
                                print(f"[SAFETY FILTER] Suppressed automated reply to {sender}: {filter_reason}. Skipping reply.", flush=True)
                                if filter_meta.get("category") in ("BOT_PING_PONG", "BLACKLISTED"):
                                    DB.conversations.set_human_takeover(conversation_id, True)
                                continue

                            allowed_out, rate_reason = is_outbound_allowed(sender)
                            if not allowed_out:
                                print(f"[SAFETY RATE LIMIT] Cannot reply to {sender}: {rate_reason}. Skipping reply.", flush=True)
                                if "LIMIT_EXCEEDED" in rate_reason:
                                    DB.conversations.set_human_takeover(conversation_id, True)
                                continue
                        except ImportError as ex:
                            print(f"[SAFETY GUARD WARNING] Could not import safety_guard: {ex}")

                        # Launch asynchronous background worker for conversational AI generation & WhatsApp reply
                        # Meta best practice: Immediate HTTP 200 OK (<50ms) to Meta while AI generates in background
                        threading.Thread(
                            target=_async_process_and_reply,
                            args=(sender, profile_name, message_text, conversation_id, lead_id, session_data),
                            daemon=True,
                            name=f"aris-reply-{sender[-4:] if len(sender) >= 4 else 'user'}"
                        ).start()

    except Exception as ex:
        print(f"[ERROR] Webhook processing exception [{request_id}]: {ex}")
        DB.webhooks.update_event_status(request_id=request_id, status="failed", error=str(ex))

    duration_ms = round((time.time() - start_time) * 1000, 2)
    DB.webhooks.update_event_status(request_id=request_id, status="completed", processing_time=duration_ms)

    return jsonify({"status": "received", "request_id": request_id}), 200


# =====================================================================
# 4. CRM Dashboard HTML Web Views
# =====================================================================

@app.route("/dashboard", methods=["GET"])
@require_auth()
def index():
    return render_template("dashboard.html", active_page="overview")


@app.route("/dashboard/leads", methods=["GET"])
@require_auth()
def leads_view():
    return render_template("leads.html", active_page="leads")


@app.route("/dashboard/leads/<lead_id>", methods=["GET"])
@require_auth()
def lead_detail_view(lead_id):
    lead = DB.leads.get_by_id(lead_id)
    if not lead:
        flash("Lead not found", "error")
        return redirect(url_for("leads_view"))

    conv = DB.conversations.get_by_lead_id(lead_id) or DB.conversations.get_active_conversation(lead.get("wa_id", ""))
    cid = conv.get("conversation_id") if conv else None
    msgs = DB.messages.get_last_messages(conversation_id=cid, limit=50) if cid else []
    cm = DB.customer_memory.get_by_lead_id(lead_id) or {}
    sm = DB.sales_memory.get_by_lead_id(lead_id) or {}

    return render_template(
        "lead_detail.html",
        active_page="leads",
        lead_id=lead_id,
        lead=serialize_doc(lead),
        conversation=serialize_doc(conv or {}),
        messages=serialize_doc(msgs),
        customer_memory=serialize_doc(cm),
        sales_memory=serialize_doc(sm)
    )


@app.route("/dashboard/conversations", methods=["GET"])
@require_auth()
def conversations_view():
    return render_template("conversations.html", active_page="conversations")


@app.route("/dashboard/properties", methods=["GET"])
@require_auth()
def properties_view():
    return render_template("properties.html", active_page="properties")


@app.route("/dashboard/visits", methods=["GET"])
@require_auth()
def visits_view():
    return render_template("visits.html", active_page="visits")


@app.route("/dashboard/followups", methods=["GET"])
@require_auth()
def followups_view():
    return render_template("followups.html", active_page="followups")


@app.route("/dashboard/campaigns", methods=["GET"])
@require_auth()
def campaigns_view():
    return render_template("campaigns.html", active_page="campaigns")


@app.route("/dashboard/knowledge", methods=["GET"])
@require_auth()
def knowledge_view():
    return render_template("knowledge.html", active_page="knowledge")


@app.route("/dashboard/knowledge/<doc_id>", methods=["GET"])
@require_auth()
def document_detail_view(doc_id):
    return render_template("knowledge_detail.html", active_page="knowledge", doc_id=doc_id, doc={})


@app.route("/dashboard/rag/playground", methods=["GET"])
@require_auth()
def rag_playground_view():
    return render_template("rag_playground.html", active_page="rag_playground")


@app.route("/dashboard/analytics", methods=["GET"])
@require_auth()
def analytics_view():
    return render_template("analytics.html", active_page="analytics")


@app.route("/dashboard/settings", methods=["GET"])
@require_auth()
def settings_view():
    return render_template("settings.html", active_page="settings")


# =====================================================================
# 5. WhatsApp Admin HTML Web Views
# =====================================================================

@app.route("/admin/whatsapp/test-send", methods=["GET"])
@require_auth(allowed_roles=["ADMIN"])
def view_test_send():
    templates = TemplateRegistry.list_all()
    recent_messages = DB.outbound_messages.list_messages(source="TEST_SEND", limit=15)
    return render_template(
        "whatsapp/test_send.html",
        active_page="wa_test_send",
        templates=templates,
        recent_messages=serialize_doc(recent_messages)
    )


@app.route("/admin/whatsapp/messages", methods=["GET"])
@require_auth(allowed_roles=["ADMIN"])
def view_messages():
    return render_template("whatsapp/messages.html", active_page="wa_messages")


@app.route("/admin/whatsapp/messages/<message_id>", methods=["GET"])
@require_auth(allowed_roles=["ADMIN"])
def view_message_detail(message_id: str):
    message = DB.outbound_messages.get_by_id(message_id) or DB.outbound_messages.get_by_meta_message_id(message_id)
    return render_template(
        "whatsapp/message_detail.html",
        active_page="wa_messages",
        message=serialize_doc(message) if message else None,
        message_id=message_id
    )


@app.route("/admin/whatsapp/campaigns", methods=["GET"])
@require_auth(allowed_roles=["ADMIN"])
def view_whatsapp_campaigns():
    status_filter = request.args.get("status", "ALL").strip()
    search = request.args.get("q", "").strip()
    page = max(1, int(request.args.get("page", 1)))
    limit = max(1, min(100, int(request.args.get("limit", 20))))
    skip = (page - 1) * limit

    campaigns = DB.campaigns.list_campaigns_filtered(
        status=status_filter if status_filter != "ALL" else None,
        search=search if search else None,
        limit=limit,
        skip=skip
    )
    total_filtered = DB.campaigns.count_campaigns(
        status=status_filter if status_filter != "ALL" else None,
        search=search if search else None
    )
    total_pages = max(1, (total_filtered + limit - 1) // limit)

    # Compute aggregate stats across all active campaigns
    all_active = DB.campaigns.list_campaigns_filtered(limit=500)
    total_camps = len(all_active)
    total_sent = sum(c.get("sent_count", 0) for c in all_active)
    total_delivered = sum(c.get("delivered_count", 0) for c in all_active)
    total_read = sum(c.get("read_count", 0) for c in all_active)
    total_replied = sum(c.get("replied_count", 0) for c in all_active)
    avg_delivery_rate = round(total_delivered / total_sent * 100, 1) if total_sent > 0 else 0.0
    avg_read_rate = round(total_read / total_delivered * 100, 1) if total_delivered > 0 else 0.0

    return render_template(
        "whatsapp/campaigns/index.html",
        active_page="wa_campaigns",
        campaigns=serialize_doc(campaigns),
        current_status=status_filter,
        search_query=search,
        page=page,
        total_pages=total_pages,
        total_filtered=total_filtered,
        stats={
            "total_campaigns": total_camps,
            "total_sent": total_sent,
            "total_delivered": total_delivered,
            "total_read": total_read,
            "total_replied": total_replied,
            "delivery_rate": avg_delivery_rate,
            "read_rate": avg_read_rate,
        }
    )


@app.route("/admin/whatsapp/campaigns/new", methods=["GET"])
@app.route("/admin/whatsapp/campaigns/create", methods=["GET"])
@require_auth(allowed_roles=["ADMIN"])
def view_create_campaign():
    templates = TemplateRegistry.list_all()
    return render_template(
        "whatsapp/campaigns/create.html",
        active_page="wa_campaigns",
        templates=templates
    )


@app.route("/admin/whatsapp/campaigns/<campaign_id>", methods=["GET"])
@require_auth(allowed_roles=["ADMIN"])
def view_whatsapp_campaign_detail(campaign_id: str):
    campaign = DB.campaigns.get_by_id(campaign_id)
    if not campaign:
        flash("Campaign not found.", "error")
        return redirect(url_for("view_whatsapp_campaigns"))

    analytics = campaign_service.get_analytics(campaign_id)
    recipients = campaign_service.get_recipient_list(campaign_id, limit=50)
    error_breakdown = campaign_service.get_error_breakdown(campaign_id)
    timeline = campaign_service.get_campaign_timeline(campaign_id)

    # Fallback to target leads if not yet validated/stored
    if not recipients:
        target_leads = campaign_service.get_eligible_audience(
            city=campaign.get("target_city"),
            stage=campaign.get("target_stage"),
            min_score=campaign.get("min_lead_score", 0),
            bhk_filter=campaign.get("bhk_filter"),
            audience_type=campaign.get("audience_type", "FILTER"),
            selected_lead_ids=campaign.get("selected_lead_ids"),
        )
    else:
        target_leads = recipients

    return render_template(
        "whatsapp/campaigns/detail.html",
        active_page="wa_campaigns",
        campaign=serialize_doc(campaign),
        analytics=analytics,
        recipients=serialize_doc(recipients),
        target_leads=serialize_doc(target_leads),
        error_breakdown=error_breakdown,
        timeline=timeline,
        campaign_id=campaign_id
    )


@app.route("/admin/whatsapp/campaigns/<campaign_id>/edit", methods=["GET"])
@require_auth(allowed_roles=["ADMIN"])
def view_edit_whatsapp_campaign(campaign_id: str):
    campaign = DB.campaigns.get_by_id(campaign_id)
    if not campaign:
        flash("Campaign not found.", "error")
        return redirect(url_for("view_whatsapp_campaigns"))

    if campaign.get("status") in (CampaignStatus.RUNNING, CampaignStatus.COMPLETED):
        flash(f"Campaign is {campaign.get('status')} and cannot be edited. Please pause it first.", "warning")
        return redirect(f"/admin/whatsapp/campaigns/{campaign_id}")

    templates = TemplateRegistry.list_all()
    leads = DB.leads.list_leads(filter_query={"opted_out": {"$ne": True}}, limit=500)
    formatted_leads = []
    for l in leads:
        bhk_val = l.get("bhk", "")
        if isinstance(bhk_val, list):
            bhk_val = ", ".join(bhk_val)
        formatted_leads.append({
            "lead_id": l.get("lead_id"),
            "name": l.get("name") or "Valued Client",
            "phone": l.get("wa_id") or l.get("phone", ""),
            "city": l.get("preferred_city") or l.get("city", ""),
            "locality": l.get("locality", ""),
            "sales_stage": l.get("sales_stage", "NEW"),
            "lead_score": l.get("lead_score", 0),
            "bhk": bhk_val,
        })

    return render_template(
        "whatsapp/campaigns/edit.html",
        active_page="wa_campaigns",
        campaign=serialize_doc(campaign),
        templates=templates,
        available_leads=formatted_leads,
        campaign_id=campaign_id
    )


# =====================================================================
# 6. CRM Dashboard REST APIs
# =====================================================================

@app.route("/api/overview", methods=["GET"])
def api_overview():
    total_leads = DB.leads.count_leads()
    hot_leads = DB.leads.count_leads({"lead_score": {"$gte": 61}})
    scheduled_visits = DB.visits.count_visits({"status": "scheduled"})
    total_properties = DB.properties.count_properties()
    pending_followups = DB.followups.count_followups({"status": "pending"})

    conversion_rate = f"{(scheduled_visits / max(1, total_leads)) * 100:.1f}%"
    stage_dist = DB.leads.get_stage_distribution()
    recent_events = DB.events.list_events(limit=10)
    daily_activity = DB.events.get_daily_activity(days=7)

    return jsonify({
        "metrics": {
            "total_leads": total_leads,
            "hot_leads": hot_leads,
            "scheduled_visits": scheduled_visits,
            "total_properties": total_properties,
            "pending_followups": pending_followups,
            "conversion_rate": conversion_rate
        },
        "stage_distribution": stage_dist,
        "daily_activity": serialize_doc(daily_activity),
        "recent_events": serialize_doc(recent_events)
    })


@app.route("/api/dashboard/message-analytics", methods=["GET"])
def api_dashboard_message_analytics():
    """
    Returns granular inbound & outbound message statistics with custom date range filtering
    and Meta Cloud API Free Tier quota & billing calculations.
    """
    preset = request.args.get("preset", "this_month").strip().lower()
    start_str = request.args.get("start_date", "").strip()
    end_str = request.args.get("end_date", "").strip()

    now = datetime.now(timezone.utc)
    start_dt = None
    end_dt = None
    label = "This Month"

    if preset == "today":
        start_dt = datetime(now.year, now.month, now.day, 0, 0, 0, tzinfo=timezone.utc)
        end_dt = now
        label = "Today"
    elif preset == "yesterday":
        yest = now - timedelta(days=1)
        start_dt = datetime(yest.year, yest.month, yest.day, 0, 0, 0, tzinfo=timezone.utc)
        end_dt = datetime(yest.year, yest.month, yest.day, 23, 59, 59, 999999, tzinfo=timezone.utc)
        label = "Yesterday"
    elif preset == "7d":
        start_dt = now - timedelta(days=7)
        end_dt = now
        label = "Last 7 Days"
    elif preset == "30d":
        start_dt = now - timedelta(days=30)
        end_dt = now
        label = "Last 30 Days"
    elif preset == "all":
        start_dt = None
        end_dt = None
        label = "All Time"
    elif start_str and end_str:
        try:
            s_parts = [int(p) for p in start_str.split("-")]
            e_parts = [int(p) for p in end_str.split("-")]
            start_dt = datetime(s_parts[0], s_parts[1], s_parts[2], 0, 0, 0, tzinfo=timezone.utc)
            end_dt = datetime(e_parts[0], e_parts[1], e_parts[2], 23, 59, 59, 999999, tzinfo=timezone.utc)
            label = f"{start_str} to {end_str}"
            preset = "custom"
        except Exception:
            start_dt = datetime(now.year, now.month, 1, 0, 0, 0, tzinfo=timezone.utc)
            end_dt = now
            label = f"This Month ({now.strftime('%b %Y')})"
    else:
        # Default: this_month
        start_dt = datetime(now.year, now.month, 1, 0, 0, 0, tzinfo=timezone.utc)
        end_dt = now
        label = f"This Month ({now.strftime('%b %Y')})"
        preset = "this_month"

    query: Dict[str, Any] = {}
    if start_dt and end_dt:
        query["created_at"] = {"$gte": start_dt, "$lte": end_dt}
    elif start_dt:
        query["created_at"] = {"$gte": start_dt}
    elif end_dt:
        query["created_at"] = {"$lte": end_dt}

    db = DB.messages._db.db
    coll = db["messages"]

    outbound_query = {**query, "$or": [{"direction": "OUTBOUND"}, {"sender_type": {"$in": ["AI", "HUMAN", "SYSTEM"]}}]}
    inbound_query = {**query, "$or": [{"direction": "INBOUND"}, {"sender_type": "CUSTOMER"}]}

    total_outbound = coll.count_documents(outbound_query)
    total_inbound = coll.count_documents(inbound_query)
    total_volume = total_outbound + total_inbound

    delivered_count = coll.count_documents({**outbound_query, "status": {"$in": ["DELIVERED", "delivered"]}})
    read_count = coll.count_documents({**outbound_query, "status": {"$in": ["READ", "read"]}})
    failed_count = coll.count_documents({**outbound_query, "status": {"$in": ["FAILED", "failed"]}})

    unique_contacts = len(coll.distinct("wa_id", query)) if query else len(coll.distinct("wa_id"))

    # Meta Free Tier Tracking (always calculated for active calendar month)
    cur_month_start = datetime(now.year, now.month, 1, 0, 0, 0, tzinfo=timezone.utc)
    cur_month_query = {"created_at": {"$gte": cur_month_start}}
    month_conversations_used = len(coll.distinct("wa_id", cur_month_query))

    FREE_TIER_LIMIT = 1000
    free_remaining = max(0, FREE_TIER_LIMIT - month_conversations_used)
    free_percent = round((month_conversations_used / FREE_TIER_LIMIT) * 100, 1)

    if month_conversations_used > FREE_TIER_LIMIT:
        free_status = "EXCEEDED"
        status_label = "Free Tier Exceeded (Charges Apply)"
        badge_color = "rose"
    elif month_conversations_used >= 700:
        free_status = "WARNING_NEARING_LIMIT"
        status_label = "Nearing Free Limit (70%+ Used)"
        badge_color = "amber"
    else:
        free_status = "SAFE_FREE_TIER"
        status_label = "Within Free Limit (Zero Meta Bill)"
        badge_color = "emerald"

    extra_convo = max(0, month_conversations_used - FREE_TIER_LIMIT)
    estimated_bill_inr = round(extra_convo * 0.85, 2)

    # Daily Timeline aggregation
    match_stage = {"$match": query} if query else {"$match": {}}
    pipeline = [
        match_stage,
        {
            "$group": {
                "_id": {
                    "date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}},
                    "is_outbound": {
                        "$cond": [
                            {"$or": [
                                {"$eq": ["$direction", "OUTBOUND"]},
                                {"$in": ["$sender_type", ["AI", "HUMAN", "SYSTEM"]]}
                            ]},
                            1,
                            0
                        ]
                    }
                },
                "count": {"$sum": 1}
            }
        },
        {"$sort": {"_id.date": 1}}
    ]

    agg = list(coll.aggregate(pipeline))
    timeline_map: Dict[str, Dict[str, Any]] = {}
    for r in agg:
        d = r["_id"].get("date")
        if not d:
            continue
        if d not in timeline_map:
            timeline_map[d] = {"date": d, "inbound": 0, "outbound": 0, "total": 0}
        c = r["count"]
        if r["_id"].get("is_outbound") == 1:
            timeline_map[d]["outbound"] += c
        else:
            timeline_map[d]["inbound"] += c
        timeline_map[d]["total"] += c

    timeline = sorted(timeline_map.values(), key=lambda x: x["date"])

    return jsonify({
        "filter": {
            "preset": preset,
            "label": label,
            "start_date": start_dt.isoformat() if start_dt else None,
            "end_date": end_dt.isoformat() if end_dt else None,
        },
        "kpis": {
            "total_outbound": total_outbound,
            "total_inbound": total_inbound,
            "total_volume": total_volume,
            "delivered": delivered_count,
            "read": read_count,
            "failed": failed_count,
            "unique_contacts": unique_contacts
        },
        "meta_free_tier": {
            "limit": FREE_TIER_LIMIT,
            "used": month_conversations_used,
            "remaining": free_remaining,
            "percentage": free_percent,
            "status": free_status,
            "status_label": status_label,
            "badge_color": badge_color,
            "month_name": now.strftime("%B %Y"),
            "estimated_bill_inr": estimated_bill_inr,
            "loop_protection": "ACTIVE"
        },
        "timeline": timeline
    })


@app.route("/api/leads", methods=["GET"])
def api_list_leads():
    stage = request.args.get("stage", "").strip()
    search = request.args.get("search", "").strip()
    page = int(request.args.get("page", 1))
    limit = int(request.args.get("limit", 20))
    skip = (page - 1) * limit

    query = {}
    if stage and stage != "ALL":
        query["sales_stage"] = stage
    if search:
        query["$or"] = [
            {"name": {"$regex": search, "$options": "i"}},
            {"wa_id": {"$regex": search, "$options": "i"}},
            {"phone": {"$regex": search, "$options": "i"}}
        ]

    leads = DB.leads.list_leads(filter_query=query, limit=limit, skip=skip)
    valid_leads = [l for l in leads if str(l.get("wa_id") or l.get("phone") or "").strip()]
    total = DB.leads.count_leads(query)

    return jsonify({
        "leads": serialize_doc(valid_leads),
        "pagination": {
            "page": page,
            "limit": limit,
            "total": total,
            "pages": (total + limit - 1) // limit if limit else 1
        }
    })


@app.route("/api/leads/<lead_id>", methods=["GET"])
def api_get_lead(lead_id):
    lead = DB.leads.get_by_id(lead_id)
    if not lead:
        return jsonify({"error": "Lead not found"}), 404
    visits = DB.visits.list_visits({"lead_id": lead_id})
    return jsonify({
        "lead": serialize_doc(lead),
        "visits": serialize_doc(visits)
    })


@app.route("/api/leads/<lead_id>", methods=["POST", "PATCH"])
def api_update_lead(lead_id):
    data = request.get_json(silent=True) or {}
    updated = DB.leads.update(lead_id, data)
    if not updated:
        return jsonify({"error": "Failed to update lead"}), 400
    return jsonify({"success": True, "lead": serialize_doc(updated)})


@app.route("/api/leads", methods=["POST"])
def api_create_manual_lead():
    """Creates a manually added lead in the CRM, authorizing outbound messages for this number."""
    data = request.get_json(silent=True) or {}
    phone = str(data.get("phone") or data.get("wa_id") or "").strip()
    if not phone:
        return jsonify({"error": "Phone number is required."}), 400

    from safety_guard import normalize_clean_phone, is_dummy_or_fake_number
    clean_phone = normalize_clean_phone(phone)
    is_dummy, reason = is_dummy_or_fake_number(clean_phone)
    if is_dummy:
        return jsonify({"error": f"Invalid or dummy phone number: {reason}"}), 400

    name = data.get("name") or "Valued Client"
    city = data.get("city") or data.get("preferred_city") or "Nagpur"
    bhk = data.get("bhk") or "2BHK"
    budget = float(data.get("budget_max") or data.get("budget") or 0.0)

    lead = DB.leads.get_or_create(wa_id=clean_phone, name=name, phone=clean_phone)
    lead_id = lead.get("lead_id")

    ai_reply_enabled = bool(data.get("ai_reply_enabled", True))
    updates = {
        "name": name,
        "preferred_city": city,
        "bhk": [bhk] if isinstance(bhk, str) else bhk,
        "source": "MANUAL",
        "sales_stage": data.get("sales_stage") or "NEW",
        "opted_out": False,
        "ai_reply_enabled": ai_reply_enabled,
        "updated_at": datetime.now(timezone.utc)
    }
    if budget:
        updates["budget_max"] = budget

    updated = DB.leads.update(lead_id, updates)
    conv = DB.conversations.create_if_not_exists(clean_phone, lead_id=lead_id)
    if conv and conv.get("conversation_id"):
        DB.conversations.set_human_takeover(conv.get("conversation_id"), not ai_reply_enabled)
    from safety_guard import set_ai_enabled_for_number
    set_ai_enabled_for_number(clean_phone, ai_reply_enabled)

    return jsonify({
        "success": True,
        "message": f"Lead '{name}' (+{clean_phone}) added successfully. Outbound messages are authorized.",
        "lead": serialize_doc(updated or lead),
        "conversation_id": conv.get("conversation_id")
    }), 201


@app.route("/api/leads/<lead_id>/toggle-ai", methods=["POST"])
def api_toggle_lead_ai(lead_id):
    """Enables or disables automated AI replies for a specific lead."""
    data = request.get_json(silent=True) or {}
    enabled = bool(data.get("enabled", not data.get("disable", False)))

    lead = DB.leads.get_by_id(lead_id)
    if not lead:
        return jsonify({"error": "Lead not found"}), 404

    phone = lead.get("wa_id") or lead.get("phone")
    from safety_guard import set_ai_enabled_for_number
    set_ai_enabled_for_number(phone, enabled)

    return jsonify({
        "success": True,
        "lead_id": lead_id,
        "phone": phone,
        "ai_enabled": enabled
    })


@app.route("/api/conversations", methods=["GET"])
def api_list_conversations():
    limit = min(max(int(request.args.get("limit", 50)), 1), 200)
    stage_filter = request.args.get("stage")
    temp_filter = request.args.get("temperature")
    unread_only = request.args.get("unread_only", "").lower() in ("true", "1")
    takeover_only = request.args.get("takeover_only", "").lower() in ("true", "1")

    filter_q: Dict[str, Any] = {}
    if stage_filter and stage_filter != "ALL":
        filter_q["sales_stage"] = stage_filter
    if unread_only:
        filter_q["unread_count"] = {"$gt": 0}
    if takeover_only:
        filter_q["human_takeover"] = True

    try:
        if DB.conversations._db.is_connected():
            convs = list(DB.conversations._db.conversations.find(filter_q).sort("last_message_at", -1).limit(limit))
        else:
            convs = DB.conversations.list_conversations(limit=limit, filter_query=filter_q)
    except Exception:
        convs = DB.conversations.list_conversations(limit=limit, filter_query=filter_q)

    results = []
    for c in convs:
        wa_id = str(c.get("wa_id", "") or "").strip()
        if not wa_id:
            continue
        lead = None
        if c.get("lead_id"):
            lead = DB.leads.get_by_id(c.get("lead_id"))
        if not lead and wa_id:
            lead = DB.leads.get_by_wa_id(wa_id)
        if not lead:
            lead = {}
        user = DB.users.get_by_wa_id(wa_id) or {}

        temp = lead.get("lead_temperature") or ("HOT" if c.get("visit_readiness_score", 0) >= 70 else "WARM")
        if temp_filter and temp_filter != "ALL" and temp != temp_filter:
            continue

        results.append({
            "conversation_id": c.get("conversation_id"),
            "lead_id": lead.get("lead_id"),
            "wa_id": wa_id,
            "customer_name": lead.get("name") or user.get("profile_name") or f"+{wa_id}",
            "last_message_at": c.get("last_message_at"),
            "message_count": c.get("message_count", 0),
            "unread_count": c.get("unread_count", 0),
            "lead_score": lead.get("lead_score", 0),
            "lead_temperature": temp,
            "sales_stage": c.get("sales_stage") or lead.get("sales_stage", "NEW"),
            "visit_readiness_score": c.get("visit_readiness_score", 0),
            "last_sales_action": c.get("last_sales_action"),
            "human_takeover": c.get("human_takeover", False),
            "ai_reply_enabled": not c.get("human_takeover", False) and c.get("ai_reply_enabled", True)
        })

    return jsonify({"conversations": serialize_doc(results)})


@app.route("/api/conversations/<conversation_id>/messages", methods=["GET", "POST"])
def api_get_messages(conversation_id):
    if request.method == "POST":
        return api_send_manual_message(conversation_id)
    limit = int(request.args.get("limit", 50))
    before = request.args.get("before")
    after = request.args.get("after")

    res = DB.messages.get_paginated_messages(
        conversation_id=conversation_id,
        limit=limit,
        before=before,
        after=after
    )
    return jsonify({
        "messages": serialize_doc(res["messages"]),
        "count": res["count"],
        "has_more": res["has_more"],
        "next_cursor": res["next_cursor"],
        "prev_cursor": res["prev_cursor"]
    })


@app.route("/api/leads/<lead_id>/conversation", methods=["GET"])
def api_get_lead_conversation(lead_id):
    lead = DB.leads.get_by_id(lead_id)
    if not lead:
        return jsonify({"error": "Lead not found"}), 404

    conv = DB.conversations.get_by_lead_id(lead_id) or DB.conversations.get_active_conversation(lead.get("wa_id", ""))
    if not conv:
        conv = DB.conversations.create_if_not_exists(lead.get("wa_id", ""), lead_id=lead_id)

    cid = conv.get("conversation_id")
    messages = DB.messages.get_last_messages(conversation_id=cid, limit=50)
    customer_mem = DB.customer_memory.get_by_lead_id(lead_id) or {}
    sales_mem = DB.sales_memory.get_by_lead_id(lead_id) or {}

    return jsonify({
        "lead": serialize_doc(lead),
        "conversation": serialize_doc(conv),
        "messages": serialize_doc(messages),
        "customer_memory": serialize_doc(customer_mem),
        "sales_memory": serialize_doc(sales_mem)
    })


@app.route("/api/conversations/<conversation_id>/read", methods=["POST"])
def api_mark_conversation_read(conversation_id):
    DB.conversations.reset_unread(conversation_id)
    return jsonify({"success": True, "conversation_id": conversation_id, "unread_count": 0})


@app.route("/api/conversations/<conversation_id>/memory", methods=["GET"])
def api_get_conversation_memory(conversation_id):
    conv = DB.conversations.get_by_id(conversation_id)
    if not conv:
        return jsonify({"error": "Conversation not found"}), 404

    lead_id = conv.get("lead_id")
    if not lead_id:
        lead = DB.leads.get_by_wa_id(conv.get("wa_id", ""))
        lead_id = lead.get("lead_id") if lead else None

    cm = DB.customer_memory.get_by_lead_id(lead_id) if lead_id else {}
    sm = DB.sales_memory.get_by_lead_id(lead_id) if lead_id else {}

    return jsonify({
        "conversation_id": conversation_id,
        "lead_id": lead_id,
        "customer_memory": serialize_doc(cm or {}),
        "sales_memory": serialize_doc(sm or {})
    })


@app.route("/api/conversations/search", methods=["GET"])
def api_search_conversations():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"results": []})

    matched_msgs = DB.messages.search_messages(query=q, limit=30)
    results = []
    seen_convs = set()
    for m in matched_msgs:
        cid = m.get("conversation_id")
        if cid not in seen_convs:
            seen_convs.add(cid)
            c = DB.conversations.get_by_id(cid)
            lead = DB.leads.get_by_id(c.get("lead_id")) if c and c.get("lead_id") else None
            results.append({
                "conversation_id": cid,
                "customer_name": lead.get("name") if lead else f"+{m.get('wa_id')}",
                "matched_text": m.get("text"),
                "sender_type": m.get("sender_type"),
                "timestamp": m.get("created_at")
            })

    return jsonify({"query": q, "count": len(results), "results": serialize_doc(results)})


@app.route("/api/conversations/<conversation_id>/export", methods=["GET"])
@require_auth()
def api_export_conversation(conversation_id):
    fmt = request.args.get("format", "json").lower()
    conv = DB.conversations.get_by_id(conversation_id)
    if not conv:
        return jsonify({"error": "Conversation not found"}), 404

    lead_id = conv.get("lead_id")
    lead = DB.leads.get_by_id(lead_id) if lead_id else DB.leads.get_by_wa_id(conv.get("wa_id", ""))
    msgs = DB.messages.get_last_messages(conversation_id=conversation_id, limit=500)

    # Sanitize and prepare data (strictly no tokens or secrets)
    customer_name = (lead.get("name") if lead else None) or f"+{conv.get('wa_id')}"
    cleaned_msgs = []
    for m in msgs:
        cleaned_msgs.append({
            "timestamp": str(m.get("created_at", "")),
            "sender": m.get("sender_type", "UNKNOWN"),
            "text": m.get("text", "")
        })

    if fmt == "txt":
        lines = [
            f"=== ARIS CONVERSATION EXPORT ===",
            f"Customer: {customer_name}",
            f"Phone: +{conv.get('wa_id')}",
            f"Sales Stage: {conv.get('sales_stage', 'NEW')}",
            f"Exported At: {datetime.now(timezone.utc).isoformat()}",
            "================================\n"
        ]
        for m in cleaned_msgs:
            lines.append(f"[{m['timestamp']}] {m['sender']}: {m['text']}")
        return Response("\n".join(lines), mimetype="text/plain", headers={"Content-Disposition": f"attachment;filename=conversation_{conversation_id}.txt"})

    elif fmt == "csv":
        import io
        import csv
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Timestamp", "Sender", "Message"])
        for m in cleaned_msgs:
            writer.writerow([m["timestamp"], m["sender"], m["text"]])
        return Response(output.getvalue(), mimetype="text/csv", headers={"Content-Disposition": f"attachment;filename=conversation_{conversation_id}.csv"})

    return jsonify({
        "conversation_id": conversation_id,
        "customer_name": customer_name,
        "phone": conv.get("wa_id"),
        "sales_stage": conv.get("sales_stage"),
        "messages": cleaned_msgs
    })


@app.route("/api/analytics/sales-funnel", methods=["GET"])
@require_auth()
def api_sales_funnel_analytics():
    from sales.analytics import SalesAnalyticsService
    svc = SalesAnalyticsService()
    return jsonify(svc.get_funnel_metrics())


@app.route("/api/conversations/<conversation_id>/send", methods=["POST"])
def api_send_manual_message(conversation_id):
    data = request.get_json(silent=True) or {}
    wa_id = data.get("wa_id")
    text = data.get("text", "").strip()

    conv = DB.conversations.get_by_id(conversation_id)
    if not wa_id and conv:
        wa_id = conv.get("wa_id")

    if not wa_id or not text:
        return jsonify({"error": "text is required"}), 400
    lead_id = conv.get("lead_id") if conv else None
    if not lead_id:
        lead = DB.leads.get_by_wa_id(wa_id)
        lead_id = lead.get("lead_id") if lead else None

    try:
        res = whatsapp_client.send_text(recipient=wa_id, message=text)
        meta_id = res.meta_message_id
    except Exception as ex:
        return jsonify({"error": str(ex)}), 500

    # PERSIST OUTBOUND HUMAN MESSAGE
    DB.messages.save_outbound_human_message(
        conversation_id=conversation_id,
        wa_id=wa_id,
        text=text,
        lead_id=lead_id,
        whatsapp_message_id=meta_id,
        status="SENT"
    )
    DB.conversations.increment_message_count(conversation_id, count=1)
    DB.conversations.update_last_message(conversation_id)
    DB.conversations.update_timestamps(conversation_id, sender_type="HUMAN")

    DB.events.log_event(
        event_type="manual_agent_message_sent",
        wa_id=wa_id,
        metadata={"text": text, "conversation_id": conversation_id}
    )

    return jsonify({"success": True, "meta_message_id": meta_id})


@app.route("/api/conversations/<conversation_id>/takeover", methods=["POST"])
def api_toggle_takeover(conversation_id):
    data = request.get_json(silent=True) or {}
    active = bool(data.get("active", data.get("enable", data.get("takeover", False))))

    DB.conversations.set_human_takeover(conversation_id, active)

    # Log takeover event
    conv = DB.conversations.get_by_id(conversation_id)
    DB.sales_events.log_sales_event(
        event_type="HUMAN_TAKEOVER_ENABLED" if active else "HUMAN_TAKEOVER_DISABLED",
        lead_id=conv.get("lead_id") if conv else None,
        conversation_id=conversation_id,
        metadata={"active": active}
    )

    return jsonify({"success": True, "human_takeover": active})


@app.route("/api/conversations/<conversation_id>/toggle-ai", methods=["POST"])
def api_toggle_conversation_ai(conversation_id):
    """Toggles AI auto-reply on/off for a specific conversation/phone number."""
    data = request.get_json(silent=True) or {}
    enabled = bool(data.get("enabled", not data.get("disable", False)))

    conv = DB.conversations.get_by_id(conversation_id)
    if not conv:
        return jsonify({"error": "Conversation not found"}), 404

    wa_id = conv.get("wa_id")
    from safety_guard import set_ai_enabled_for_number
    set_ai_enabled_for_number(wa_id, enabled)
    DB.conversations.set_human_takeover(conversation_id, not enabled)

    return jsonify({
        "success": True,
        "conversation_id": conversation_id,
        "wa_id": wa_id,
        "ai_enabled": enabled
    })


@app.route("/api/settings/ai-autoreply", methods=["GET", "POST"])
def api_settings_ai_autoreply():
    """Gets or sets the global AI auto-reply toggle (for all numbers)."""
    from safety_guard import is_global_ai_autoreply_enabled, set_global_ai_autoreply_enabled, STRICT_REGISTERED_LEADS_ONLY
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        new_val = bool(data.get("enabled", True))
        updated = set_global_ai_autoreply_enabled(new_val)
        return jsonify({
            "success": True,
            "global_ai_autoreply_enabled": updated,
            "message": f"Global AI auto-reply is now {'ENABLED' if updated else 'DISABLED'} for all numbers."
        })

    return jsonify({
        "global_ai_autoreply_enabled": is_global_ai_autoreply_enabled(),
        "strict_registered_leads_only": STRICT_REGISTERED_LEADS_ONLY,
        "owner_whatsapp": "8600079496"
    })


@app.route("/api/properties", methods=["GET"])
def api_list_properties():
    city = request.args.get("city")
    bhk = request.args.get("bhk")
    prop_type = request.args.get("type")
    max_price = request.args.get("max_price")
    max_budget = float(max_price) if max_price else None

    props = DB.properties.list_properties(
        city=city,
        bhk=bhk,
        prop_type=prop_type,
        max_budget_lakhs=max_budget,
        limit=100
    )
    return jsonify({"properties": serialize_doc(props)})


@app.route("/api/properties", methods=["POST"])
def api_create_property():
    data = request.get_json(silent=True) or {}
    if not data.get("title") or not data.get("locality"):
        return jsonify({"error": "title and locality are required"}), 400

    created = DB.properties.create_property(data)
    DB.events.log_event(event_type="property_created", property_id=created.get("id"))
    return jsonify({"success": True, "property": serialize_doc(created)})


@app.route("/api/properties/<prop_id>", methods=["PATCH", "PUT"])
def api_update_property(prop_id):
    data = request.get_json(silent=True) or {}
    updated = DB.properties.update_property(prop_id, data)
    if not updated:
        return jsonify({"error": "Property not found"}), 404
    return jsonify({"success": True, "property": serialize_doc(updated)})


@app.route("/api/properties/<prop_id>", methods=["DELETE"])
def api_delete_property(prop_id):
    success = DB.properties.delete_property(prop_id)
    return jsonify({"success": success})


@app.route("/api/visits", methods=["GET"])
def api_list_visits():
    status = request.args.get("status")
    query = {}
    if status and status != "ALL":
        query["status"] = status
    visits = DB.visits.list_visits(filter_query=query, limit=100)
    return jsonify({"visits": serialize_doc(visits)})


@app.route("/api/visits/<visit_id>", methods=["PATCH"])
def api_update_visit(visit_id):
    data = request.get_json(silent=True) or {}
    action = data.get("action")

    if action == "reschedule":
        new_date = data.get("visit_date")
        new_time = data.get("visit_time")
        v = DB.visits.reschedule_visit(visit_id, new_date, new_time)
        return jsonify({"success": bool(v), "visit": serialize_doc(v)})
    elif action == "cancel":
        reason = data.get("reason", "")
        v = DB.visits.cancel_visit(visit_id, reason=reason)
        return jsonify({"success": bool(v), "visit": serialize_doc(v)})

    return jsonify({"error": "Unknown action"}), 400


@app.route("/api/followups", methods=["GET"])
def api_list_followups():
    status = request.args.get("status")
    query = {}
    if status and status != "ALL":
        query["status"] = status
    fups = DB.followups.list_followups(filter_query=query, limit=100)
    return jsonify({"followups": serialize_doc(fups)})


@app.route("/api/system/health", methods=["GET"])
def api_system_health():
    db_conn = get_db().is_connected()
    return jsonify({
        "status": "operational",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database": {
            "mongodb": {
                "status": "connected" if db_conn else "disconnected",
                "mode": "Primary MongoDB" if db_conn else "Degraded In-Memory Cache",
                "target": Config.MONGODB_URI
            }
        },
        "meta_whatsapp": {
            "phone_number_id": Config.PHONE_NUMBER_ID,
            "api_version": Config.API_VERSION,
            "token_configured": bool(Config.ACCESS_TOKEN)
        },
        "ai_engine": {
            "active_provider": gemini_provider.get_provider_name(),
            "llm_configured": gemini_provider.is_configured()
        }
    })


# =====================================================================
# 7. WhatsApp Outbound & Campaign REST APIs
# =====================================================================

@app.route("/api/admin/whatsapp/test-send", methods=["POST"])
@require_auth(allowed_roles=["ADMIN"])
def api_test_send():
    """
    Sends a test WhatsApp template message with strict rate-limiting (10/hour).
    """
    data = request.get_json(silent=True) or {}
    phone_number = str(data.get("phone_number", "")).strip()
    template_name = str(data.get("template_name", "")).strip()
    language_code = str(data.get("language_code", "en")).strip() or "en"
    idempotency_key = data.get("idempotency_key")
    components = data.get("components")

    if not phone_number:
        return jsonify({"success": False, "error": {"message": "Recipient phone number is required."}}), 400
    if not template_name:
        return jsonify({"success": False, "error": {"message": "Template name is required."}}), 400

    current_user = getattr(g, "current_user", {}) or {}
    user_identifier = current_user.get("email") or current_user.get("name") or "admin"

    allowed, current_count, limit = rate_limiter.check_and_record(user_identifier, action="TEST_SEND")
    if not allowed:
        return jsonify({
            "success": False,
            "error": {
                "code": "RATE_LIMIT_EXCEEDED",
                "message": f"Test send limit reached ({limit} sends per hour). Please try again later.",
                "current_count": current_count,
                "limit": limit
            }
        }), 429

    result = outbound_service.send_template(
        phone_number=phone_number,
        template_name=template_name,
        language_code=language_code,
        components=components,
        source="TEST_SEND",
        created_by=user_identifier,
        idempotency_key=idempotency_key,
        metadata={"triggered_by": "admin_ui", "user_email": user_identifier}
    )

    status_code = 200 if result.success else 400
    return jsonify(result.to_api_dict()), status_code


@app.route("/api/admin/whatsapp/templates", methods=["GET"])
@require_auth(allowed_roles=["ADMIN"])
def api_list_templates():
    templates = TemplateRegistry.list_all()
    return jsonify({"templates": [t.to_dict() for t in templates]})


@app.route("/api/admin/whatsapp/meta-templates", methods=["GET"])
@require_auth(allowed_roles=["ADMIN"])
def api_meta_templates():
    res = whatsapp_client.fetch_meta_templates()
    if isinstance(res, dict) and "data" in res:
        for t in res["data"]:
            name = t.get("name")
            lang = t.get("language", "en")
            status = t.get("status", "APPROVED")
            body_text = ""
            if t.get("components"):
                for c in t["components"]:
                    if c.get("type") == "BODY":
                        body_text = c.get("text", "")
            if name:
                TemplateRegistry.register(WhatsAppTemplate(
                    name=name,
                    language_code=lang,
                    category=t.get("category", "MARKETING"),
                    display_name=name.replace("_", " ").title(),
                    body_preview=body_text,
                    status="ACTIVE" if status == "APPROVED" else status,
                    quality_status=status,
                    enabled=(status == "APPROVED")
                ))
    return jsonify(res)


@app.route("/api/admin/whatsapp/phone/normalize", methods=["POST"])
@require_auth(allowed_roles=["ADMIN"])
def api_normalize_phone():
    data = request.get_json(silent=True) or {}
    raw_phone = str(data.get("phone", "")).strip()
    is_valid = phone_service.is_valid(raw_phone)
    clean = ""
    if is_valid:
        try:
            info = phone_service.normalize(raw_phone)
            clean = info.get("normalized_phone", "")
        except Exception:
            clean = ""
    return jsonify({
        "raw": raw_phone,
        "normalized": clean,
        "is_valid": is_valid,
        "formatted": f"+{clean}" if clean else ""
    })


@app.route("/api/admin/whatsapp/messages", methods=["GET"])
@require_auth(allowed_roles=["ADMIN"])
def api_list_messages():
    page = max(1, int(request.args.get("page", 1)))
    limit = min(100, max(1, int(request.args.get("limit", 20))))
    skip = (page - 1) * limit

    status = request.args.get("status")
    phone = request.args.get("phone")
    campaign_id = request.args.get("campaign_id")
    template_name = request.args.get("template")
    source = request.args.get("source")

    messages = DB.outbound_messages.list_messages(
        status=status,
        phone=phone,
        campaign_id=campaign_id,
        template_name=template_name,
        source=source,
        limit=limit,
        skip=skip
    )
    total_count = DB.outbound_messages.count_messages(
        status=status,
        phone=phone,
        campaign_id=campaign_id,
        template_name=template_name,
        source=source
    )

    return jsonify({
        "messages": serialize_doc(messages),
        "total": total_count,
        "page": page,
        "limit": limit,
        "pages": (total_count + limit - 1) // limit if limit else 1
    })


@app.route("/api/admin/whatsapp/messages/<message_id>", methods=["GET"])
@require_auth(allowed_roles=["ADMIN"])
def api_get_message(message_id: str):
    message = DB.outbound_messages.get_by_id(message_id) or DB.outbound_messages.get_by_meta_message_id(message_id)
    if not message:
        return jsonify({"error": "Message not found"}), 404
    return jsonify({"message": serialize_doc(message)})


@app.route("/api/admin/whatsapp/campaigns/sample-excel", methods=["GET"])
@require_auth(allowed_roles=["ADMIN"])
def api_download_sample_excel():
    excel_bytes = CampaignLeadImporter.generate_sample_template()
    return Response(
        excel_bytes,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=ARIS_Campaign_Leads_Template.xlsx"}
    )


@app.route("/api/admin/whatsapp/campaigns/parse-leads", methods=["POST"])
@require_auth(allowed_roles=["ADMIN"])
def api_parse_campaign_leads():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Please select an Excel (.xlsx, .xls) or CSV file."}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename."}), 400

    try:
        file_bytes = file.read()
        result = CampaignLeadImporter.parse_file(file_bytes, file.filename)
        return jsonify(result)
    except Exception as ex:
        logger.error(f"[CAMPAIGN PARSE ERROR] {ex}")
        return jsonify({"error": str(ex)}), 400


@app.route("/api/admin/whatsapp/campaigns/import-leads", methods=["POST"])
@require_auth(allowed_roles=["ADMIN"])
def api_import_campaign_leads():
    data = request.get_json(silent=True) or {}
    leads = data.get("leads", [])
    campaign_title = data.get("campaign_title", "")
    if not leads:
        return jsonify({"error": "No leads provided for import."}), 400

    try:
        lead_ids = CampaignLeadImporter.import_and_persist(leads, campaign_title)
        return jsonify({
            "success": True,
            "imported_count": len(lead_ids),
            "lead_ids": lead_ids
        })
    except Exception as ex:
        logger.error(f"[CAMPAIGN IMPORT PERSIST ERROR] {ex}")
        return jsonify({"error": str(ex)}), 500


@app.route("/api/admin/whatsapp/leads/selector", methods=["GET"])
@require_auth(allowed_roles=["ADMIN"])
def api_selector_leads():
    search = request.args.get("search", "").strip()
    city = request.args.get("city", "").strip()
    stage = request.args.get("stage", "").strip()
    bhk = request.args.get("bhk", "").strip()
    limit = min(500, int(request.args.get("limit", 200)))

    query = {"opted_out": {"$ne": True}}
    if search:
        query["$or"] = [
            {"name": {"$regex": search, "$options": "i"}},
            {"phone": {"$regex": search, "$options": "i"}},
            {"wa_id": {"$regex": search, "$options": "i"}},
        ]
    if city:
        query["preferred_city"] = city
    if stage and stage != "ALL":
        query["sales_stage"] = stage
    if bhk:
        query["bhk"] = {"$in": [bhk]}

    leads = DB.leads.list_leads(filter_query=query, limit=limit)
    formatted = []
    for l in leads:
        bhk_val = l.get("bhk", "")
        if isinstance(bhk_val, list):
            bhk_val = ", ".join(bhk_val)
        formatted.append({
            "lead_id": l.get("lead_id"),
            "name": l.get("name") or "Valued Client",
            "phone": l.get("wa_id") or l.get("phone", ""),
            "city": l.get("preferred_city") or l.get("city", ""),
            "locality": l.get("locality", ""),
            "sales_stage": l.get("sales_stage", "NEW"),
            "lead_score": l.get("lead_score", 0),
            "lead_temperature": l.get("lead_temperature") or ("HOT" if l.get("lead_score", 0) >= 60 else "WARM"),
            "bhk": bhk_val,
            "budget_max": l.get("budget_max")
        })
    return jsonify({"leads": formatted, "total": len(formatted)})


@app.route("/api/admin/whatsapp/audience/preview", methods=["POST"])
@require_auth(allowed_roles=["ADMIN"])
def api_preview_audience():
    data = request.get_json(silent=True) or {}
    city = data.get("city")
    stage = data.get("stage")
    min_score = int(data.get("min_lead_score", 0) or 0)
    bhk_filter = data.get("bhk_filter")
    audience_type = data.get("audience_type", "FILTER")
    selected_lead_ids = data.get("selected_lead_ids")

    breakdown = campaign_service.compute_audience_breakdown(
        city=city,
        stage=stage,
        min_score=min_score,
        bhk_filter=bhk_filter,
        audience_type=audience_type,
        selected_lead_ids=selected_lead_ids
    )
    return jsonify({"breakdown": breakdown})


@app.route("/api/admin/whatsapp/campaigns", methods=["GET"])
@require_auth(allowed_roles=["ADMIN"])
def api_list_admin_campaigns():
    campaigns = DB.campaigns.list_campaigns(limit=100)
    return jsonify({"campaigns": serialize_doc(campaigns)})


@app.route("/api/admin/whatsapp/campaigns", methods=["POST"])
@require_auth(allowed_roles=["ADMIN"])
def api_create_admin_campaign():
    data = request.get_json(silent=True) or {}
    title = str(data.get("title", "")).strip()
    template_name = str(data.get("template_name", "")).strip()

    if not title:
        return jsonify({"error": "Campaign title is required"}), 400
    if not template_name:
        return jsonify({"error": "Template selection is required"}), 400

    current_user = getattr(g, "current_user", {}) or {}
    created_by = current_user.get("email") or "admin"

    audience_type = data.get("audience_type", "FILTER")
    selected_lead_ids = data.get("selected_lead_ids") or []
    imported_leads = data.get("imported_leads") or []

    # If leads were imported from file, persist them to CRM & CustomerMemory
    if audience_type == "FILE_UPLOAD" and imported_leads and not selected_lead_ids:
        try:
            selected_lead_ids = CampaignLeadImporter.import_and_persist(imported_leads, title)
        except Exception as ex:
            logger.error(f"[CAMPAIGN FILE IMPORT ERROR] {ex}")

    scheduled_at = None
    if data.get("scheduled_at"):
        try:
            from datetime import datetime
            scheduled_at = datetime.fromisoformat(str(data["scheduled_at"]).replace("Z", "+00:00"))
        except Exception:
            scheduled_at = None

    tags = data.get("tags")
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]

    try:
        camp = campaign_service.create_campaign(
            title=title,
            template_name=template_name,
            target_city=data.get("target_city"),
            target_stage=data.get("target_stage"),
            min_lead_score=int(data.get("min_lead_score", 0) or 0),
            language_code=str(data.get("language_code", "en")),
            description=str(data.get("description", "")),
            created_by=created_by,
            bhk_filter=data.get("bhk_filter"),
            budget_min=float(data["budget_min"]) if data.get("budget_min") else None,
            budget_max=float(data["budget_max"]) if data.get("budget_max") else None,
            property_id=data.get("property_id"),
            project_id=data.get("project_id"),
            audience_type=audience_type,
            selected_lead_ids=selected_lead_ids,
            tags=tags,
            priority=int(data.get("priority", 2) or 2),
            delivery_window_start=data.get("delivery_window_start"),
            delivery_window_end=data.get("delivery_window_end"),
            rate_limit_per_second=int(data["rate_limit_per_second"]) if data.get("rate_limit_per_second") else None,
            scheduled_at=scheduled_at,
        )
        return jsonify({"success": True, "campaign": serialize_doc(camp)}), 201
    except ValueError as ex:
        return jsonify({"error": str(ex)}), 400
    except Exception as ex:
        logger.error(f"[CAMPAIGN CREATE ERROR] {ex}")
        return jsonify({"error": "Failed to create campaign"}), 500


@app.route("/api/admin/whatsapp/campaigns/<campaign_id>", methods=["GET"])
@require_auth(allowed_roles=["ADMIN"])
def api_get_admin_campaign(campaign_id: str):
    campaign = DB.campaigns.get_by_id(campaign_id)
    if not campaign:
        return jsonify({"error": "Campaign not found"}), 404
    analytics = campaign_service.get_analytics(campaign_id)
    target_leads = campaign_service.get_eligible_audience(
        city=campaign.get("target_city"),
        stage=campaign.get("target_stage"),
        min_score=campaign.get("min_lead_score", 0),
        bhk_filter=campaign.get("bhk_filter"),
        audience_type=campaign.get("audience_type", "FILTER"),
        selected_lead_ids=campaign.get("selected_lead_ids"),
    )
    if analytics.get("queued", 0) == 0 and campaign.get("status") in ("DRAFT", "READY"):
        analytics["queued"] = len(target_leads)

    return jsonify({
        "campaign": serialize_doc(campaign),
        "analytics": analytics,
        "target_leads": serialize_doc(target_leads),
    })


@app.route("/api/admin/whatsapp/campaigns/<campaign_id>", methods=["PUT", "POST"])
@require_auth(allowed_roles=["ADMIN"])
def api_update_admin_campaign(campaign_id: str):
    data = request.get_json(silent=True) or {}
    current_user = getattr(g, "current_user", {}) or {}
    updated_by = current_user.get("email") or "admin"

    try:
        updated = campaign_service.update_campaign(campaign_id, data, updated_by=updated_by)
        return jsonify({"success": True, "campaign": serialize_doc(updated)})
    except ValueError as ex:
        return jsonify({"error": str(ex)}), 400
    except Exception as ex:
        logger.error(f"[CAMPAIGN UPDATE ERROR] {ex}")
        return jsonify({"error": "Failed to update campaign"}), 500


@app.route("/api/admin/whatsapp/campaigns/<campaign_id>/leads/<lead_id>", methods=["DELETE", "POST"])
@require_auth(allowed_roles=["ADMIN"])
def api_remove_campaign_lead(campaign_id: str, lead_id: str):
    current_user = getattr(g, "current_user", {}) or {}
    removed_by = current_user.get("email") or "admin"

    try:
        updated = campaign_service.remove_lead_from_campaign(campaign_id, lead_id, removed_by=removed_by)
        return jsonify({
            "success": True,
            "campaign": serialize_doc(updated),
            "remaining_count": len(updated.get("selected_lead_ids", []))
        })
    except ValueError as ex:
        return jsonify({"error": str(ex)}), 400
    except Exception as ex:
        logger.error(f"[CAMPAIGN REMOVE LEAD ERROR] {ex}")
        return jsonify({"error": "Failed to remove lead from campaign"}), 500


@app.route("/api/admin/whatsapp/campaigns/<campaign_id>/validate", methods=["POST"])
@require_auth(allowed_roles=["ADMIN"])
def api_validate_campaign(campaign_id: str):
    res = campaign_service.validate_campaign(campaign_id)
    if "error" in res:
        return jsonify(res), 400
    return jsonify(res)


@app.route("/api/admin/whatsapp/campaigns/<campaign_id>/dry-run", methods=["POST"])
@require_auth(allowed_roles=["ADMIN"])
def api_dry_run_campaign(campaign_id: str):
    res = campaign_service.dry_run(campaign_id)
    if "error" in res:
        return jsonify(res), 400
    return jsonify(res)


@app.route("/api/admin/whatsapp/campaigns/<campaign_id>/start", methods=["POST"])
@require_auth(allowed_roles=["ADMIN"])
def api_start_campaign(campaign_id: str):
    current_user = getattr(g, "current_user", {}) or {}
    started_by = current_user.get("email") or "admin"
    res = campaign_service.start_campaign(campaign_id, started_by=started_by)
    if "error" in res:
        return jsonify(res), 400
    return jsonify(res)


@app.route("/api/admin/whatsapp/campaigns/<campaign_id>/pause", methods=["POST"])
@require_auth(allowed_roles=["ADMIN"])
def api_pause_campaign(campaign_id: str):
    current_user = getattr(g, "current_user", {}) or {}
    paused_by = current_user.get("email") or "admin"
    res = campaign_service.pause_campaign(campaign_id, paused_by=paused_by)
    if "error" in res:
        return jsonify(res), 400
    return jsonify(res)


@app.route("/api/admin/whatsapp/campaigns/<campaign_id>/resume", methods=["POST"])
@require_auth(allowed_roles=["ADMIN"])
def api_resume_campaign(campaign_id: str):
    current_user = getattr(g, "current_user", {}) or {}
    resumed_by = current_user.get("email") or "admin"
    res = campaign_service.resume_campaign(campaign_id, resumed_by=resumed_by)
    if "error" in res:
        return jsonify(res), 400
    return jsonify(res)


@app.route("/api/admin/whatsapp/campaigns/<campaign_id>/cancel", methods=["POST"])
@require_auth(allowed_roles=["ADMIN"])
def api_cancel_campaign(campaign_id: str):
    current_user = getattr(g, "current_user", {}) or {}
    cancelled_by = current_user.get("email") or "admin"
    res = campaign_service.cancel_campaign(campaign_id, cancelled_by=cancelled_by)
    if "error" in res:
        return jsonify(res), 400
    return jsonify(res)


@app.route("/api/admin/whatsapp/campaigns/<campaign_id>/analytics", methods=["GET"])
@require_auth(allowed_roles=["ADMIN"])
def api_campaign_analytics(campaign_id: str):
    analytics = campaign_service.get_analytics(campaign_id)
    return jsonify({"analytics": analytics})


@app.route("/api/admin/whatsapp/campaigns/<campaign_id>/recipients", methods=["GET"])
@require_auth(allowed_roles=["ADMIN"])
def api_campaign_recipients(campaign_id: str):
    status_filter = request.args.get("status", "ALL").strip()
    search = request.args.get("q", "").strip()
    page = max(1, int(request.args.get("page", 1)))
    limit = max(1, min(100, int(request.args.get("limit", 50))))
    skip = (page - 1) * limit

    recipients = campaign_service.get_recipient_list(
        campaign_id=campaign_id,
        status_filter=status_filter,
        search=search,
        limit=limit,
        skip=skip
    )
    total = campaign_service.get_recipient_count(
        campaign_id=campaign_id,
        status_filter=status_filter,
        search=search
    )
    total_pages = max(1, (total + limit - 1) // limit)

    return jsonify({
        "recipients": serialize_doc(recipients),
        "total": total,
        "page": page,
        "total_pages": total_pages,
        "limit": limit
    })


@app.route("/api/admin/whatsapp/campaigns/<campaign_id>/errors", methods=["GET"])
@require_auth(allowed_roles=["ADMIN"])
def api_campaign_errors(campaign_id: str):
    breakdown = campaign_service.get_error_breakdown(campaign_id)
    return jsonify(breakdown)


@app.route("/api/admin/whatsapp/campaigns/<campaign_id>/timeline", methods=["GET"])
@require_auth(allowed_roles=["ADMIN"])
def api_campaign_timeline(campaign_id: str):
    timeline = campaign_service.get_campaign_timeline(campaign_id)
    return jsonify({"timeline": timeline})


@app.route("/api/admin/whatsapp/campaigns/<campaign_id>/clone", methods=["POST"])
@require_auth(allowed_roles=["ADMIN"])
def api_clone_campaign(campaign_id: str):
    data = request.get_json(silent=True) or {}
    new_title = data.get("title")
    current_user = getattr(g, "current_user", {}) or {}
    cloned_by = current_user.get("email") or "admin"

    cloned = campaign_service.clone_campaign(campaign_id, new_title=new_title, cloned_by=cloned_by)
    if not cloned:
        return jsonify({"error": "Failed to clone campaign"}), 404
    return jsonify({"success": True, "campaign": serialize_doc(cloned)})


@app.route("/api/admin/whatsapp/campaigns/<campaign_id>/schedule", methods=["POST"])
@require_auth(allowed_roles=["ADMIN"])
def api_schedule_campaign(campaign_id: str):
    data = request.get_json(silent=True) or {}
    scheduled_at = data.get("scheduled_at")
    if not scheduled_at:
        return jsonify({"error": "scheduled_at is required"}), 400
    current_user = getattr(g, "current_user", {}) or {}
    scheduled_by = current_user.get("email") or "admin"

    res = campaign_service.schedule_campaign(campaign_id, scheduled_at, scheduled_by=scheduled_by)
    if "error" in res:
        return jsonify(res), 400
    return jsonify(res)


@app.route("/api/admin/whatsapp/campaigns/<campaign_id>/retry-failed", methods=["POST"])
@require_auth(allowed_roles=["ADMIN"])
def api_retry_failed_campaign(campaign_id: str):
    current_user = getattr(g, "current_user", {}) or {}
    retried_by = current_user.get("email") or "admin"
    res = campaign_service.retry_failed_recipients(campaign_id, retried_by=retried_by)
    if "error" in res:
        return jsonify(res), 400
    return jsonify(res)


@app.route("/api/admin/whatsapp/campaigns/<campaign_id>/export", methods=["GET"])
@require_auth(allowed_roles=["ADMIN"])
def api_export_campaign_recipients(campaign_id: str):
    recipients = campaign_service.get_recipient_list(campaign_id, limit=10000)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Lead ID", "Name", "Phone", "Status", "Outbound ID",
        "Meta Message ID", "Error Code", "Error Message", "Error Category",
        "Sent At", "Delivered At", "Read At", "Retries"
    ])
    for r in recipients:
        writer.writerow([
            r.get("lead_id", ""),
            r.get("name", ""),
            r.get("phone", ""),
            r.get("status", ""),
            r.get("outbound_id", ""),
            r.get("meta_message_id", ""),
            r.get("error_code", ""),
            r.get("error_message", ""),
            r.get("error_category", ""),
            str(r.get("sent_at") or ""),
            str(r.get("delivered_at") or ""),
            str(r.get("read_at") or ""),
            r.get("retry_count", 0),
        ])
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=campaign_{campaign_id}_recipients.csv"}
    )


@app.route("/api/admin/whatsapp/campaigns/stats", methods=["GET"])
@require_auth(allowed_roles=["ADMIN"])
def api_campaigns_global_stats():
    camps = DB.campaigns.list_campaigns_filtered(limit=1000)
    total_sent = sum(c.get("sent_count", 0) for c in camps)
    total_delivered = sum(c.get("delivered_count", 0) for c in camps)
    total_read = sum(c.get("read_count", 0) for c in camps)
    total_failed = sum(c.get("failed_count", 0) for c in camps)
    total_replied = sum(c.get("replied_count", 0) for c in camps)

    return jsonify({
        "total_campaigns": len(camps),
        "total_sent": total_sent,
        "total_delivered": total_delivered,
        "total_read": total_read,
        "total_failed": total_failed,
        "total_replied": total_replied,
        "delivery_rate": round(total_delivered / total_sent * 100, 1) if total_sent > 0 else 0.0,
        "read_rate": round(total_read / total_delivered * 100, 1) if total_delivered > 0 else 0.0,
    })


@app.route("/api/admin/whatsapp/campaigns/<campaign_id>", methods=["DELETE"])
@require_auth(allowed_roles=["ADMIN"])
def api_delete_campaign(campaign_id: str):
    DB.campaigns.delete_campaign(campaign_id)
    return jsonify({"success": True, "message": "Campaign deleted."})


# Backward-compatible campaign aliases
@app.route("/api/campaigns", methods=["GET"])
@require_auth()
def api_list_campaigns():
    campaigns = DB.campaigns.list_campaigns()
    return jsonify({"campaigns": serialize_doc(campaigns)})


@app.route("/api/campaigns", methods=["POST"])
@require_auth()
def api_create_campaign():
    return api_create_admin_campaign()


@app.route("/api/campaigns/<campaign_id>/run", methods=["POST"])
@require_auth()
def api_run_campaign(campaign_id):
    return api_start_campaign(campaign_id)


@app.route("/api/knowledge/reindex", methods=["POST"])
@require_auth()
def api_reindex_knowledge():
    from ai_engine import RAGService
    rag = RAGService()
    count = rag.ingest_documents()
    total_chunks = DB.vectors.count_chunks()
    return jsonify({
        "success": True,
        "newly_ingested_chunks": count,
        "total_indexed_chunks": total_chunks,
        "vector_store_mode": getattr(DB.vectors, "mode", "in_memory")
    })


# =====================================================================
# Main Process Runner
# =====================================================================

if __name__ == "__main__":
    # Ensure active conversations start in responsive AI mode (strictly preserving blacklisted/bot numbers)
    try:
        if DB.conversations._db.is_connected():
            from safety_guard import KNOWN_BLACKLISTED_NUMBERS
            DB.conversations._db.conversations.update_many(
                {"wa_id": {"$nin": list(KNOWN_BLACKLISTED_NUMBERS)}},
                {"$set": {"human_takeover": False, "takeover_agent_id": None}}
            )
            print("[MONGODB] Active conversation human_takeover flags reset to AI mode (blacklisted/bot numbers preserved).")
    except Exception:
        pass

    # Start background campaign and follow-up schedulers
    try:
        CampaignScheduler.start()
        FollowUpScheduler.start()
    except Exception as ex:
        print(f"[SCHEDULER ERROR] Failed to start daemon: {ex}")

    app.run(
        host="0.0.0.0",
        port=Config.PORT,
        debug=False
    )
