"""
Database, Data Models & Persistence Layer for ARIS.
Consolidates MongoDB connection manager, data models, and repositories into a single unified module.
Supports high-concurrency WhatsApp webhook persistence with in-memory offline fallback.
"""

import logging
import uuid
import time
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from bson import ObjectId
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.database import Database
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

from config import Config

logger = logging.getLogger(__name__)


# =============================================================================
# JSON SERIALIZATION HELPER
# =============================================================================

def serialize_doc(doc: Any) -> Any:
    """Recursively converts MongoDB ObjectIds, datetimes, and custom objects for JSON serialization."""
    if isinstance(doc, list):
        return [serialize_doc(item) for item in doc]
    if isinstance(doc, dict):
        out = {}
        for k, v in doc.items():
            if isinstance(v, ObjectId):
                out[k] = str(v)
            elif isinstance(v, datetime):
                out[k] = v.isoformat()
            elif isinstance(v, (dict, list)):
                out[k] = serialize_doc(v)
            else:
                out[k] = v
        return out
    if isinstance(doc, ObjectId):
        return str(doc)
    if isinstance(doc, datetime):
        return doc.isoformat()
    return doc


# =============================================================================
# DATA MODELS & SCHEMAS
# =============================================================================

class SalesStage:
    NEW = "NEW"
    CONTACTED = "CONTACTED"
    QUALIFIED = "QUALIFIED"
    VISIT_SCHEDULED = "VISIT_SCHEDULED"
    VISIT_COMPLETED = "VISIT_COMPLETED"
    NEGOTIATION = "NEGOTIATION"
    CLOSED_WON = "CLOSED_WON"
    CLOSED_LOST = "CLOSED_LOST"


class CampaignStatus:
    DRAFT = "DRAFT"
    VALIDATING = "VALIDATING"
    READY = "READY"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class OutboundStatus:
    QUEUED = "QUEUED"
    SENDING = "SENDING"
    ACCEPTED = "ACCEPTED"
    SENT = "SENT"
    DELIVERED = "DELIVERED"
    READ = "READ"
    FAILED = "FAILED"


class RecipientStatus:
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    SENT = "SENT"
    DELIVERED = "DELIVERED"
    READ = "READ"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class OutboundSource:
    TEST_SEND = "TEST_SEND"
    CAMPAIGN = "CAMPAIGN"
    AGENT = "AGENT"
    TRANSACTIONAL = "TRANSACTIONAL"


@dataclass
class User:
    wa_id: str
    profile_name: str = ""
    phone_number: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "active"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "User":
        data = data.copy()
        data.pop("_id", None)
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class Lead:
    lead_id: str
    wa_id: str
    name: str = "Anonymous"
    phone: str = ""
    preferred_city: str = ""
    preferred_location: str = ""
    property_type: str = ""
    bhk: str = ""
    budget_min: float = 0.0
    budget_max: float = 0.0
    timeline: str = ""
    investment_intent: str = "END_USE"
    lead_score: int = 0
    score_breakdown: Dict[str, Any] = field(default_factory=dict)
    sales_stage: str = SalesStage.NEW
    lead_temperature: str = "WARM"
    conversation_id: Optional[str] = None
    assigned_agent_id: Optional[str] = None
    whatsapp_opt_in: bool = True
    marketing_opt_in: bool = True
    opted_out: bool = False
    opt_out_date: Optional[datetime] = None
    source: str = "WHATSAPP"
    notes: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Lead":
        data = data.copy()
        data.pop("_id", None)
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class Conversation:
    conversation_id: str
    wa_id: str
    lead_id: Optional[str] = None
    user_id: Optional[str] = None
    channel: str = "WHATSAPP"
    status: str = "active"
    unread_count: int = 0
    message_count: int = 0
    human_takeover: bool = False
    takeover_agent_id: Optional[str] = None
    sales_stage: str = "NEW"
    last_sales_action: Optional[str] = None
    visit_readiness_score: int = 0
    summary: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_message_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_customer_message_at: Optional[datetime] = None
    last_ai_message_at: Optional[datetime] = None
    last_human_message_at: Optional[datetime] = None
    opted_out: bool = False
    last_followup_at: Optional[datetime] = None
    followup_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Conversation":
        data = data.copy()
        data.pop("_id", None)
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class Message:
    message_id: str = field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:12]}")
    conversation_id: str = ""
    lead_id: Optional[str] = None
    user_id: Optional[str] = None
    channel: str = "WHATSAPP"
    direction: str = "INBOUND"          # INBOUND or OUTBOUND
    sender_type: str = "CUSTOMER"       # CUSTOMER, AI, HUMAN, SYSTEM
    role: str = "user"                  # Backward compatible: user, assistant, system
    source: str = "WHATSAPP"            # WHATSAPP, AI_AGENT, HUMAN_AGENT, CAMPAIGN
    wa_id: str = ""
    text: str = ""
    message_type: str = "TEXT"          # TEXT, IMAGE, DOCUMENT, AUDIO, VIDEO, LOCATION, INTERACTIVE, TEMPLATE, UNKNOWN
    delivery_status: str = "RECEIVED"   # Backward compatible
    status: str = "RECEIVED"            # RECEIVED, QUEUED, SENDING, ACCEPTED, SENT, DELIVERED, READ, FAILED
    meta_message_id: Optional[str] = None
    whatsapp_message_id: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    media: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_payload: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        data = data.copy()
        data.pop("_id", None)
        # Harmonize legacy role vs sender_type
        if "role" not in data and "sender_type" in data:
            st = str(data["sender_type"]).upper()
            data["role"] = "user" if st == "CUSTOMER" else ("assistant" if st in ("AI", "HUMAN") else "system")
        elif "sender_type" not in data and "role" in data:
            r = str(data["role"]).lower()
            data["sender_type"] = "CUSTOMER" if r in ("user", "customer") else ("AI" if r in ("assistant", "bot") else "SYSTEM")
        # Harmonize whatsapp_message_id vs meta_message_id
        if not data.get("whatsapp_message_id") and data.get("meta_message_id"):
            data["whatsapp_message_id"] = data["meta_message_id"]
        elif not data.get("meta_message_id") and data.get("whatsapp_message_id"):
            data["meta_message_id"] = data["whatsapp_message_id"]
        # Harmonize status vs delivery_status
        if "status" not in data and "delivery_status" in data:
            data["status"] = str(data["delivery_status"]).upper()
        elif "delivery_status" not in data and "status" in data:
            data["delivery_status"] = str(data["status"]).lower()
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class Session:
    session_id: str
    wa_id: str
    state: str = "initial"
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        data = data.copy()
        data.pop("_id", None)
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class Property:
    id: str = field(default_factory=lambda: f"ARIS-{uuid.uuid4().hex[:6].upper()}")
    title: str = ""
    project_name: str = ""
    builder_name: str = "ARIS Premier Builders"
    city: str = "Nagpur"
    locality: str = ""
    address: str = ""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    type: str = "Apartment"
    bhk: str = "2BHK"
    bathrooms: int = 2
    balconies: int = 1
    carpet_area: str = "950 sq.ft"
    built_up_area: str = "1,150 sq.ft"
    price_lakhs: float = 50.0
    price_display: str = "₹50.0 Lakhs"
    price_per_sqft: Optional[float] = None
    possession_status: str = "Ready to Move"
    possession_date: str = "Immediate"
    amenities: List[str] = field(default_factory=lambda: ["Covered Parking", "Lift", "Security"])
    parking: str = "Covered"
    furnishing: str = "Semi-Furnished"
    floor: str = "4 of 12"
    nearby_landmarks: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)
    brochure_url: str = ""
    location_url: str = ""
    description: str = ""
    highlights: List[str] = field(default_factory=list)
    inventory_count: int = 5
    availability_status: str = "Available"
    project_id: str = ""
    organization_id: str = "default_org"
    document_ids: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        if not self.project_name:
            self.project_name = self.title
        if not self.price_display and self.price_lakhs:
            if self.price_lakhs >= 100:
                self.price_display = f"₹{self.price_lakhs / 100.0:.2f} Cr"
            else:
                self.price_display = f"₹{self.price_lakhs:.1f} Lakhs"

    @property
    def property_id(self) -> str:
        return self.id

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["property_id"] = self.id
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Property":
        data = data.copy()
        data.pop("_id", None)
        if "property_id" in data and "id" not in data:
            data["id"] = data["property_id"]
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})



@dataclass
class SiteVisit:
    visit_id: str
    lead_id: str
    wa_id: str
    property_id: str
    visit_date: str
    visit_time: str
    status: str = "scheduled"
    client_name: str = ""
    client_phone: str = ""
    assigned_agent: Optional[str] = None
    notes: Optional[str] = None
    cab_required: bool = True
    pickup_address: Optional[str] = None
    cab_status: str = "pending"  # pending, assigned, dispatched, completed, not_required
    idempotency_key: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SiteVisit":
        data = data.copy()
        data.pop("_id", None)
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class KnowledgeDocument:
    id: str = field(default_factory=lambda: f"doc_{uuid.uuid4().hex[:12]}")
    title: str = ""
    filename: str = ""
    document_type: str = "brochure"
    property_id: Optional[str] = None
    project_id: Optional[str] = None
    city: Optional[str] = None
    locality: Optional[str] = None
    content: str = ""
    chunk_count: int = 0
    status: str = "ready"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeDocument":
        data = data.copy()
        data.pop("_id", None)
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class DocumentChunk:
    id: str = field(default_factory=lambda: f"chk_{uuid.uuid4().hex[:12]}")
    document_id: str = ""
    chunk_index: int = 0
    content: str = ""
    property_id: Optional[str] = None
    project_id: Optional[str] = None
    city: Optional[str] = None
    locality: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentChunk":
        data = data.copy()
        data.pop("_id", None)
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class FollowUp:
    followup_id: str
    lead_id: str
    scheduled_for: datetime
    message_template: str
    status: str = "pending"
    sent_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FollowUp":
        data = data.copy()
        data.pop("_id", None)
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class Campaign:
    campaign_id: str
    title: str
    template_name: str
    target_city: Optional[str] = None
    target_stage: Optional[str] = None
    min_lead_score: int = 0
    language_code: str = "en"
    description: str = ""
    status: str = CampaignStatus.DRAFT
    property_id: Optional[str] = None
    project_id: Optional[str] = None
    bhk_filter: Optional[str] = None
    budget_min: Optional[float] = None
    budget_max: Optional[float] = None
    audience_type: str = "FILTER"
    selected_lead_ids: List[str] = field(default_factory=list)
    total_recipients: int = 0
    eligible_count: int = 0
    queued_count: int = 0
    sent_count: int = 0
    delivered_count: int = 0
    read_count: int = 0
    replied_count: int = 0
    failed_count: int = 0
    excluded_no_consent: int = 0
    excluded_opted_out: int = 0
    excluded_invalid: int = 0
    completed_at: Optional[datetime] = None
    audience_snapshot: List[Dict[str, Any]] = field(default_factory=list)
    audience_snapshot_at: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    cloned_from: Optional[str] = None
    error_summary: Dict[str, Any] = field(default_factory=dict)
    rate_limit_per_second: float = 10.0
    estimated_cost: float = 0.0
    priority: int = 2
    delivery_window_start: Optional[str] = None
    delivery_window_end: Optional[str] = None
    is_deleted: bool = False
    created_by: str = "admin"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    paused_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    validated_at: Optional[datetime] = None
    scheduled_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    updated_by: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Campaign":
        data = data.copy()
        data.pop("_id", None)
        if "campaign_id" not in data:
            data["campaign_id"] = f"cmp_{uuid.uuid4().hex[:10]}"
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class OutboundMessage:
    outbound_id: str = field(default_factory=lambda: f"out_{uuid.uuid4().hex[:12]}")
    channel: str = "WHATSAPP"
    direction: str = "OUTBOUND"
    source: str = OutboundSource.TEST_SEND
    phone_number: str = ""
    normalized_phone: str = ""
    country_code: str = "IN"
    recipient_name: Optional[str] = None
    error_category: Optional[str] = None
    lead_id: Optional[str] = None
    campaign_id: Optional[str] = None
    conversation_id: Optional[str] = None
    message_type: str = "TEMPLATE"
    template_name: str = ""
    template_language: str = "en"
    template_components: List[Dict[str, Any]] = field(default_factory=list)
    meta_message_id: Optional[str] = None
    status: str = OutboundStatus.QUEUED
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    idempotency_key: Optional[str] = None
    retry_count: int = 0
    created_by: str = "system"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    accepted_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    read_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OutboundMessage":
        data = data.copy()
        data.pop("_id", None)
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class CampaignRecipient:
    campaign_id: str
    lead_id: str
    phone: str
    name: str = "Valued Customer"
    status: str = "PENDING"  # PENDING, QUEUED, SENT, DELIVERED, READ, FAILED, SKIPPED
    outbound_id: Optional[str] = None
    meta_message_id: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    error_category: Optional[str] = None
    queued_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    read_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None
    retry_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CampaignRecipient":
        data = data.copy()
        data.pop("_id", None)
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class AnalyticsEvent:
    event_id: str = field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:12]}")
    event_type: str = ""
    lead_id: Optional[str] = None
    wa_id: str = ""
    property_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalyticsEvent":
        data = data.copy()
        data.pop("_id", None)
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


# =============================================================================
# MONGODB CONNECTION MANAGER (SINGLETON)
# =============================================================================

class MongoDB:
    _instance: Optional["MongoDB"] = None

    def __init__(self):
        self.uri = Config.MONGODB_URI
        self.db_name = Config.DATABASE_NAME
        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None
        self._connected: bool = False
        self.connect()

    def connect(self):
        """Establishes connection to MongoDB with timeouts and initializes indexes."""
        try:
            kwargs = {
                "serverSelectionTimeoutMS": 10000,
                "connectTimeoutMS": 10000,
                "socketTimeoutMS": 15000,
                "maxPoolSize": 50
            }
            try:
                import certifi
                kwargs["tlsCAFile"] = certifi.where()
            except Exception:
                pass

            self.client = MongoClient(self.uri, **kwargs)
            self.db = self.client[self.db_name]
            self.client.admin.command("ping")
            self._connected = True
            print(f"[MONGODB] Connected successfully to database '{self.db_name}'")
            self.init_indexes()
        except (ConnectionFailure, ServerSelectionTimeoutError) as ex:
            self._connected = False
            print(f"[MONGODB WARNING] Database connection failed: {ex}. Operating in degraded offline cache mode.")
        except Exception as ex:
            self._connected = False
            print(f"[MONGODB ERROR] Connection error: {ex}")

    def is_connected(self) -> bool:
        if not self._connected or self.client is None:
            return False
        try:
            self.client.admin.command("ping")
            self._connected = True
            return True
        except Exception:
            self._connected = False
            return False

    def init_indexes(self):
        """Ensures indexes exist across all collections."""
        if not self._connected or self.db is None:
            return
        
        index_defs = [
            (self.users, [("wa_id", ASCENDING)], {"unique": True}),
            (self.conversations, [("conversation_id", ASCENDING)], {"unique": True}),
            (self.conversations, [("wa_id", ASCENDING), ("status", ASCENDING)], {}),
            (self.conversations, [("lead_id", ASCENDING)], {}),
            (self.conversations, [("sales_stage", ASCENDING)], {}),
            (self.conversations, [("last_message_at", DESCENDING)], {}),
            (self.messages, [("conversation_id", ASCENDING), ("created_at", ASCENDING)], {}),
            (self.messages, [("lead_id", ASCENDING), ("created_at", ASCENDING)], {}),
            (self.messages, [("whatsapp_message_id", ASCENDING)], {}),
            (self.messages, [("meta_message_id", ASCENDING)], {}),
            (self.messages, [("direction", ASCENDING)], {}),
            (self.messages, [("sender_type", ASCENDING)], {}),
            (self.messages, [("status", ASCENDING)], {}),
            (self.messages, [("created_at", ASCENDING)], {}),
            (self.sessions, [("wa_id", ASCENDING)], {"unique": True}),
            (self.leads, [("lead_id", ASCENDING)], {"unique": True}),
            (self.leads, [("wa_id", ASCENDING)], {"unique": True, "sparse": True}),
            (self.customer_memory, [("lead_id", ASCENDING)], {"unique": True}),
            (self.sales_memory, [("lead_id", ASCENDING)], {"unique": True}),
            (self.sales_events, [("lead_id", ASCENDING), ("created_at", ASCENDING)], {}),
            (self.sales_events, [("conversation_id", ASCENDING), ("created_at", ASCENDING)], {}),
            (self.sales_events, [("event_type", ASCENDING)], {}),
            (self.properties, [("property_id", ASCENDING)], {"unique": True, "sparse": True}),
            (self.properties, [("id", ASCENDING)], {"unique": True, "sparse": True}),
            (self.properties, [("city", ASCENDING), ("property_type", ASCENDING)], {}),
            (self.visits, [("visit_id", ASCENDING)], {"unique": True}),
            (self.campaigns, [("campaign_id", ASCENDING)], {"unique": True}),
            (self.campaigns, [("status", ASCENDING), ("created_at", DESCENDING)], {}),
            (self.campaigns, [("scheduled_at", ASCENDING), ("status", ASCENDING)], {}),
            (self.campaign_recipients, [("campaign_id", ASCENDING), ("status", ASCENDING)], {}),
            (self.campaign_recipients, [("campaign_id", ASCENDING), ("lead_id", ASCENDING)], {"unique": True}),
            (self.campaign_recipients, [("meta_message_id", ASCENDING)], {"sparse": True}),
            (self.whatsapp_outbound_messages, [("outbound_id", ASCENDING)], {"unique": True}),
            (self.whatsapp_outbound_messages, [("meta_message_id", ASCENDING)], {}),
            (self.whatsapp_outbound_messages, [("idempotency_key", ASCENDING)], {}),
        ]

        created = 0
        for coll, keys, opts in index_defs:
            try:
                coll.create_index(keys, **opts)
                created += 1
            except Exception:
                pass
        print(f"[MONGODB] Database indexes verified and initialized ({created}/{len(index_defs)} active).")

    @property
    def users(self) -> Collection:
        return self.db["users"]

    @property
    def conversations(self) -> Collection:
        return self.db["conversations"]

    @property
    def messages(self) -> Collection:
        return self.db["messages"]

    @property
    def sessions(self) -> Collection:
        return self.db["sessions"]

    @property
    def webhook_events(self) -> Collection:
        return self.db["webhook_events"]

    @property
    def leads(self) -> Collection:
        return self.db["leads"]

    @property
    def customer_memory(self) -> Collection:
        return self.db["customer_memory"]

    @property
    def sales_memory(self) -> Collection:
        return self.db["sales_memory"]

    @property
    def sales_events(self) -> Collection:
        return self.db["sales_events"]

    @property
    def properties(self) -> Collection:
        return self.db["properties"]

    @property
    def visits(self) -> Collection:
        return self.db["visits"]

    @property
    def followups(self) -> Collection:
        return self.db["followups"]

    @property
    def campaigns(self) -> Collection:
        return self.db["campaigns"]

    @property
    def campaign_recipients(self) -> Collection:
        return self.db["campaign_recipients"]

    @property
    def whatsapp_outbound_messages(self) -> Collection:
        return self.db["whatsapp_outbound_messages"]

    @property
    def events(self) -> Collection:
        return self.db["events"]


def get_db() -> MongoDB:
    """Returns singleton MongoDB manager."""
    if MongoDB._instance is None:
        MongoDB._instance = MongoDB()
    return MongoDB._instance


# =============================================================================
# REPOSITORIES & DATA ACCESS LAYER
# =============================================================================

class UserRepository:
    _cache: Dict[str, Dict[str, Any]] = {}

    def __init__(self):
        self._db = get_db()

    def get_or_create(self, wa_id: str, profile_name: str = "", phone: str = "") -> Dict[str, Any]:
        if self._db.is_connected():
            try:
                user = self._db.users.find_one({"wa_id": wa_id})
                if user:
                    self._db.users.update_one({"wa_id": wa_id}, {"$set": {"last_seen": datetime.now(timezone.utc)}})
                    user["last_seen"] = datetime.now(timezone.utc)
                    self._cache[wa_id] = user
                    return user
                new_u = User(wa_id=wa_id, profile_name=profile_name, phone_number=phone or wa_id).to_dict()
                self._db.users.insert_one(new_u)
                self._cache[wa_id] = new_u
                return new_u
            except Exception:
                pass

        if wa_id not in self._cache:
            self._cache[wa_id] = User(wa_id=wa_id, profile_name=profile_name, phone_number=phone or wa_id).to_dict()
        else:
            self._cache[wa_id]["last_seen"] = datetime.now(timezone.utc)
        return self._cache[wa_id]

    def get_by_wa_id(self, wa_id: str) -> Optional[Dict[str, Any]]:
        if self._db.is_connected():
            try:
                return self._db.users.find_one({"wa_id": wa_id})
            except Exception:
                pass
        return self._cache.get(wa_id)


class ConversationRepository:
    _cache: Dict[str, Dict[str, Any]] = {}

    def __init__(self):
        self._db = get_db()

    def get_active_conversation(self, wa_id: str) -> Optional[Dict[str, Any]]:
        if self._db.is_connected():
            try:
                return self._db.conversations.find_one({"wa_id": wa_id, "status": "active"})
            except Exception:
                pass
        return self._cache.get(wa_id)

    def get_or_create_conversation(self, wa_id: str, profile_name: str = "", phone: str = "") -> Dict[str, Any]:
        """Ensures both lead and conversation exist and returns the conversation."""
        from database import DB
        lead = DB.leads.get_or_create(wa_id, name=profile_name, phone=phone or wa_id)
        lead_id = lead.get("_id") if isinstance(lead, dict) else getattr(lead, "id", None)
        conv = self.create_if_not_exists(wa_id, lead_id=str(lead_id) if lead_id else None)
        if "_id" not in conv and "conversation_id" in conv:
            conv["_id"] = conv["conversation_id"]
        return conv

    get_or_create = get_or_create_conversation

    def create_if_not_exists(self, wa_id: str, lead_id: Optional[str] = None, user_id: Optional[str] = None) -> Dict[str, Any]:
        active = self.get_active_conversation(wa_id)
        if active:
            if "_id" not in active and "conversation_id" in active:
                active["_id"] = active["conversation_id"]
            if lead_id and not active.get("lead_id"):
                active["lead_id"] = lead_id
                if self._db.is_connected():
                    try:
                        self._db.conversations.update_one({"conversation_id": active["conversation_id"]}, {"$set": {"lead_id": lead_id}})
                    except Exception:
                        pass
            return active

        conv_id = f"conv_{uuid.uuid4().hex[:12]}"
        new_conv = Conversation(
            conversation_id=conv_id,
            wa_id=wa_id,
            lead_id=lead_id,
            user_id=user_id,
            started_at=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
            last_message_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        ).to_dict()
        new_conv["_id"] = conv_id
        if self._db.is_connected():
            try:
                self._db.conversations.insert_one(new_conv)
            except Exception:
                pass
        self._cache[wa_id] = new_conv
        return new_conv

    def get_by_id(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        if not conversation_id:
            return None
        if self._db.is_connected():
            try:
                doc = self._db.conversations.find_one({"$or": [{"conversation_id": str(conversation_id)}, {"_id": str(conversation_id)}]})
                if doc:
                    if "_id" not in doc or not doc["_id"]:
                        doc["_id"] = doc.get("conversation_id")
                    return doc
            except Exception:
                pass
        for c in self._cache.values():
            if c.get("conversation_id") == conversation_id or c.get("_id") == conversation_id:
                return c
        return None

    def update_conversation(self, conversation_id: str, updates: Dict[str, Any]) -> bool:
        updates = updates.copy()
        updates["updated_at"] = datetime.now(timezone.utc)
        if self._db.is_connected():
            try:
                self._db.conversations.update_one(
                    {"$or": [{"conversation_id": str(conversation_id)}, {"_id": str(conversation_id)}]},
                    {"$set": updates}
                )
                return True
            except Exception:
                pass
        for c in self._cache.values():
            if c.get("conversation_id") == conversation_id or c.get("_id") == conversation_id:
                c.update(updates)
                return True
        return False

    def get_by_lead_id(self, lead_id: str) -> Optional[Dict[str, Any]]:
        if not lead_id:
            return None
        if self._db.is_connected():
            try:
                doc = self._db.conversations.find_one({"lead_id": str(lead_id)})
                if doc:
                    return doc
            except Exception:
                pass
        for c in self._cache.values():
            if c.get("lead_id") == lead_id:
                return c
        return None

    def increment_message_count(self, conversation_id: str, count: int = 1) -> bool:
        if self._db.is_connected():
            try:
                self._db.conversations.update_one({"conversation_id": conversation_id}, {"$inc": {"message_count": count}})
                return True
            except Exception:
                pass
        for c in self._cache.values():
            if c.get("conversation_id") == conversation_id:
                c["message_count"] = c.get("message_count", 0) + count
        return True

    def increment_unread(self, conversation_id: str, count: int = 1) -> bool:
        if self._db.is_connected():
            try:
                self._db.conversations.update_one({"conversation_id": conversation_id}, {"$inc": {"unread_count": count}})
                return True
            except Exception:
                pass
        for c in self._cache.values():
            if c.get("conversation_id") == conversation_id:
                c["unread_count"] = c.get("unread_count", 0) + count
        return True

    def reset_unread(self, conversation_id: str) -> bool:
        if self._db.is_connected():
            try:
                self._db.conversations.update_one({"conversation_id": conversation_id}, {"$set": {"unread_count": 0}})
                return True
            except Exception:
                pass
        for c in self._cache.values():
            if c.get("conversation_id") == conversation_id:
                c["unread_count"] = 0
        return True

    def update_last_message(self, conversation_id: str) -> bool:
        now = datetime.now(timezone.utc)
        if self._db.is_connected():
            try:
                self._db.conversations.update_one(
                    {"conversation_id": conversation_id},
                    {"$set": {"last_message_at": now, "updated_at": now}}
                )
                return True
            except Exception:
                pass
        for c in self._cache.values():
            if c.get("conversation_id") == conversation_id:
                c["last_message_at"] = now
                c["updated_at"] = now
        return True

    def update_timestamps(self, conversation_id: str, sender_type: str = "CUSTOMER") -> bool:
        now = datetime.now(timezone.utc)
        st = str(sender_type).upper()
        field_name = "last_customer_message_at" if st == "CUSTOMER" else (
            "last_ai_message_at" if st == "AI" else "last_human_message_at"
        )
        if self._db.is_connected():
            try:
                self._db.conversations.update_one(
                    {"conversation_id": conversation_id},
                    {"$set": {field_name: now, "last_message_at": now, "updated_at": now}}
                )
                return True
            except Exception:
                pass
        for c in self._cache.values():
            if c.get("conversation_id") == conversation_id:
                c[field_name] = now
                c["last_message_at"] = now
                c["updated_at"] = now
        return True

    def update_sales_state(
        self,
        conversation_id: str,
        sales_stage: Optional[str] = None,
        last_sales_action: Optional[str] = None,
        visit_readiness_score: Optional[int] = None
    ) -> bool:
        now = datetime.now(timezone.utc)
        updates: Dict[str, Any] = {"updated_at": now}
        if sales_stage is not None:
            updates["sales_stage"] = str(sales_stage)
        if last_sales_action is not None:
            updates["last_sales_action"] = str(last_sales_action)
        if visit_readiness_score is not None:
            updates["visit_readiness_score"] = int(visit_readiness_score)

        if self._db.is_connected():
            try:
                self._db.conversations.update_one({"conversation_id": conversation_id}, {"$set": updates})
                return True
            except Exception:
                pass
        for c in self._cache.values():
            if c.get("conversation_id") == conversation_id:
                c.update(updates)
        return True

    def update_summary(self, conversation_id: str, summary: str) -> bool:
        now = datetime.now(timezone.utc)
        if self._db.is_connected():
            try:
                self._db.conversations.update_one(
                    {"conversation_id": conversation_id},
                    {"$set": {"summary": summary, "updated_at": now}}
                )
                return True
            except Exception:
                pass
        for c in self._cache.values():
            if c.get("conversation_id") == conversation_id:
                c["summary"] = summary
                c["updated_at"] = now
        return True

    def list_conversations(self, limit: int = 50, filter_query: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        q = filter_query or {}
        if self._db.is_connected():
            try:
                return list(self._db.conversations.find(q).sort("last_message_at", -1).limit(limit))
            except Exception:
                pass
        res = list(self._cache.values())
        return sorted(res, key=lambda x: str(x.get("last_message_at", "")), reverse=True)[:limit]

    def set_human_takeover(self, conversation_id: str, enabled: bool, agent_id: str = "human_agent") -> bool:
        now = datetime.now(timezone.utc)
        if self._db.is_connected():
            try:
                self._db.conversations.update_one(
                    {"conversation_id": conversation_id},
                    {"$set": {"human_takeover": enabled, "takeover_agent_id": agent_id if enabled else None, "updated_at": now}}
                )
                return True
            except Exception:
                pass
        for c in self._cache.values():
            if c.get("conversation_id") == conversation_id:
                c["human_takeover"] = enabled
                c["takeover_agent_id"] = agent_id if enabled else None
                c["updated_at"] = now
        return True

    def reset_human_takeover(self, target: str) -> bool:
        if not target:
            return False
        now = datetime.now(timezone.utc)
        if self._db.is_connected():
            try:
                self._db.conversations.update_many(
                    {"$or": [{"conversation_id": str(target)}, {"wa_id": str(target)}]},
                    {"$set": {"human_takeover": False, "takeover_agent_id": None, "updated_at": now}}
                )
            except Exception:
                pass
        for c in self._cache.values():
            if c.get("conversation_id") == target or c.get("wa_id") == target:
                c["human_takeover"] = False
                c["takeover_agent_id"] = None
                c["updated_at"] = now
        return True

    def set_opted_out(self, target: str, opted_out: bool = True) -> bool:
        if not target:
            return False
        now = datetime.now(timezone.utc)
        updates = {
            "opted_out": opted_out,
            "status": "opted_out" if opted_out else "active",
            "updated_at": now
        }
        if self._db.is_connected():
            try:
                self._db.conversations.update_many(
                    {"$or": [{"conversation_id": str(target)}, {"wa_id": str(target)}]},
                    {"$set": updates}
                )
            except Exception:
                pass
        for c in self._cache.values():
            if c.get("conversation_id") == target or c.get("wa_id") == target:
                c.update(updates)
        return True

    def record_followup(self, conversation_id: str) -> bool:
        now = datetime.now(timezone.utc)
        if self._db.is_connected():
            try:
                self._db.conversations.update_one(
                    {"conversation_id": str(conversation_id)},
                    {
                        "$set": {"last_followup_at": now, "updated_at": now},
                        "$inc": {"followup_count": 1}
                    }
                )
            except Exception:
                pass
        for c in self._cache.values():
            if c.get("conversation_id") == conversation_id:
                c["last_followup_at"] = now
                c["updated_at"] = now
                c["followup_count"] = c.get("followup_count", 0) + 1
        return True


class MessageRepository:
    _cache: List[Dict[str, Any]] = []

    def __init__(self):
        self._db = get_db()

    def check_idempotency_wamid(self, whatsapp_message_id: str) -> bool:
        """Returns True if message with this Meta ID already exists."""
        if not whatsapp_message_id:
            return False
        if self._db.is_connected():
            try:
                found = self._db.messages.find_one({
                    "$or": [
                        {"whatsapp_message_id": str(whatsapp_message_id)},
                        {"meta_message_id": str(whatsapp_message_id)}
                    ]
                })
                return found is not None
            except Exception:
                pass
        return any(
            m.get("whatsapp_message_id") == whatsapp_message_id or m.get("meta_message_id") == whatsapp_message_id
            for m in self._cache
        )

    def save_inbound_message(
        self,
        conversation_id: str,
        wa_id: str,
        text: str,
        lead_id: Optional[str] = None,
        user_id: Optional[str] = None,
        whatsapp_message_id: Optional[str] = None,
        message_type: str = "TEXT",
        media: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        raw_payload: Optional[Dict[str, Any]] = None,
        source: str = "WHATSAPP"
    ) -> Dict[str, Any]:
        """Persists customer inbound message before AI processing."""
        msg = Message(
            message_id=f"msg_{uuid.uuid4().hex[:12]}",
            conversation_id=conversation_id,
            lead_id=lead_id,
            user_id=user_id,
            channel="WHATSAPP",
            direction="INBOUND",
            sender_type="CUSTOMER",
            role="user",
            source="WHATSAPP",
            wa_id=wa_id,
            text=text,
            message_type=message_type.upper(),
            delivery_status="RECEIVED",
            status="RECEIVED",
            meta_message_id=whatsapp_message_id,
            whatsapp_message_id=whatsapp_message_id,
            media=media,
            metadata=metadata or {},
            raw_payload=raw_payload,
            timestamp=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc)
        ).to_dict()

        if self._db.is_connected():
            try:
                self._db.messages.insert_one(msg)
            except Exception:
                pass
        self._cache.append(msg)
        return msg

    def save_outbound_ai_message(
        self,
        conversation_id: str,
        wa_id: str,
        text: str,
        lead_id: Optional[str] = None,
        whatsapp_message_id: Optional[str] = None,
        message_type: str = "TEXT",
        metadata: Optional[Dict[str, Any]] = None,
        status: str = "ACCEPTED"
    ) -> Dict[str, Any]:
        """Persists outbound AI response."""
        msg = Message(
            message_id=f"msg_{uuid.uuid4().hex[:12]}",
            conversation_id=conversation_id,
            lead_id=lead_id,
            channel="WHATSAPP",
            direction="OUTBOUND",
            sender_type="AI",
            role="assistant",
            source="AI_AGENT",
            wa_id=wa_id,
            text=text,
            message_type=message_type.upper(),
            delivery_status=status,
            status=status,
            meta_message_id=whatsapp_message_id,
            whatsapp_message_id=whatsapp_message_id,
            metadata=metadata or {},
            timestamp=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc)
        ).to_dict()

        if self._db.is_connected():
            try:
                self._db.messages.insert_one(msg)
            except Exception:
                pass
        self._cache.append(msg)
        return msg

    def save_outbound_human_message(
        self,
        conversation_id: str,
        wa_id: str,
        text: str,
        lead_id: Optional[str] = None,
        user_id: Optional[str] = None,
        whatsapp_message_id: Optional[str] = None,
        message_type: str = "TEXT",
        metadata: Optional[Dict[str, Any]] = None,
        status: str = "SENT",
        agent_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Persists outbound human agent message."""
        meta = metadata.copy() if metadata else {}
        if agent_id:
            meta["agent_id"] = agent_id

        msg = Message(
            message_id=f"msg_{uuid.uuid4().hex[:12]}",
            conversation_id=conversation_id,
            lead_id=lead_id,
            user_id=user_id,
            channel="WHATSAPP",
            direction="OUTBOUND",
            sender_type="HUMAN",
            role="assistant",
            source="HUMAN_AGENT",
            wa_id=wa_id,
            text=text,
            message_type=message_type.upper(),
            delivery_status=status,
            status=status,
            meta_message_id=whatsapp_message_id,
            whatsapp_message_id=whatsapp_message_id,
            metadata=meta,
            timestamp=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc)
        ).to_dict()
        if agent_id:
            msg["agent_id"] = agent_id

        if self._db.is_connected():
            try:
                self._db.messages.insert_one(msg)
            except Exception:
                pass
        self._cache.append(msg)
        return msg

    # Backward compatibility aliases
    def save_user_message(
        self,
        conversation_id: str,
        wa_id: str,
        text: str,
        meta_message_id: str = "",
        message_type: str = "text",
        raw_payload: Optional[Dict] = None,
        lead_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        return self.save_inbound_message(
            conversation_id=conversation_id,
            wa_id=wa_id,
            text=text,
            lead_id=lead_id,
            user_id=user_id,
            whatsapp_message_id=meta_message_id,
            message_type=message_type,
            raw_payload=raw_payload
        )

    def save_assistant_message(
        self,
        conversation_id: str,
        wa_id: str,
        text: str,
        message_type: str = "text",
        lead_id: Optional[str] = None,
        meta_message_id: Optional[str] = None
    ) -> Dict[str, Any]:
        return self.save_outbound_ai_message(
            conversation_id=conversation_id,
            wa_id=wa_id,
            text=text,
            lead_id=lead_id,
            whatsapp_message_id=meta_message_id,
            message_type=message_type
        )

    def update_delivery_status_by_wamid(
        self,
        whatsapp_message_id: str,
        status: str,
        error_code: str = "",
        error_message: str = ""
    ) -> bool:
        """Correlates Meta webhook status event with persisted message."""
        if not whatsapp_message_id:
            return False
        st = str(status).upper()
        updates: Dict[str, Any] = {
            "status": st,
            "delivery_status": st.lower()
        }
        if error_code:
            updates["error_code"] = str(error_code)
        if error_message:
            updates["error_message"] = str(error_message)

        if self._db.is_connected():
            try:
                self._db.messages.update_one(
                    {
                        "$or": [
                            {"whatsapp_message_id": str(whatsapp_message_id)},
                            {"meta_message_id": str(whatsapp_message_id)}
                        ]
                    },
                    {"$set": updates}
                )
                return True
            except Exception:
                pass
        for m in self._cache:
            if m.get("whatsapp_message_id") == whatsapp_message_id or m.get("meta_message_id") == whatsapp_message_id:
                m.update(updates)
        return True

    def get_message_by_wamid(self, wamid: str) -> Optional[Dict[str, Any]]:
        if not wamid:
            return None
        if self._db.is_connected():
            try:
                doc = self._db.messages.find_one({"$or": [{"whatsapp_message_id": str(wamid)}, {"meta_message_id": str(wamid)}]})
                if doc:
                    return doc
            except Exception:
                pass
        for m in reversed(self._cache):
            if m.get("whatsapp_message_id") == wamid or m.get("meta_message_id") == wamid:
                return m
        return None

    def get_last_messages(self, conversation_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        msgs = []
        if self._db.is_connected():
            try:
                cursor = self._db.messages.find({"conversation_id": conversation_id}).sort("created_at", -1).limit(limit)
                msgs = list(cursor)
                msgs.reverse()
            except Exception:
                pass
        if not msgs:
            filtered = [m for m in self._cache if m.get("conversation_id") == conversation_id]
            msgs = sorted(filtered, key=lambda x: str(x.get("created_at", "")))[-limit:]
        for m in msgs:
            if "role" not in m and "sender_type" in m:
                m["role"] = "user" if m["sender_type"] == "CUSTOMER" else "assistant"
            elif "sender_type" not in m and "role" in m:
                m["sender_type"] = "CUSTOMER" if m["role"] == "user" else "AI"
        return msgs

    def get_paginated_messages(
        self,
        conversation_id: str,
        limit: int = 50,
        before: Optional[str] = None,
        after: Optional[str] = None
    ) -> Dict[str, Any]:
        """Cursor/timestamp-based pagination for messages."""
        limit = min(max(int(limit), 1), 200)
        query: Dict[str, Any] = {"conversation_id": conversation_id}
        if before:
            try:
                query["created_at"] = {"$lt": datetime.fromisoformat(before.replace("Z", "+00:00"))}
            except Exception:
                pass
        elif after:
            try:
                query["created_at"] = {"$gt": datetime.fromisoformat(after.replace("Z", "+00:00"))}
            except Exception:
                pass

        msgs = []
        if self._db.is_connected():
            try:
                cursor = self._db.messages.find(query).sort("created_at", 1).limit(limit)
                msgs = list(cursor)
            except Exception:
                pass

        if not msgs:
            filtered = [m for m in self._cache if m.get("conversation_id") == conversation_id]
            msgs = sorted(filtered, key=lambda x: str(x.get("created_at", "")))[:limit]

        has_more = len(msgs) == limit
        next_cursor = msgs[-1]["created_at"].isoformat() if (has_more and msgs and isinstance(msgs[-1].get("created_at"), datetime)) else None
        prev_cursor = msgs[0]["created_at"].isoformat() if (msgs and isinstance(msgs[0].get("created_at"), datetime)) else None

        return {
            "messages": msgs,
            "count": len(msgs),
            "has_more": has_more,
            "next_cursor": next_cursor,
            "prev_cursor": prev_cursor
        }

    def search_messages(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        lead_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Searches message text."""
        if not query:
            return []
        filter_dict: Dict[str, Any] = {
            "text": {"$regex": re.escape(query), "$options": "i"}
        }
        if conversation_id:
            filter_dict["conversation_id"] = conversation_id
        if lead_id:
            filter_dict["lead_id"] = lead_id

        if self._db.is_connected():
            try:
                return list(self._db.messages.find(filter_dict).sort("created_at", -1).limit(limit))
            except Exception:
                pass

        res = []
        lower_q = query.lower()
        for m in reversed(self._cache):
            if lower_q in str(m.get("text", "")).lower():
                if conversation_id and m.get("conversation_id") != conversation_id:
                    continue
                if lead_id and m.get("lead_id") != lead_id:
                    continue
                res.append(m)
                if len(res) >= limit:
                    break
        return res


class CustomerMemoryRepository:
    """Manages persistent Customer Memory in MongoDB."""
    _cache: Dict[str, Dict[str, Any]] = {}

    def __init__(self):
        self._db = get_db()

    def get_by_lead_id(self, lead_id: str) -> Optional[Dict[str, Any]]:
        if not lead_id:
            return None
        if self._db.is_connected():
            try:
                doc = self._db.customer_memory.find_one({"lead_id": str(lead_id)})
                if doc:
                    return doc
            except Exception:
                pass
        return self._cache.get(str(lead_id))

    def save_or_update(self, lead_id: str, memory_data: Dict[str, Any]) -> Dict[str, Any]:
        if not lead_id:
            return {}
        memory_data["lead_id"] = str(lead_id)
        memory_data["updated_at"] = datetime.now(timezone.utc)
        if "created_at" not in memory_data:
            memory_data["created_at"] = datetime.now(timezone.utc)

        if self._db.is_connected():
            try:
                self._db.customer_memory.update_one(
                    {"lead_id": str(lead_id)},
                    {"$set": memory_data},
                    upsert=True
                )
            except Exception:
                pass
        self._cache[str(lead_id)] = memory_data
        return memory_data

    def delete(self, lead_id: str) -> bool:
        if not lead_id:
            return False
        if self._db.is_connected():
            try:
                self._db.customer_memory.delete_many({"lead_id": str(lead_id)})
            except Exception:
                pass
        self._cache.pop(str(lead_id), None)
        return True


class SalesMemoryRepository:
    """Manages persistent Sales Memory in MongoDB."""
    _cache: Dict[str, Dict[str, Any]] = {}

    def __init__(self):
        self._db = get_db()

    def get_by_lead_id(self, lead_id: str) -> Optional[Dict[str, Any]]:
        if not lead_id:
            return None
        if self._db.is_connected():
            try:
                doc = self._db.sales_memory.find_one({"lead_id": str(lead_id)})
                if doc:
                    return doc
            except Exception:
                pass
        return self._cache.get(str(lead_id))

    def save_or_update(self, lead_id: str, sales_data: Dict[str, Any]) -> Dict[str, Any]:
        if not lead_id:
            return {}
        sales_data["lead_id"] = str(lead_id)
        sales_data["updated_at"] = datetime.now(timezone.utc)
        if "created_at" not in sales_data:
            sales_data["created_at"] = datetime.now(timezone.utc)

        if self._db.is_connected():
            try:
                self._db.sales_memory.update_one(
                    {"lead_id": str(lead_id)},
                    {"$set": sales_data},
                    upsert=True
                )
            except Exception:
                pass
        self._cache[str(lead_id)] = sales_data
        return sales_data

    def save_memory(self, memory: Any) -> Dict[str, Any]:
        data = memory.to_dict() if hasattr(memory, "to_dict") else dict(memory)
        lead_id = data.get("lead_id") or data.get("phone")
        return self.save_or_update(lead_id, data)

    def delete(self, lead_id: str) -> bool:
        if not lead_id:
            return False
        if self._db.is_connected():
            try:
                self._db.sales_memory.delete_many({"lead_id": str(lead_id)})
            except Exception:
                pass
        self._cache.pop(str(lead_id), None)
        return True


class SalesEventRepository:
    """Audit log for structured sales events."""
    _cache: List[Dict[str, Any]] = []

    def __init__(self):
        self._db = get_db()

    def log_sales_event(
        self,
        event_type: str,
        lead_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        evt = {
            "event_id": f"sevt_{uuid.uuid4().hex[:12]}",
            "event_type": str(event_type).upper(),
            "lead_id": str(lead_id) if lead_id else None,
            "conversation_id": str(conversation_id) if conversation_id else None,
            "metadata": metadata or {},
            "timestamp": timestamp or datetime.now(timezone.utc),
            "created_at": datetime.now(timezone.utc)
        }
        if self._db.is_connected():
            try:
                self._db.sales_events.insert_one(evt)
            except Exception:
                pass
        self._cache.append(evt)
        return evt

    def get_lead_events(self, lead_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        if not lead_id:
            return []
        if self._db.is_connected():
            try:
                return list(self._db.sales_events.find({"lead_id": str(lead_id)}).sort("created_at", -1).limit(limit))
            except Exception:
                pass
        return [e for e in reversed(self._cache) if e.get("lead_id") == lead_id][:limit]

    def get_events_for_conversation(self, conversation_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        if not conversation_id:
            return []
        if self._db.is_connected():
            try:
                return list(self._db.sales_events.find({"conversation_id": str(conversation_id)}).sort("created_at", -1).limit(limit))
            except Exception:
                pass
        return [e for e in reversed(self._cache) if e.get("conversation_id") == conversation_id][:limit]

    def get_funnel_timestamps(self, lead_id: str) -> Dict[str, Any]:
        evts = self.get_lead_events(lead_id, limit=200)
        timestamps = {}
        for e in sorted(evts, key=lambda x: str(x.get("created_at", ""))):
            et = e.get("event_type", "").lower()
            if et not in timestamps:
                timestamps[et] = e.get("timestamp") or e.get("created_at")
        return timestamps


class SessionRepository:
    _cache: Dict[str, Dict[str, Any]] = {}

    def __init__(self):
        self._db = get_db()

    def get_or_create_session(self, wa_id: str) -> Dict[str, Any]:
        if self._db.is_connected():
            try:
                session = self._db.sessions.find_one({"wa_id": wa_id})
                if session:
                    self._cache[wa_id] = session
                    return session
                new_s = Session(session_id=f"sess_{uuid.uuid4().hex[:12]}", wa_id=wa_id).to_dict()
                self._db.sessions.insert_one(new_s)
                self._cache[wa_id] = new_s
                return new_s
            except Exception:
                pass

        if wa_id not in self._cache:
            self._cache[wa_id] = Session(session_id=f"sess_{uuid.uuid4().hex[:12]}", wa_id=wa_id).to_dict()
        return self._cache[wa_id]

    def update_session(self, wa_id: str, state: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> bool:
        now = datetime.now(timezone.utc)
        updates: Dict[str, Any] = {"updated_at": now}
        if state is not None:
            updates["state"] = state
        if context is not None:
            updates["context"] = context

        if self._db.is_connected():
            try:
                self._db.sessions.update_one({"wa_id": wa_id}, {"$set": updates})
            except Exception:
                pass

        if wa_id in self._cache:
            self._cache[wa_id].update(updates)
        return True


class LeadRepository:
    _cache: Dict[str, Dict[str, Any]] = {}
    _wa_to_id: Dict[str, str] = {}

    def __init__(self):
        self._db = get_db()
        self._db_manager = self._db

    def get_or_create(self, wa_id: str, name: str = "Anonymous", phone: str = "") -> Dict[str, Any]:
        if self._db.is_connected():
            try:
                lead = self._db.leads.find_one({"wa_id": wa_id})
                if lead:
                    self._sync_cache(lead)
                    return lead
                lead_id = f"lead_{uuid.uuid4().hex[:10]}"
                new_lead = Lead(lead_id=lead_id, wa_id=wa_id, name=name or "Anonymous", phone=phone or wa_id).to_dict()
                self._db.leads.insert_one(new_lead)
                self._sync_cache(new_lead)
                return new_lead
            except Exception:
                pass

        cached_id = self._wa_to_id.get(str(wa_id))
        if cached_id and cached_id in self._cache:
            return self._cache[cached_id]

        lead_id = f"lead_{uuid.uuid4().hex[:10]}"
        new_lead = Lead(lead_id=lead_id, wa_id=wa_id, name=name or "Anonymous", phone=phone or wa_id).to_dict()
        self._sync_cache(new_lead)
        return new_lead

    def get_by_id(self, lead_id: str) -> Optional[Dict[str, Any]]:
        if not lead_id:
            return None
        if self._db.is_connected():
            try:
                doc = self._db.leads.find_one({"lead_id": str(lead_id)})
                if doc:
                    self._sync_cache(doc)
                    return doc
            except Exception:
                pass
        return self._cache.get(str(lead_id))

    def get_by_wa_id(self, wa_id: str) -> Optional[Dict[str, Any]]:
        if not wa_id:
            return None
        if self._db.is_connected():
            try:
                doc = self._db.leads.find_one({"wa_id": str(wa_id)})
                if doc:
                    self._sync_cache(doc)
                    return doc
            except Exception:
                pass
        cached_id = self._wa_to_id.get(str(wa_id))
        if cached_id:
            return self._cache.get(cached_id)
        return None

    def get_by_phone(self, phone: str) -> Optional[Any]:
        doc = self.get_by_wa_id(str(phone))
        if not doc and self._db.is_connected():
            try:
                doc = self._db.leads.find_one({"$or": [{"phone": str(phone)}, {"wa_id": str(phone)}]})
                if doc:
                    self._sync_cache(doc)
            except Exception:
                pass
        if doc and isinstance(doc, dict):
            class LeadWrapper(dict):
                def __getattr__(self, name):
                    if name == "preferred_bhk":
                        bhk_val = self.get("bhk")
                        if isinstance(bhk_val, list) and bhk_val:
                            b = bhk_val[0]
                        else:
                            b = str(bhk_val or "")
                        m = re.match(r"^([1-5])\s*BHK$", b, re.IGNORECASE)
                        if m:
                            return f"{m.group(1)} BHK"
                        return b
                    return self.get(name)
            return LeadWrapper(doc)
        return doc

    def update(self, lead_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not lead_id or not updates:
            return None
        updates["updated_at"] = datetime.now(timezone.utc)
        if self._db.is_connected():
            try:
                self._db.leads.update_one(
                    {"$or": [{"lead_id": str(lead_id)}, {"wa_id": str(lead_id)}]},
                    {"$set": updates}
                )
                updated = self._db.leads.find_one({"$or": [{"lead_id": str(lead_id)}, {"wa_id": str(lead_id)}]})
                if updated:
                    self._sync_cache(updated)
                    return updated
            except Exception:
                pass
        target_id = self._wa_to_id.get(str(lead_id), str(lead_id))
        if target_id in self._cache:
            self._cache[target_id].update(updates)
            return self._cache[target_id]
        return None

    def update_by_wa_id(self, wa_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not wa_id or not updates:
            return None
        updates["updated_at"] = datetime.now(timezone.utc)
        updates["last_interaction_at"] = datetime.now(timezone.utc)
        if self._db.is_connected():
            try:
                self._db.leads.update_one({"wa_id": str(wa_id)}, {"$set": updates})
                updated = self._db.leads.find_one({"wa_id": str(wa_id)})
                if updated:
                    self._sync_cache(updated)
                    return updated
            except Exception:
                pass
        cached_id = self._wa_to_id.get(str(wa_id))
        if cached_id and cached_id in self._cache:
            self._cache[cached_id].update(updates)
            return self._cache[cached_id]
        return None

    def update_score(self, lead_id: str, score: int, stage: Optional[str] = None) -> Optional[Dict[str, Any]]:
        updates: Dict[str, Any] = {"lead_score": score}
        if stage:
            updates["sales_stage"] = stage
        return self.update(lead_id, updates)

    def shortlist_property(self, wa_id: str, property_id: str) -> bool:
        if not wa_id or not property_id:
            return False
        if self._db.is_connected():
            try:
                self._db.leads.update_one(
                    {"wa_id": str(wa_id)},
                    {"$addToSet": {"shortlisted_properties": property_id}, "$set": {"updated_at": datetime.now(timezone.utc)}}
                )
                lead = self.get_by_wa_id(wa_id)
                if lead:
                    self._sync_cache(lead)
                return True
            except Exception:
                pass
        lead = self.get_by_wa_id(wa_id)
        if lead:
            shortlist = lead.setdefault("shortlisted_properties", [])
            if property_id not in shortlist:
                shortlist.append(property_id)
        return True

    def list_leads(
        self,
        filter_query: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        skip: int = 0,
        sort_by: str = "updated_at",
        sort_desc: bool = True
    ) -> List[Dict[str, Any]]:
        query = filter_query or {}
        direction = -1 if sort_desc else 1
        if self._db.is_connected():
            try:
                cursor = self._db.leads.find(query).sort(sort_by, direction).skip(skip).limit(limit)
                leads = list(cursor)
                for l in leads:
                    self._sync_cache(l)
                return leads
            except Exception:
                pass
        all_cached = list(self._cache.values())
        filtered = [l for l in all_cached if all(l.get(k) == v for k, v in query.items())]
        return filtered[skip: skip + limit]

    def count_leads(self, filter_query: Optional[Dict[str, Any]] = None) -> int:
        query = filter_query or {}
        if self._db.is_connected():
            try:
                return self._db.leads.count_documents(query)
            except Exception:
                pass
        return len(self._cache)

    def get_stage_distribution(self) -> Dict[str, int]:
        if self._db.is_connected():
            try:
                pipeline = [{"$group": {"_id": "$sales_stage", "count": {"$sum": 1}}}]
                results = list(self._db.leads.aggregate(pipeline))
                return {r["_id"]: r["count"] for r in results if r.get("_id")}
            except Exception:
                pass
        dist: Dict[str, int] = {}
        for l in self._cache.values():
            stage = l.get("sales_stage", "NEW")
            dist[stage] = dist.get(stage, 0) + 1
        return dist

    def bulk_opt_out(self, lead_ids: List[str]) -> int:
        """Mark multiple leads as opted out from marketing/whatsapp campaigns."""
        if not lead_ids:
            return 0
        now = datetime.now(timezone.utc)
        updates = {
            "opted_out": True,
            "marketing_opt_in": False,
            "opt_out_date": now,
            "updated_at": now
        }
        modified_count = 0
        if self._db.is_connected():
            try:
                res = self._db.leads.update_many(
                    {"lead_id": {"$in": [str(lid) for lid in lead_ids]}},
                    {"$set": updates}
                )
                modified_count = res.modified_count
            except Exception:
                pass
        for lid in lead_ids:
            s_lid = str(lid)
            if s_lid in self._cache:
                self._cache[s_lid].update(updates)
                modified_count = max(modified_count, 1)
        return modified_count

    def opt_out(self, identifier: str) -> bool:
        """Mark a lead as opted out by wa_id or lead_id."""
        if not identifier:
            return False
        now = datetime.now(timezone.utc)
        updates = {
            "opted_out": True,
            "marketing_opt_in": False,
            "sales_stage": "OPTED_OUT",
            "opt_out_date": now,
            "updated_at": now
        }
        if self._db.is_connected():
            try:
                self._db.leads.update_many(
                    {"$or": [{"wa_id": str(identifier)}, {"lead_id": str(identifier)}]},
                    {"$set": updates}
                )
            except Exception:
                pass
        for l in self._cache.values():
            if l.get("wa_id") == identifier or l.get("lead_id") == identifier:
                l.update(updates)
        return True

    def opt_in(self, identifier: str) -> bool:
        """Re-enable marketing and messaging for a lead."""
        if not identifier:
            return False
        now = datetime.now(timezone.utc)
        updates = {
            "opted_out": False,
            "marketing_opt_in": True,
            "whatsapp_opt_in": True,
            "sales_stage": "NEW",
            "updated_at": now
        }
        if self._db.is_connected():
            try:
                self._db.leads.update_many(
                    {"$or": [{"wa_id": str(identifier)}, {"lead_id": str(identifier)}]},
                    {"$set": updates}
                )
            except Exception:
                pass
        for l in self._cache.values():
            if l.get("wa_id") == identifier or l.get("lead_id") == identifier:
                l.update(updates)
        return True

    def get_consent_stats(self) -> Dict[str, int]:
        """Returns counts for opted_in, opted_out, and total leads."""
        if self._db.is_connected():
            try:
                total = self._db.leads.count_documents({})
                opted_out = self._db.leads.count_documents({"opted_out": True})
                marketing_in = self._db.leads.count_documents({"marketing_opt_in": True, "opted_out": {"$ne": True}})
                whatsapp_in = self._db.leads.count_documents({"whatsapp_opt_in": True, "opted_out": {"$ne": True}})
                return {
                    "total_leads": total,
                    "opted_in": marketing_in,
                    "whatsapp_opt_in": whatsapp_in,
                    "opted_out": opted_out,
                    "unspecified": max(0, total - (opted_out + marketing_in))
                }
            except Exception:
                pass
        total = len(self._cache)
        opted_out = sum(1 for l in self._cache.values() if l.get("opted_out") is True)
        m_in = sum(1 for l in self._cache.values() if l.get("marketing_opt_in") is True and not l.get("opted_out"))
        w_in = sum(1 for l in self._cache.values() if l.get("whatsapp_opt_in") is True and not l.get("opted_out"))
        return {
            "total_leads": total,
            "opted_in": m_in,
            "whatsapp_opt_in": w_in,
            "opted_out": opted_out,
            "unspecified": max(0, total - (opted_out + m_in))
        }

    def search_leads_for_campaign(
        self,
        query: str = "",
        city: str = "",
        stage: str = "",
        bhk: str = "",
        min_score: int = 0,
        limit: int = 500
    ) -> List[Dict[str, Any]]:
        """Search and filter eligible leads for targeted bulk campaigns."""
        filter_dict: Dict[str, Any] = {
            "opted_out": {"$ne": True}
        }
        if city:
            filter_dict["preferred_city"] = {"$regex": f"^{city}$", "$options": "i"}
        if stage:
            filter_dict["sales_stage"] = stage
        if bhk:
            filter_dict["bhk"] = {"$regex": bhk, "$options": "i"}
        if min_score > 0:
            filter_dict["lead_score"] = {"$gte": int(min_score)}

        if query:
            q_clean = query.strip()
            filter_dict["$or"] = [
                {"name": {"$regex": q_clean, "$options": "i"}},
                {"phone": {"$regex": q_clean, "$options": "i"}},
                {"wa_id": {"$regex": q_clean, "$options": "i"}},
            ]

        if self._db.is_connected():
            try:
                cursor = self._db.leads.find(filter_dict).sort("lead_score", -1).limit(limit)
                return list(cursor)
            except Exception:
                pass

        results = []
        for l in self._cache.values():
            if l.get("opted_out") is True:
                continue
            if city and l.get("preferred_city", "").lower() != city.lower():
                continue
            if stage and l.get("sales_stage") != stage:
                continue
            if bhk and bhk.lower() not in l.get("bhk", "").lower():
                continue
            if min_score > 0 and int(l.get("lead_score", 0)) < min_score:
                continue
            if query:
                qc = query.lower()
                if (qc not in str(l.get("name", "")).lower() and
                    qc not in str(l.get("phone", "")) and
                    qc not in str(l.get("wa_id", ""))):
                    continue
            results.append(l)
        results.sort(key=lambda x: int(x.get("lead_score", 0)), reverse=True)
        return results[:limit]

    def _sync_cache(self, doc: Dict[str, Any]):
        if not doc:
            return
        lid = str(doc.get("lead_id", ""))
        wa_id = str(doc.get("wa_id", ""))
        if lid:
            self._cache[lid] = doc
        if wa_id and lid:
            self._wa_to_id[wa_id] = lid


INITIAL_SEEDED_PROPERTIES: List[Dict[str, Any]] = [
    {
        "id": "ARIS-NGP-01",
        "title": "Green Meadows Residency",
        "project_name": "Green Meadows",
        "builder_name": "ARIS Premier Builders",
        "city": "Nagpur",
        "locality": "Manish Nagar",
        "address": "Plot 42, Near Manish Nagar Metro, Nagpur, Maharashtra 440015",
        "latitude": 21.0965,
        "longitude": 79.0825,
        "type": "Apartment",
        "bhk": "2BHK",
        "bathrooms": 2,
        "balconies": 1,
        "carpet_area": "980 sq.ft",
        "built_up_area": "1,180 sq.ft",
        "price_lakhs": 48.5,
        "price_display": "₹48.5 Lakhs",
        "price_per_sqft": 4110.0,
        "possession_status": "Ready to Move",
        "possession_date": "Immediate",
        "amenities": ["Covered Parking", "Lift", "Power Backup", "24x7 Security", "Intercom"],
        "parking": "Covered Stilt",
        "furnishing": "Semi-Furnished",
        "floor": "3 of 7",
        "nearby_landmarks": ["Manish Nagar Metro (300m)", "Airport (3.5km)", "Purti Supermarket"],
        "images": ["https://images.unsplash.com/photo-1545324418-cc1a3fa10c00?auto=format&fit=crop&w=800&q=80"],
        "brochure_url": "https://raw.githubusercontent.com/aris-realestate/assets/main/brochures/green_meadows.pdf",
        "location_url": "https://maps.google.com/?q=21.0965,79.0825",
        "description": "Spacious 2BHK east-facing apartment with modern kitchen and modular fittings, close to metro station.",
        "highlights": ["300m from Metro", "Vastu Compliant", "Zero Brokerage"],
        "inventory_count": 4,
        "availability_status": "Available"
    },
    {
        "id": "ARIS-NGP-02",
        "title": "Royal Palms Heights",
        "project_name": "Royal Palms",
        "builder_name": "Skyline Infrastructure",
        "city": "Nagpur",
        "locality": "Besa - Ghogli Road",
        "address": "Survey 108, Besa-Ghogli Road, Besa, Nagpur 440037",
        "latitude": 21.0821,
        "longitude": 79.1012,
        "type": "Apartment",
        "bhk": "3BHK",
        "bathrooms": 3,
        "balconies": 2,
        "carpet_area": "1,350 sq.ft",
        "built_up_area": "1,620 sq.ft",
        "price_lakhs": 68.0,
        "price_display": "₹68.0 Lakhs",
        "price_per_sqft": 4197.0,
        "possession_status": "Under Construction",
        "possession_date": "Dec 2026",
        "amenities": ["Clubhouse", "Swimming Pool", "Gym", "Kids Play Area", "EV Charging", "Jogging Track"],
        "parking": "Covered Basement",
        "furnishing": "Unfurnished",
        "floor": "7 of 14",
        "nearby_landmarks": ["Podar International School", "Revati Nagar", "D-Mart Besa"],
        "images": ["https://images.unsplash.com/photo-1512917774080-9991f1c4c750?auto=format&fit=crop&w=800&q=80"],
        "brochure_url": "https://raw.githubusercontent.com/aris-realestate/assets/main/brochures/royal_palms.pdf",
        "location_url": "https://maps.google.com/?q=21.0821,79.1012",
        "description": "Premium 3BHK high-rise apartment with panoramic skyline views and full lifestyle club amenities.",
        "highlights": ["Olympic Size Pool", "RERA Approved", "High Appreciation Corridor"],
        "inventory_count": 7,
        "availability_status": "Available"
    },
    {
        "id": "ARIS-NGP-03",
        "title": "Heritage Serene Enclave",
        "project_name": "Heritage Serene",
        "builder_name": "Heritage Homes Nagpur",
        "city": "Nagpur",
        "locality": "Wardha Road",
        "address": "Near MIHAN Flyover, Wardha Road, Nagpur 441108",
        "latitude": 21.0234,
        "longitude": 79.0512,
        "type": "Villa",
        "bhk": "4BHK",
        "bathrooms": 4,
        "balconies": 3,
        "carpet_area": "2,400 sq.ft",
        "built_up_area": "2,950 sq.ft",
        "price_lakhs": 125.0,
        "price_display": "₹1.25 Cr",
        "price_per_sqft": 4237.0,
        "possession_status": "Ready to Move",
        "possession_date": "Immediate",
        "amenities": ["Private Garden", "Dual Car Parking", "Gated Community", "Solar Water Heating", "Clubhouse"],
        "parking": "2 Covered Dedicated",
        "furnishing": "Semi-Furnished",
        "floor": "G+1 Independent",
        "nearby_landmarks": ["AIIMS Nagpur (2km)", "IIM Nagpur", "Infosys MIHAN Campus"],
        "images": ["https://images.unsplash.com/photo-1613977257363-707ba9348227?auto=format&fit=crop&w=800&q=80"],
        "brochure_url": "https://raw.githubusercontent.com/aris-realestate/assets/main/brochures/heritage_serene.pdf",
        "location_url": "https://maps.google.com/?q=21.0234,79.0512",
        "description": "Luxurious 4BHK duplex villa inside an exclusive gated community right next to MIHAN SEZ corridor.",
        "highlights": ["Near AIIMS & IIM", "Independent Villa", "Private Garden"],
        "inventory_count": 2,
        "availability_status": "Few Units Left"
    },
    {
        "id": "ARIS-NGP-04",
        "title": "Metro Smart Homes",
        "project_name": "Metro Smart",
        "builder_name": "Central India Developers",
        "city": "Nagpur",
        "locality": "Dharampeth",
        "address": "West High Court Road, Dharampeth, Nagpur 440010",
        "latitude": 21.1425,
        "longitude": 79.0658,
        "type": "Apartment",
        "bhk": "1BHK",
        "bathrooms": 1,
        "balconies": 1,
        "carpet_area": "620 sq.ft",
        "built_up_area": "750 sq.ft",
        "price_lakhs": 32.0,
        "price_display": "₹32.0 Lakhs",
        "price_per_sqft": 4266.0,
        "possession_status": "Ready to Move",
        "possession_date": "Immediate",
        "amenities": ["Lift", "Security Guard", "Reserved Bike/Car Parking", "CCTV"],
        "parking": "Covered 2-Wheeler + Open 4-Wheeler",
        "furnishing": "Unfurnished",
        "floor": "2 of 5",
        "nearby_landmarks": ["Coffee House Dharampeth", "Laxmi Nagar", "Traffic Park"],
        "images": ["https://images.unsplash.com/photo-1560448204-e02f11c3d0e2?auto=format&fit=crop&w=800&q=80"],
        "brochure_url": "https://raw.githubusercontent.com/aris-realestate/assets/main/brochures/metro_smart.pdf",
        "location_url": "https://maps.google.com/?q=21.1425,79.0658",
        "description": "Compact and efficient 1BHK in prime central city location with easy access to markets & schools.",
        "highlights": ["Heart of Nagpur", "Low Maintenance", "High Rental Yield"],
        "inventory_count": 3,
        "availability_status": "Available"
    },
    {
        "id": "ARIS-PUN-01",
        "title": "Elysium Tech Park View",
        "project_name": "Elysium Residences",
        "builder_name": "Elysium Group",
        "city": "Pune",
        "locality": "Hinjewadi Phase 1",
        "address": "Phase 1, Hinjewadi Rajiv Gandhi Infotech Park, Pune 411057",
        "latitude": 18.5912,
        "longitude": 73.7389,
        "type": "Apartment",
        "bhk": "2BHK",
        "bathrooms": 2,
        "balconies": 2,
        "carpet_area": "890 sq.ft",
        "built_up_area": "1,080 sq.ft",
        "price_lakhs": 72.0,
        "price_display": "₹72.0 Lakhs",
        "price_per_sqft": 6666.0,
        "possession_status": "Ready to Move",
        "possession_date": "Immediate",
        "amenities": ["Gym", "Infinity Pool", "Clubhouse", "Work-from-Home Pods", "Cafeteria"],
        "parking": "1 Covered Basement",
        "furnishing": "Semi-Furnished",
        "floor": "11 of 18",
        "nearby_landmarks": ["Wipro Circle", "Infosys Campus", "Metro Line 3 Station"],
        "images": ["https://images.unsplash.com/photo-1522708323590-d24dbb6b0267?auto=format&fit=crop&w=800&q=80"],
        "brochure_url": "https://raw.githubusercontent.com/aris-realestate/assets/main/brochures/elysium_pune.pdf",
        "location_url": "https://maps.google.com/?q=18.5912,73.7389",
        "description": "Modern urban 2BHK 5 minutes away from IT companies and metro line.",
        "highlights": ["Walking Distance to Wipro", "Co-Working Lounge", "Smart Home Automation"],
        "inventory_count": 5,
        "availability_status": "Available"
    },
    {
        "id": "ARIS-PUN-02",
        "title": "Vanguard Valley Residences",
        "project_name": "Vanguard Valley",
        "builder_name": "Vanguard Spaces",
        "city": "Pune",
        "locality": "Wakad",
        "address": "Dutta Mandir Road, Wakad, Pimpri-Chinchwad, Pune 411057",
        "latitude": 18.6011,
        "longitude": 73.7645,
        "type": "Apartment",
        "bhk": "3BHK",
        "bathrooms": 3,
        "balconies": 2,
        "carpet_area": "1,220 sq.ft",
        "built_up_area": "1,490 sq.ft",
        "price_lakhs": 98.0,
        "price_display": "₹98.0 Lakhs",
        "price_per_sqft": 6577.0,
        "possession_status": "Under Construction",
        "possession_date": "March 2027",
        "amenities": ["Landscape Gardens", "Badminton Court", "EV Charging", "Banquet Hall", "Jogging Track"],
        "parking": "2 Covered Basement",
        "furnishing": "Unfurnished",
        "floor": "9 of 21",
        "nearby_landmarks": ["Phoenix Marketcity Wakad", "Sayaji Hotel", "Mumbai-Pune Expressway"],
        "images": ["https://images.unsplash.com/photo-1502672260266-1c1ef2d93688?auto=format&fit=crop&w=800&q=80"],
        "brochure_url": "https://raw.githubusercontent.com/aris-realestate/assets/main/brochures/vanguard_valley.pdf",
        "location_url": "https://maps.google.com/?q=18.6011,73.7645",
        "description": "Large-format 3BHK flat with double balconies in a family-oriented society.",
        "highlights": ["Near Phoenix Mall", "Double Balconies", "Large Clubhouse"],
        "inventory_count": 6,
        "availability_status": "Available"
    }
]


class PropertyRepository:
    """
    Data Access Repository for Property Catalog.
    Synchronizes initial listings into database on initialization.
    """
    _cache: Dict[str, Dict[str, Any]] = {}
    _seeded: bool = False

    def __init__(self):
        self._db_manager = get_db()
        self._seed_if_needed()

    def _seed_if_needed(self):
        if PropertyRepository._seeded:
            return

        for p in INITIAL_SEEDED_PROPERTIES:
            prop_obj = Property.from_dict(p)
            doc = prop_obj.to_dict()
            PropertyRepository._cache[doc["id"]] = doc

            try:
                self._db_manager.properties.update_one(
                    {"id": doc["id"]},
                    {"$set": doc},
                    upsert=True
                )
            except Exception:
                pass

        PropertyRepository._seeded = True

    def get_by_id(self, prop_id: str) -> Optional[Dict[str, Any]]:
        if not prop_id:
            return None
        clean_id = str(prop_id).strip().upper()

        try:
            doc = self._db_manager.properties.find_one({"id": clean_id})
            if doc:
                PropertyRepository._cache[clean_id] = doc
                return doc
        except Exception:
            pass

        if clean_id in PropertyRepository._cache:
            return PropertyRepository._cache[clean_id]

        import re
        digits = re.findall(r"\d+", clean_id)
        if digits:
            idx = int(digits[0]) - 1
            all_props = list(PropertyRepository._cache.values())
            if 0 <= idx < len(all_props):
                return all_props[idx]

        return None

    def list_properties(
        self,
        city: Optional[str] = None,
        locality: Optional[str] = None,
        bhk: Optional[str] = None,
        max_budget_lakhs: Optional[float] = None,
        min_budget_lakhs: Optional[float] = None,
        prop_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        skip: int = 0
    ) -> List[Dict[str, Any]]:
        import re
        query: Dict[str, Any] = {}
        if city:
            query["city"] = {"$regex": re.escape(city.strip()), "$options": "i"}
        if locality:
            query["locality"] = {"$regex": re.escape(locality.strip()), "$options": "i"}
        if bhk:
            clean_bhk = bhk.upper().replace(" ", "")
            query["bhk"] = {"$regex": re.escape(clean_bhk), "$options": "i"}
        if prop_type:
            query["type"] = {"$regex": re.escape(prop_type.strip()), "$options": "i"}
        if status:
            query["possession_status"] = {"$regex": re.escape(status.strip()), "$options": "i"}

        price_filter: Dict[str, Any] = {}
        if max_budget_lakhs is not None:
            price_filter["$lte"] = float(max_budget_lakhs)
        if min_budget_lakhs is not None:
            price_filter["$gte"] = float(min_budget_lakhs)
        if price_filter:
            query["price_lakhs"] = price_filter

        try:
            docs = list(self._db_manager.properties.find(query).skip(skip).limit(limit))
            if docs:
                for d in docs:
                    PropertyRepository._cache[d["id"]] = d
                return docs
        except Exception:
            pass

        results = []
        for prop in PropertyRepository._cache.values():
            if city and city.lower() not in prop.get("city", "").lower():
                continue
            if locality and locality.lower() not in prop.get("locality", "").lower():
                continue
            if bhk and bhk.lower().replace(" ", "") not in prop.get("bhk", "").lower().replace(" ", ""):
                continue
            if prop_type and prop_type.lower() not in prop.get("type", "").lower():
                continue
            if max_budget_lakhs is not None and prop.get("price_lakhs", 0) > max_budget_lakhs:
                continue
            if min_budget_lakhs is not None and prop.get("price_lakhs", 0) < min_budget_lakhs:
                continue
            if status and status.lower() not in prop.get("possession_status", "").lower():
                continue
            results.append(prop)

        return results[skip: skip + limit]

    def count_properties(self, filter_query: Optional[Dict[str, Any]] = None) -> int:
        query = filter_query or {}
        try:
            return self._db_manager.properties.count_documents(query)
        except Exception:
            return len(PropertyRepository._cache)

    def create_property(self, data: Dict[str, Any]) -> Dict[str, Any]:
        prop_obj = Property.from_dict(data)
        doc = prop_obj.to_dict()
        try:
            self._db_manager.properties.insert_one(doc)
        except Exception:
            pass
        PropertyRepository._cache[doc["id"]] = doc
        return doc

    def update_property(self, prop_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not prop_id or not updates:
            return None
        clean_id = str(prop_id).strip().upper()
        updates["updated_at"] = datetime.now(timezone.utc)
        try:
            self._db_manager.properties.update_one({"id": clean_id}, {"$set": updates})
            doc = self._db_manager.properties.find_one({"id": clean_id})
            if doc:
                PropertyRepository._cache[clean_id] = doc
                return doc
        except Exception:
            pass
        cached = PropertyRepository._cache.get(clean_id)
        if cached:
            cached.update(updates)
            return cached
        return None

    def delete_property(self, prop_id: str) -> bool:
        if not prop_id:
            return False
        clean_id = str(prop_id).strip().upper()
        try:
            self._db_manager.properties.delete_one({"id": clean_id})
        except Exception:
            pass
        PropertyRepository._cache.pop(clean_id, None)
        return True

    def search(self, city: Optional[str] = None, bhk: Optional[str] = None, max_price: Optional[float] = None, limit: int = 10) -> List[Dict[str, Any]]:
        return self.list_properties(city=city, bhk=bhk, max_budget_lakhs=max_price, limit=limit)

    def list_all(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self.list_properties(limit=limit)



class VisitRepository:
    """
    Data Access Repository for Property Site Visits.
    Supports idempotency keys to prevent duplicate booking.
    """
    _cache_by_id: Dict[str, Dict[str, Any]] = {}
    _cache_by_key: Dict[str, Dict[str, Any]] = {}

    def __init__(self):
        self._db = get_db()
        self._db_manager = self._db

    def create_visit(self, visit_data: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        doc_data = dict(visit_data) if isinstance(visit_data, dict) else {}
        doc_data.update(kwargs)
        if not doc_data:
            raise ValueError("visit_data or parameters must be provided")

        key = doc_data.get("idempotency_key")
        if key:
            existing = self.get_by_idempotency(key)
            if existing:
                return existing

        visit_id = doc_data.get("visit_id") or f"visit_{uuid.uuid4().hex[:12]}"
        doc = dict(doc_data)
        doc["visit_id"] = visit_id
        doc.setdefault("created_at", datetime.now(timezone.utc))

        if self._db.is_connected():
            try:
                self._db.visits.insert_one(doc)
            except Exception as ex:
                print(f"[VISIT_REPO ERROR] Failed to insert site visit: {ex}")

        self._sync_cache(doc)
        return doc

    def book_visit(
        self,
        lead_id: str,
        wa_id: str,
        property_id: str,
        visit_date: str,
        visit_time: str,
        client_name: str = "",
        client_phone: str = "",
        cab_required: bool = True,
        pickup_address: Optional[str] = None
    ) -> Dict[str, Any]:
        return self.create_visit({
            "lead_id": lead_id,
            "wa_id": wa_id,
            "property_id": property_id,
            "visit_date": visit_date,
            "visit_time": visit_time,
            "customer_name": client_name,
            "client_name": client_name,
            "client_phone": client_phone or wa_id,
            "status": "scheduled",
            "cab_required": cab_required,
            "pickup_address": pickup_address or "",
            "cab_status": "assigned" if cab_required else "not_required",
            "idempotency_key": f"{lead_id}_{property_id}_{visit_date}_{visit_time}"
        })

    def get_by_id(self, visit_id: str) -> Optional[Dict[str, Any]]:
        if not visit_id:
            return None
        if self._db.is_connected():
            try:
                doc = self._db.visits.find_one({"visit_id": str(visit_id)})
                if doc:
                    self._sync_cache(doc)
                    return doc
            except Exception:
                pass
        return self._cache_by_id.get(str(visit_id))

    def get_by_idempotency(self, idempotency_key: str) -> Optional[Dict[str, Any]]:
        if not idempotency_key:
            return None
        if self._db.is_connected():
            try:
                doc = self._db.visits.find_one({"idempotency_key": str(idempotency_key)})
                if doc:
                    self._sync_cache(doc)
                    return doc
            except Exception:
                pass
        return self._cache_by_key.get(str(idempotency_key))

    def reschedule_visit(self, visit_id: str, new_date: str, new_time: str, notes: str = "") -> Optional[Dict[str, Any]]:
        if not visit_id:
            return None
        updates = {
            "visit_date": new_date,
            "visit_time": new_time,
            "status": "rescheduled",
            "updated_at": datetime.now(timezone.utc)
        }
        if notes:
            updates["notes"] = notes

        if self._db.is_connected():
            try:
                self._db.visits.update_one({"visit_id": str(visit_id)}, {"$set": updates})
                doc = self._db.visits.find_one({"visit_id": str(visit_id)})
                if doc:
                    self._sync_cache(doc)
                    return doc
            except Exception:
                pass

        cached = self._cache_by_id.get(str(visit_id))
        if cached:
            cached.update(updates)
            return cached
        return None

    def cancel_visit(self, visit_id: str, reason: str = "") -> Optional[Dict[str, Any]]:
        if not visit_id:
            return None
        updates = {"status": "cancelled", "updated_at": datetime.now(timezone.utc)}
        if reason:
            updates["notes"] = reason

        if self._db.is_connected():
            try:
                self._db.visits.update_one({"visit_id": str(visit_id)}, {"$set": updates})
                doc = self._db.visits.find_one({"visit_id": str(visit_id)})
                if doc:
                    self._sync_cache(doc)
                    return doc
            except Exception:
                pass

        cached = self._cache_by_id.get(str(visit_id))
        if cached:
            cached.update(updates)
            return cached
        return None

    def list_visits(self, filter_query: Optional[Dict[str, Any]] = None, limit: int = 50, skip: int = 0, sort_by: str = "created_at", sort_desc: bool = True) -> List[Dict[str, Any]]:
        query = filter_query or {}
        direction = -1 if sort_desc else 1
        if self._db.is_connected():
            try:
                cursor = self._db.visits.find(query).sort(sort_by, direction).skip(skip).limit(limit)
                docs = list(cursor)
                for d in docs:
                    self._sync_cache(d)
                return docs
            except Exception:
                pass

        all_cached = list(self._cache_by_id.values())
        filtered = [v for v in all_cached if all(v.get(k) == val for k, val in query.items())]
        return filtered[skip: skip + limit]

    def count_visits(self, filter_query: Optional[Dict[str, Any]] = None) -> int:
        query = filter_query or {}
        if self._db.is_connected():
            try:
                return self._db.visits.count_documents(query)
            except Exception:
                pass
        return len(self._cache_by_id)

    def _sync_cache(self, doc: Dict[str, Any]):
        if not doc:
            return
        vid = str(doc.get("visit_id", ""))
        key = str(doc.get("idempotency_key", ""))
        if vid:
            self._cache_by_id[vid] = doc
        if key:
            self._cache_by_key[key] = doc


class FollowUpRepository:
    """
    Data Access Repository for Automated & Scheduled Follow-ups.
    """
    _cache_by_id: Dict[str, Dict[str, Any]] = {}

    def __init__(self):
        self._db = get_db()
        self._db_manager = self._db

    def create_followup(self, followup_data: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(followup_data, dict):
            raise ValueError("followup_data must be a dictionary")

        doc = dict(followup_data)
        doc.setdefault("followup_id", f"fol_{uuid.uuid4().hex[:10]}")
        doc.setdefault("created_at", datetime.now(timezone.utc))

        if self._db.is_connected():
            try:
                self._db.followups.insert_one(doc)
            except Exception as ex:
                print(f"[FOLLOWUP_REPO ERROR] Failed to create follow-up: {ex}")

        self._sync_cache(doc)
        return doc

    def schedule(self, lead_id: str, scheduled_for: datetime, message_template: str) -> Dict[str, Any]:
        return self.create_followup({
            "lead_id": lead_id,
            "scheduled_at": scheduled_for,
            "scheduled_for": scheduled_for,
            "message_template": message_template,
            "status": "pending"
        })

    def get_by_id(self, followup_id: str) -> Optional[Dict[str, Any]]:
        if not followup_id:
            return None
        if self._db.is_connected():
            try:
                doc = self._db.followups.find_one({"followup_id": str(followup_id)})
                if doc:
                    self._sync_cache(doc)
                    return doc
            except Exception:
                pass
        return self._cache_by_id.get(str(followup_id))

    def list_pending_due(self, before_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        now = before_time or datetime.now(timezone.utc)
        if self._db.is_connected():
            try:
                docs = list(self._db.followups.find({
                    "status": "pending",
                    "scheduled_at": {"$lte": now}
                }).sort("scheduled_at", 1))
                for d in docs:
                    self._sync_cache(d)
                return docs
            except Exception:
                pass
        due = []
        for f in self._cache_by_id.values():
            if f.get("status") == "pending":
                sched = f.get("scheduled_at") or f.get("scheduled_for")
                if sched and sched <= now:
                    due.append(f)
        return sorted(due, key=lambda x: x.get("scheduled_at", now))

    def mark_as_sent(self, followup_id: str) -> Optional[Dict[str, Any]]:
        if not followup_id:
            return None
        now = datetime.now(timezone.utc)
        updates = {"status": "sent", "sent_at": now}
        if self._db.is_connected():
            try:
                self._db.followups.update_one({"followup_id": str(followup_id)}, {"$set": updates})
                doc = self._db.followups.find_one({"followup_id": str(followup_id)})
                if doc:
                    self._sync_cache(doc)
                    return doc
            except Exception:
                pass
        cached = self._cache_by_id.get(str(followup_id))
        if cached:
            cached.update(updates)
            return cached
        return None

    def list_followups(self, filter_query: Optional[Dict[str, Any]] = None, limit: int = 50, skip: int = 0, sort_by: str = "scheduled_at", sort_desc: bool = True) -> List[Dict[str, Any]]:
        query = filter_query or {}
        direction = -1 if sort_desc else 1
        if self._db.is_connected():
            try:
                cursor = self._db.followups.find(query).sort(sort_by, direction).skip(skip).limit(limit)
                docs = list(cursor)
                for d in docs:
                    self._sync_cache(d)
                return docs
            except Exception:
                pass
        all_cached = list(self._cache_by_id.values())
        filtered = [f for f in all_cached if all(f.get(k) == val for k, val in query.items())]
        return filtered[skip: skip + limit]

    def count_followups(self, filter_query: Optional[Dict[str, Any]] = None) -> int:
        query = filter_query or {}
        if self._db.is_connected():
            try:
                return self._db.followups.count_documents(query)
            except Exception:
                pass
        return len(self._cache_by_id)

    def _sync_cache(self, doc: Dict[str, Any]):
        if not doc:
            return
        fid = str(doc.get("followup_id", ""))
        if fid:
            self._cache_by_id[fid] = doc


class OutboundMessageRepository:
    _cache: Dict[str, Dict[str, Any]] = {}

    def __init__(self):
        self._db = get_db()

    def save(self, message: OutboundMessage) -> Dict[str, Any]:
        doc = message.to_dict()
        self._cache[message.outbound_id] = doc
        if self._db.is_connected():
            try:
                self._db.whatsapp_outbound_messages.insert_one(doc)
            except Exception:
                pass
        return doc

    def update_status(self, outbound_id: str, status: str, meta_message_id: str = "", error_code: str = "", error_message: str = "") -> bool:
        now = datetime.now(timezone.utc)
        updates: Dict[str, Any] = {"status": status}
        if meta_message_id:
            updates["meta_message_id"] = meta_message_id
        if status == OutboundStatus.ACCEPTED:
            updates["accepted_at"] = now
        elif status == OutboundStatus.SENT:
            updates["sent_at"] = now
        elif status == OutboundStatus.FAILED:
            updates["failed_at"] = now
            if error_code:
                updates["error_code"] = error_code
            if error_message:
                updates["error_message"] = error_message

        if outbound_id in self._cache:
            self._cache[outbound_id].update(updates)
        if self._db.is_connected():
            try:
                self._db.whatsapp_outbound_messages.update_one({"outbound_id": outbound_id}, {"$set": updates})
                return True
            except Exception:
                pass
        return False

    def update_delivery_event(self, meta_message_id: str, event_type: str, timestamp: Optional[datetime] = None, error_code: str = "", error_message: str = "") -> bool:
        ts = timestamp or datetime.now(timezone.utc)
        status_map = {
            "sent": OutboundStatus.SENT,
            "delivered": OutboundStatus.DELIVERED,
            "read": OutboundStatus.READ,
            "failed": OutboundStatus.FAILED,
        }
        status = status_map.get(event_type.lower())
        if not status:
            return False

        updates: Dict[str, Any] = {"status": status}
        if event_type == "sent":
            updates["sent_at"] = ts
        elif event_type == "delivered":
            updates["delivered_at"] = ts
        elif event_type == "read":
            updates["read_at"] = ts
        elif event_type == "failed":
            updates["failed_at"] = ts
            if error_code:
                updates["error_code"] = str(error_code)

        for doc in self._cache.values():
            if doc.get("meta_message_id") == meta_message_id:
                doc.update(updates)

        if self._db.is_connected():
            try:
                res = self._db.whatsapp_outbound_messages.update_one({"meta_message_id": meta_message_id}, {"$set": updates})
                return res.modified_count > 0
            except Exception:
                pass
        return False

    def get_by_id(self, outbound_id: str) -> Optional[Dict[str, Any]]:
        if self._db.is_connected():
            try:
                return self._db.whatsapp_outbound_messages.find_one({"outbound_id": outbound_id})
            except Exception:
                pass
        return self._cache.get(outbound_id)

    def get_by_meta_message_id(self, meta_message_id: str) -> Optional[Dict[str, Any]]:
        if self._db.is_connected():
            try:
                return self._db.whatsapp_outbound_messages.find_one({"meta_message_id": meta_message_id})
            except Exception:
                pass
        for doc in self._cache.values():
            if doc.get("meta_message_id") == meta_message_id:
                return doc
        return None

    def check_idempotency(self, idempotency_key: str) -> Optional[Dict[str, Any]]:
        if not idempotency_key:
            return None
        if self._db.is_connected():
            try:
                doc = self._db.whatsapp_outbound_messages.find_one({"idempotency_key": idempotency_key})
                if doc:
                    return doc
            except Exception:
                pass
        for doc in self._cache.values():
            if doc.get("idempotency_key") == idempotency_key:
                return doc
        return None

    def list_messages(self, filters: Optional[Dict[str, Any]] = None, limit: int = 50, skip: int = 0, sort_by: str = "created_at", sort_dir: int = DESCENDING, status: Optional[str] = None, phone: Optional[str] = None, campaign_id: Optional[str] = None, template_name: Optional[str] = None, source: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        query = dict(filters or {})
        if status:
            query["status"] = status
        if phone:
            query["$or"] = [{"phone_number": {"$regex": phone, "$options": "i"}}, {"normalized_phone": {"$regex": phone, "$options": "i"}}]
        if campaign_id:
            query["campaign_id"] = campaign_id
        if template_name:
            query["template_name"] = {"$regex": template_name, "$options": "i"}
        if source:
            query["source"] = source
        for k, v in kwargs.items():
            if v is not None:
                query[k] = v

        if self._db.is_connected():
            try:
                cursor = self._db.whatsapp_outbound_messages.find(query).sort(sort_by, sort_dir).skip(skip).limit(limit)
                return list(cursor)
            except Exception:
                pass
        cached = list(self._cache.values())
        if source:
            cached = [m for m in cached if m.get("source") == source]
        if status:
            cached = [m for m in cached if m.get("status") == status]
        return cached[skip:skip+limit]

    def count_messages(self, filters: Optional[Dict[str, Any]] = None, status: Optional[str] = None, phone: Optional[str] = None, campaign_id: Optional[str] = None, template_name: Optional[str] = None, source: Optional[str] = None, **kwargs) -> int:
        query = dict(filters or {})
        if status:
            query["status"] = status
        if phone:
            query["$or"] = [{"phone_number": {"$regex": phone, "$options": "i"}}, {"normalized_phone": {"$regex": phone, "$options": "i"}}]
        if campaign_id:
            query["campaign_id"] = campaign_id
        if template_name:
            query["template_name"] = {"$regex": template_name, "$options": "i"}
        if source:
            query["source"] = source
        if self._db.is_connected():
            try:
                return self._db.whatsapp_outbound_messages.count_documents(query)
            except Exception:
                pass
        return len(self._cache)

    def get_analytics_for_campaign(self, campaign_id: str) -> Dict[str, Any]:
        match = {"campaign_id": campaign_id}
        if self._db.is_connected():
            try:
                pipeline = [{"$match": match}, {"$group": {"_id": "$status", "count": {"$sum": 1}}}]
                res = list(self._db.whatsapp_outbound_messages.aggregate(pipeline))
                counts = {item["_id"]: item["count"] for item in res}
                queued = counts.get(OutboundStatus.QUEUED, 0)
                accepted = counts.get(OutboundStatus.ACCEPTED, 0)
                sent_raw = counts.get(OutboundStatus.SENT, 0)
                delivered = counts.get(OutboundStatus.DELIVERED, 0)
                read = counts.get(OutboundStatus.READ, 0)
                failed = counts.get(OutboundStatus.FAILED, 0)

                total_delivered = delivered + read
                total_sent = sent_raw + accepted + total_delivered

                return {
                    "queued": queued,
                    "sent": total_sent,
                    "delivered": total_delivered,
                    "read": read,
                    "failed": failed,
                    "delivery_rate": round(total_delivered / total_sent * 100, 1) if total_sent > 0 else 0.0,
                    "read_rate": round(read / total_delivered * 100, 1) if total_delivered > 0 else 0.0,
                }
            except Exception:
                pass

        # In-memory fallback
        matching = [m for m in self._cache.values() if m.get("campaign_id") == campaign_id]
        queued = sum(1 for m in matching if m.get("status") == OutboundStatus.QUEUED)
        accepted = sum(1 for m in matching if m.get("status") == OutboundStatus.ACCEPTED)
        sent_raw = sum(1 for m in matching if m.get("status") == OutboundStatus.SENT)
        delivered = sum(1 for m in matching if m.get("status") == OutboundStatus.DELIVERED)
        read = sum(1 for m in matching if m.get("status") == OutboundStatus.READ)
        failed = sum(1 for m in matching if m.get("status") == OutboundStatus.FAILED)

        total_delivered = delivered + read
        total_sent = sent_raw + accepted + total_delivered

        return {
            "queued": queued,
            "sent": total_sent,
            "delivered": total_delivered,
            "read": read,
            "failed": failed,
            "delivery_rate": round(total_delivered / total_sent * 100, 1) if total_sent > 0 else 0.0,
            "read_rate": round(read / total_delivered * 100, 1) if total_delivered > 0 else 0.0,
        }


class CampaignRepository:
    _cache: Dict[str, Dict[str, Any]] = {}

    def __init__(self):
        self._db = get_db()

    def create_campaign(self, data: Dict[str, Any]) -> Dict[str, Any]:
        camp = Campaign.from_dict(data)
        doc = camp.to_dict()
        if self._db.is_connected():
            try:
                self._db.campaigns.insert_one(doc)
            except Exception:
                pass
        self._cache[camp.campaign_id] = doc
        return doc

    def get_by_id(self, campaign_id: str) -> Optional[Dict[str, Any]]:
        if self._db.is_connected():
            try:
                res = self._db.campaigns.find_one({"campaign_id": campaign_id})
                if res:
                    return res
            except Exception:
                pass
        return self._cache.get(campaign_id)

    def list_campaigns(self, limit: int = 50) -> List[Dict[str, Any]]:
        if self._db.is_connected():
            try:
                return list(self._db.campaigns.find().sort("created_at", -1).limit(limit))
            except Exception:
                pass
        return sorted(list(self._cache.values()), key=lambda x: str(x.get("created_at", "")), reverse=True)[:limit]

    def update_campaign(self, campaign_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if self._db.is_connected():
            try:
                self._db.campaigns.update_one({"campaign_id": campaign_id}, {"$set": updates})
            except Exception:
                pass
        if campaign_id in self._cache:
            self._cache[campaign_id].update(updates)
            return self._cache[campaign_id]
        return self.get_by_id(campaign_id)

    def increment_counter(self, campaign_id: str, field_name: str, inc: int = 1) -> bool:
        if self._db.is_connected():
            try:
                self._db.campaigns.update_one({"campaign_id": campaign_id}, {"$inc": {field_name: inc}})
            except Exception:
                pass
        if campaign_id in self._cache:
            self._cache[campaign_id][field_name] = self._cache[campaign_id].get(field_name, 0) + inc
        return True

    def bulk_update_counters(self, campaign_id: str, counters_dict: Dict[str, int]) -> bool:
        """Atomically increment multiple counters at once via $inc."""
        if not counters_dict:
            return True
        if self._db.is_connected():
            try:
                self._db.campaigns.update_one({"campaign_id": campaign_id}, {"$inc": counters_dict})
            except Exception:
                pass
        if campaign_id in self._cache:
            for k, v in counters_dict.items():
                self._cache[campaign_id][k] = self._cache[campaign_id].get(k, 0) + v
        return True

    def delete_campaign(self, campaign_id: str) -> bool:
        """Soft-delete a campaign by setting is_deleted=True and status=CANCELLED."""
        now = datetime.now(timezone.utc)
        updates = {
            "is_deleted": True,
            "status": CampaignStatus.CANCELLED,
            "cancelled_at": now,
            "updated_at": now
        }
        self.update_campaign(campaign_id, updates)
        return True

    def list_campaigns_filtered(
        self,
        status: Optional[str] = None,
        search: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: int = 50,
        skip: int = 0,
        sort_by: str = "created_at",
        sort_dir: int = -1
    ) -> List[Dict[str, Any]]:
        query: Dict[str, Any] = {"is_deleted": {"$ne": True}}
        if status and status.upper() != "ALL":
            query["status"] = status.upper()
        if search:
            s = search.strip()
            query["$or"] = [
                {"title": {"$regex": s, "$options": "i"}},
                {"template_name": {"$regex": s, "$options": "i"}},
                {"description": {"$regex": s, "$options": "i"}},
                {"tags": {"$in": [s]}}
            ]
        if self._db.is_connected():
            try:
                cursor = self._db.campaigns.find(query).sort(sort_by, sort_dir).skip(skip).limit(limit)
                return list(cursor)
            except Exception:
                pass

        results = [c for c in self._cache.values() if not c.get("is_deleted")]
        if status and status.upper() != "ALL":
            results = [c for c in results if c.get("status") == status.upper()]
        if search:
            sc = search.lower()
            results = [
                c for c in results
                if sc in str(c.get("title", "")).lower()
                or sc in str(c.get("template_name", "")).lower()
                or sc in str(c.get("description", "")).lower()
                or sc in [str(t).lower() for t in c.get("tags", [])]
            ]
        results.sort(key=lambda x: str(x.get(sort_by, "")), reverse=(sort_dir == -1))
        return results[skip:skip+limit]

    def count_campaigns(self, status: Optional[str] = None, search: Optional[str] = None) -> int:
        query: Dict[str, Any] = {"is_deleted": {"$ne": True}}
        if status and status.upper() != "ALL":
            query["status"] = status.upper()
        if search:
            s = search.strip()
            query["$or"] = [
                {"title": {"$regex": s, "$options": "i"}},
                {"template_name": {"$regex": s, "$options": "i"}},
                {"description": {"$regex": s, "$options": "i"}},
                {"tags": {"$in": [s]}}
            ]
        if self._db.is_connected():
            try:
                return self._db.campaigns.count_documents(query)
            except Exception:
                pass
        return len(self.list_campaigns_filtered(status=status, search=search, limit=100000))

    def get_campaign_history(self, campaign_id: str) -> List[Dict[str, Any]]:
        """Fetch audit events related to this campaign."""
        query = {
            "$or": [
                {"metadata.campaign_id": campaign_id},
                {"lead_id": campaign_id},
                {"property_id": campaign_id}
            ]
        }
        if self._db.is_connected():
            try:
                return list(self._db.events.find(query).sort("timestamp", -1).limit(100))
            except Exception:
                pass
        return []

    def clone_campaign(self, campaign_id: str, new_title: Optional[str] = None, cloned_by: str = "admin") -> Optional[Dict[str, Any]]:
        """Duplicates a campaign configuration as a new DRAFT campaign."""
        original = self.get_by_id(campaign_id)
        if not original:
            return None
        cloned_data = dict(original)
        cloned_data.pop("_id", None)
        cloned_data["campaign_id"] = f"cmp_{uuid.uuid4().hex[:10]}"
        cloned_data["title"] = new_title or f"Copy of {original.get('title', 'Campaign')}"
        cloned_data["status"] = CampaignStatus.DRAFT
        cloned_data["cloned_from"] = campaign_id
        cloned_data["created_by"] = cloned_by
        now = datetime.now(timezone.utc)
        cloned_data["created_at"] = now
        cloned_data["updated_at"] = now
        cloned_data["started_at"] = None
        cloned_data["paused_at"] = None
        cloned_data["cancelled_at"] = None
        cloned_data["validated_at"] = None
        cloned_data["completed_at"] = None
        cloned_data["scheduled_at"] = None
        # Reset counters
        cloned_data["queued_count"] = 0
        cloned_data["sent_count"] = 0
        cloned_data["delivered_count"] = 0
        cloned_data["read_count"] = 0
        cloned_data["replied_count"] = 0
        cloned_data["failed_count"] = 0
        cloned_data["error_summary"] = {}
        cloned_data["is_deleted"] = False
        return self.create_campaign(cloned_data)


class CampaignRecipientRepository:
    _cache: Dict[str, List[Dict[str, Any]]] = {}

    def __init__(self):
        self._db = get_db()

    def bulk_insert(self, campaign_id: str, recipients: List[Dict[str, Any]]) -> int:
        if not recipients:
            return 0
        now = datetime.now(timezone.utc)
        docs = []
        for r in recipients:
            doc = {
                "campaign_id": campaign_id,
                "lead_id": str(r.get("lead_id", "")),
                "phone": str(r.get("phone") or r.get("wa_id", "")),
                "name": str(r.get("name") or "Valued Customer"),
                "status": r.get("status", "PENDING"),
                "outbound_id": r.get("outbound_id"),
                "meta_message_id": r.get("meta_message_id"),
                "error_code": r.get("error_code"),
                "error_message": r.get("error_message"),
                "error_category": r.get("error_category"),
                "queued_at": r.get("queued_at"),
                "sent_at": r.get("sent_at"),
                "delivered_at": r.get("delivered_at"),
                "read_at": r.get("read_at"),
                "failed_at": r.get("failed_at"),
                "retry_count": int(r.get("retry_count", 0)),
                "created_at": now
            }
            docs.append(doc)

        inserted_count = 0
        if self._db.is_connected():
            try:
                res = self._db.campaign_recipients.insert_many(docs, ordered=False)
                inserted_count = len(res.inserted_ids)
            except Exception:
                for d in docs:
                    try:
                        self._db.campaign_recipients.update_one(
                            {"campaign_id": campaign_id, "lead_id": d["lead_id"]},
                            {"$setOnInsert": d},
                            upsert=True
                        )
                        inserted_count += 1
                    except Exception:
                        pass
        self._cache[campaign_id] = docs
        return inserted_count or len(docs)

    def claim_next_batch(self, campaign_id: str, batch_size: int = 50) -> List[Dict[str, Any]]:
        """Atomically claim a batch of PENDING recipients and mark them QUEUED."""
        now = datetime.now(timezone.utc)
        claimed = []
        if self._db.is_connected():
            try:
                for _ in range(batch_size):
                    doc = self._db.campaign_recipients.find_one_and_update(
                        {"campaign_id": campaign_id, "status": "PENDING"},
                        {"$set": {"status": "QUEUED", "queued_at": now}},
                        return_document=True
                    )
                    if not doc:
                        break
                    claimed.append(doc)
                return claimed
            except Exception:
                pass

        # In-memory fallback
        camp_list = self._cache.get(campaign_id, [])
        for r in camp_list:
            if r.get("status") == "PENDING" and len(claimed) < batch_size:
                r["status"] = "QUEUED"
                r["queued_at"] = now
                claimed.append(r)
        return claimed

    def update_status(self, campaign_id: str, lead_id: str, status: str, **kwargs) -> bool:
        now = datetime.now(timezone.utc)
        updates: Dict[str, Any] = {"status": status}
        if status == "SENT":
            updates["sent_at"] = now
        elif status == "DELIVERED":
            updates["delivered_at"] = now
        elif status == "READ":
            updates["read_at"] = now
        elif status == "FAILED":
            updates["failed_at"] = now
        for k, v in kwargs.items():
            if v is not None:
                updates[k] = v

        if self._db.is_connected():
            try:
                self._db.campaign_recipients.update_one(
                    {"campaign_id": campaign_id, "lead_id": str(lead_id)},
                    {"$set": updates}
                )
            except Exception:
                pass

        for r in self._cache.get(campaign_id, []):
            if r.get("lead_id") == str(lead_id):
                r.update(updates)
                break
        return True

    def update_by_meta_id(self, meta_message_id: str, status: str, **kwargs) -> Optional[str]:
        """Update recipient by Meta message ID (for webhook callbacks). Returns campaign_id if found."""
        if not meta_message_id:
            return None
        now = datetime.now(timezone.utc)
        updates: Dict[str, Any] = {"status": status}
        if status == "DELIVERED":
            updates["delivered_at"] = now
        elif status == "READ":
            updates["read_at"] = now
        elif status == "FAILED":
            updates["failed_at"] = now
        for k, v in kwargs.items():
            if v is not None:
                updates[k] = v

        campaign_id = None
        if self._db.is_connected():
            try:
                doc = self._db.campaign_recipients.find_one_and_update(
                    {"meta_message_id": meta_message_id},
                    {"$set": updates}
                )
                if doc:
                    campaign_id = doc.get("campaign_id")
            except Exception:
                pass

        if not campaign_id:
            for cid, r_list in self._cache.items():
                for r in r_list:
                    if r.get("meta_message_id") == meta_message_id:
                        r.update(updates)
                        return cid
        return campaign_id

    def get_stats(self, campaign_id: str) -> Dict[str, Any]:
        if self._db.is_connected():
            try:
                pipeline = [
                    {"$match": {"campaign_id": campaign_id}},
                    {"$group": {"_id": "$status", "count": {"$sum": 1}}}
                ]
                res = list(self._db.campaign_recipients.aggregate(pipeline))
                counts = {item["_id"]: item["count"] for item in res}
                pending = counts.get("PENDING", 0)
                queued = counts.get("QUEUED", 0)
                sent = counts.get("SENT", 0)
                delivered = counts.get("DELIVERED", 0)
                read = counts.get("READ", 0)
                failed = counts.get("FAILED", 0)
                skipped = counts.get("SKIPPED", 0)
                total = sum(counts.values())
                total_delivered = delivered + read
                total_sent = sent + total_delivered
                return {
                    "total": total,
                    "pending": pending,
                    "queued": queued,
                    "sent": total_sent,
                    "sent_only": sent,
                    "delivered": total_delivered,
                    "read": read,
                    "failed": failed,
                    "skipped": skipped,
                    "delivery_rate": round(total_delivered / total_sent * 100, 1) if total_sent > 0 else 0.0,
                    "read_rate": round(read / total_delivered * 100, 1) if total_delivered > 0 else 0.0,
                }
            except Exception:
                pass

        r_list = self._cache.get(campaign_id, [])
        pending = sum(1 for r in r_list if r.get("status") == "PENDING")
        queued = sum(1 for r in r_list if r.get("status") == "QUEUED")
        sent = sum(1 for r in r_list if r.get("status") == "SENT")
        delivered = sum(1 for r in r_list if r.get("status") == "DELIVERED")
        read = sum(1 for r in r_list if r.get("status") == "READ")
        failed = sum(1 for r in r_list if r.get("status") == "FAILED")
        skipped = sum(1 for r in r_list if r.get("status") == "SKIPPED")
        total = len(r_list)
        total_delivered = delivered + read
        total_sent = sent + total_delivered
        return {
            "total": total,
            "pending": pending,
            "queued": queued,
            "sent": total_sent,
            "sent_only": sent,
            "delivered": total_delivered,
            "read": read,
            "failed": failed,
            "skipped": skipped,
            "delivery_rate": round(total_delivered / total_sent * 100, 1) if total_sent > 0 else 0.0,
            "read_rate": round(read / total_delivered * 100, 1) if total_delivered > 0 else 0.0,
        }

    def list_recipients(
        self,
        campaign_id: str,
        status_filter: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 50,
        skip: int = 0
    ) -> List[Dict[str, Any]]:
        query: Dict[str, Any] = {"campaign_id": campaign_id}
        if status_filter and status_filter.upper() != "ALL":
            query["status"] = status_filter.upper()
        if search:
            s = search.strip()
            query["$or"] = [
                {"name": {"$regex": s, "$options": "i"}},
                {"phone": {"$regex": s, "$options": "i"}},
                {"lead_id": {"$regex": s, "$options": "i"}},
            ]
        if self._db.is_connected():
            try:
                return list(self._db.campaign_recipients.find(query).sort("created_at", 1).skip(skip).limit(limit))
            except Exception:
                pass
        items = self._cache.get(campaign_id, [])
        if status_filter and status_filter.upper() != "ALL":
            items = [r for r in items if r.get("status") == status_filter.upper()]
        if search:
            sc = search.lower()
            items = [r for r in items if sc in str(r.get("name", "")).lower() or sc in str(r.get("phone", "")) or sc in str(r.get("lead_id", ""))]
        return items[skip:skip+limit]

    def count_recipients(self, campaign_id: str, status_filter: Optional[str] = None, search: Optional[str] = None) -> int:
        query: Dict[str, Any] = {"campaign_id": campaign_id}
        if status_filter and status_filter.upper() != "ALL":
            query["status"] = status_filter.upper()
        if search:
            s = search.strip()
            query["$or"] = [
                {"name": {"$regex": s, "$options": "i"}},
                {"phone": {"$regex": s, "$options": "i"}},
                {"lead_id": {"$regex": s, "$options": "i"}},
            ]
        if self._db.is_connected():
            try:
                return self._db.campaign_recipients.count_documents(query)
            except Exception:
                pass
        return len(self.list_recipients(campaign_id, status_filter, search, limit=100000))

    def mark_failed_for_retry(self, campaign_id: str, lead_id: Optional[str] = None) -> int:
        """Reset failed recipients back to PENDING and increment retry_count."""
        filter_dict: Dict[str, Any] = {"campaign_id": campaign_id, "status": "FAILED"}
        if lead_id:
            filter_dict["lead_id"] = str(lead_id)
        retried = 0
        if self._db.is_connected():
            try:
                res = self._db.campaign_recipients.update_many(
                    filter_dict,
                    {
                        "$set": {"status": "PENDING", "error_code": None, "error_message": None, "failed_at": None},
                        "$inc": {"retry_count": 1}
                    }
                )
                retried = res.modified_count
            except Exception:
                pass
        for r in self._cache.get(campaign_id, []):
            if r.get("status") == "FAILED":
                if not lead_id or r.get("lead_id") == str(lead_id):
                    r["status"] = "PENDING"
                    r["error_code"] = None
                    r["error_message"] = None
                    r["retry_count"] = r.get("retry_count", 0) + 1
                    retried = max(retried, 1)
        return retried

    def get_error_breakdown(self, campaign_id: str) -> Dict[str, Any]:
        """Aggregate failure error codes and categories."""
        if self._db.is_connected():
            try:
                pipeline = [
                    {"$match": {"campaign_id": campaign_id, "status": "FAILED"}},
                    {"$group": {"_id": {"code": "$error_code", "category": "$error_category"}, "count": {"$sum": 1}, "samples": {"$push": "$phone"}}}
                ]
                res = list(self._db.campaign_recipients.aggregate(pipeline))
                errors = []
                for item in res:
                    _id = item.get("_id") or {}
                    errors.append({
                        "error_code": _id.get("code") or "UNKNOWN",
                        "error_category": _id.get("category") or "OTHER",
                        "count": item.get("count", 0),
                        "sample_phones": item.get("samples", [])[:3]
                    })
                return {"total_failed": sum(e["count"] for e in errors), "errors": errors}
            except Exception:
                pass
        return {"total_failed": 0, "errors": []}


class WebhookRepository:
    def __init__(self):
        self._db = get_db()

    def save_event(self, request_id: str, headers: Dict, payload: Dict, status: str = "received") -> str:
        doc = {
            "request_id": request_id,
            "headers": headers,
            "payload": payload,
            "status": status,
            "created_at": datetime.now(timezone.utc)
        }
        if self._db.is_connected():
            try:
                self._db.webhook_events.insert_one(doc)
            except Exception:
                pass
        return request_id

    def update_event_status(self, request_id: str, status: str, processing_time: float = 0.0, error: str = ""):
        if self._db.is_connected():
            try:
                updates: Dict[str, Any] = {"status": status}
                if processing_time:
                    updates["processing_time_ms"] = processing_time
                if error:
                    updates["error"] = error
                self._db.webhook_events.update_one({"request_id": request_id}, {"$set": updates})
            except Exception:
                pass


class EventRepository:
    _cache_events: List[Dict[str, Any]] = []

    def __init__(self):
        self._db = get_db()
        self._db_manager = self._db

    def log_event(
        self,
        event_type: str,
        lead_id: Optional[str] = None,
        wa_id: str = "",
        property_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if not event_type:
            raise ValueError("event_type is required")

        evt = AnalyticsEvent(
            event_type=event_type,
            lead_id=lead_id,
            wa_id=str(wa_id),
            property_id=property_id,
            metadata=metadata or {},
            timestamp=datetime.now(timezone.utc)
        ).to_dict()

        if self._db.is_connected():
            try:
                self._db.events.insert_one(evt)
            except Exception:
                pass

        self._cache_events.append(evt)
        if len(self._cache_events) > 1000:
            self._cache_events.pop(0)
        return evt

    def list_events(
        self,
        filter_query: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        skip: int = 0
    ) -> List[Dict[str, Any]]:
        query = filter_query or {}
        if self._db.is_connected():
            try:
                return list(self._db.events.find(query).sort("timestamp", -1).skip(skip).limit(limit))
            except Exception:
                pass
        filtered = [
            e for e in reversed(self._cache_events)
            if all(e.get(k) == val for k, val in query.items())
        ]
        return filtered[skip: skip + limit]

    def get_counts_by_type(self, start_time: Optional[datetime] = None) -> Dict[str, int]:
        match_stage: Dict[str, Any] = {}
        if start_time:
            match_stage["timestamp"] = {"$gte": start_time}

        if self._db.is_connected():
            try:
                pipeline = []
                if match_stage:
                    pipeline.append({"$match": match_stage})
                pipeline.append({"$group": {"_id": "$event_type", "count": {"$sum": 1}}})
                results = list(self._db.events.aggregate(pipeline))
                return {r["_id"]: r["count"] for r in results if r.get("_id")}
            except Exception:
                pass
        counts: Dict[str, int] = {}
        for e in self._cache_events:
            if not start_time or e.get("timestamp", datetime.min.replace(tzinfo=timezone.utc)) >= start_time:
                t = e.get("event_type", "unknown")
                counts[t] = counts.get(t, 0) + 1
        return counts

    def get_daily_activity(self, days: int = 7) -> List[Dict[str, Any]]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        if self._db.is_connected():
            try:
                pipeline = [
                    {"$match": {"timestamp": {"$gte": cutoff}}},
                    {
                        "$group": {
                            "_id": {
                                "$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}
                            },
                            "total": {"$sum": 1},
                            "messages": {"$sum": {"$cond": [{"$eq": ["$event_type", "message_received"]}, 1, 0]}},
                            "visits": {"$sum": {"$cond": [{"$eq": ["$event_type", "visit_booked"]}, 1, 0]}}
                        }
                    },
                    {"$sort": {"_id": 1}}
                ]
                return list(self._db.events.aggregate(pipeline))
            except Exception:
                pass

        daily: Dict[str, Dict[str, int]] = {}
        for d in range(days):
            day_str = (cutoff + timedelta(days=d)).strftime("%Y-%m-%d")
            daily[day_str] = {"_id": day_str, "total": 0, "messages": 0, "visits": 0}
        for e in self._cache_events:
            ts = e.get("timestamp")
            if ts and ts >= cutoff:
                day_str = ts.strftime("%Y-%m-%d")
                if day_str in daily:
                    daily[day_str]["total"] += 1
                    if e.get("event_type") == "message_received":
                        daily[day_str]["messages"] += 1
                    elif e.get("event_type") == "visit_booked":
                        daily[day_str]["visits"] += 1
        return list(daily.values())


# =============================================================================
# VECTOR STORE & KNOWLEDGE BASE REPOSITORY (QDRANT & HYBRID FALLBACK)
# =============================================================================

import json
import os
import math


class VectorStoreManager:
    """
    Vector Store Manager supporting:
    1. Docker Qdrant (http://localhost:6333) via qdrant-client
    2. Embedded/Persistent local Qdrant (path="./storage/qdrant")
    3. In-memory cosine similarity fallback using pre-computed embeddings (storage/knowledge_store.json)
    """
    _chunks_cache: List[Dict[str, Any]] = []
    _initialized: bool = False

    def __init__(self):
        self.host = getattr(Config, "QDRANT_HOST", "localhost")
        self.port = getattr(Config, "QDRANT_PORT", 6333)
        self.collection_name = getattr(Config, "QDRANT_COLLECTION", "aris_real_estate_knowledge")
        self.client = None
        self.mode = "in_memory"
        self._init_vector_store()
        self._load_seed_chunks()

    def _init_vector_store(self):
        try:
            import socket
            from qdrant_client import QdrantClient
            from qdrant_client.http.models import Distance, VectorParams

            # 1. Check if Docker Qdrant is listening on port
            docker_reachable = False
            try:
                with socket.create_connection((self.host, self.port), timeout=0.3):
                    docker_reachable = True
            except (socket.timeout, ConnectionRefusedError, OSError):
                docker_reachable = False

            if docker_reachable:
                try:
                    client = QdrantClient(url=f"http://{self.host}:{self.port}", timeout=1.0, check_compatibility=False)
                    collections = [c.name for c in client.get_collections().collections]
                    if self.collection_name not in collections:
                        client.create_collection(
                            collection_name=self.collection_name,
                            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                        )
                    self.client = client
                    self.mode = "qdrant_docker"
                    print(f"[VECTOR_DB] Connected to Docker Qdrant at http://{self.host}:{self.port}")
                    return
                except Exception:
                    pass

            # 2. In-memory / Embedded Qdrant Engine (instant, zero network latency)
            try:
                client = QdrantClient(location=":memory:")
                client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                )
                self.client = client
                self.mode = "qdrant_memory"
                print(f"[VECTOR_DB] Initialized in-memory Qdrant engine (Collection: {self.collection_name})")
                return
            except Exception:
                pass
        except Exception as e:
            print(f"[VECTOR_DB NOTICE] Qdrant library note: {e}")

        self.mode = "in_memory_cosine"
        print("[VECTOR_DB] Running with high-availability in-memory vector cosine similarity index.")

    def _load_seed_chunks(self):
        if VectorStoreManager._initialized:
            return
        json_path = os.path.join("storage", "knowledge_store.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    chunks = data.get("chunks", [])
                    for chk in chunks:
                        self.upsert_chunk(chk)
                print(f"[VECTOR_DB] Loaded {len(VectorStoreManager._chunks_cache)} seed knowledge chunks from storage.")
            except Exception as ex:
                print(f"[VECTOR_DB ERROR] Failed to load knowledge_store.json: {ex}")
        VectorStoreManager._initialized = True

    def upsert_chunk(self, chunk: Dict[str, Any]) -> bool:
        chunk_id = chunk.get("id") or f"chk_{uuid.uuid4().hex[:12]}"
        chunk["id"] = chunk_id
        existing = [i for i, c in enumerate(VectorStoreManager._chunks_cache) if c.get("id") == chunk_id]
        if existing:
            VectorStoreManager._chunks_cache[existing[0]] = chunk
        else:
            VectorStoreManager._chunks_cache.append(chunk)

        if self.client:
            try:
                from qdrant_client.http.models import PointStruct
                embedding = chunk.get("embedding")
                if embedding and len(embedding) == 384:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=[
                            PointStruct(
                                id=abs(hash(chunk_id)) % (2**63 - 1),
                                vector=embedding,
                                payload={k: v for k, v in chunk.items() if k != "embedding"}
                            )
                        ]
                    )
            except Exception:
                pass
        return True

    def search(self, query_embedding: List[float], top_k: int = 3, filter_property_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if not query_embedding or len(query_embedding) != 384:
            return []

        if self.client:
            try:
                hits = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=top_k
                )
                results = []
                for h in hits:
                    item = dict(h.payload)
                    item["score"] = h.score
                    if not filter_property_id or item.get("property_id") == filter_property_id:
                        results.append(item)
                if results:
                    return results[:top_k]
            except Exception:
                pass

        # In-memory Cosine Similarity Fallback
        def cosine_sim(v1: List[float], v2: List[float]) -> float:
            dot = sum(a * b for a, b in zip(v1, v2))
            norm1 = math.sqrt(sum(a * a for a in v1))
            norm2 = math.sqrt(sum(b * b for b in v2))
            return dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0

        scored = []
        for chk in VectorStoreManager._chunks_cache:
            emb = chk.get("embedding")
            if not emb or len(emb) != 384:
                continue
            if filter_property_id and chk.get("property_id") != filter_property_id:
                continue
            score = cosine_sim(query_embedding, emb)
            item = dict(chk)
            item["score"] = score
            scored.append(item)

        scored.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return scored[:top_k]

    def count_chunks(self) -> int:
        return len(VectorStoreManager._chunks_cache)


class KnowledgeRepository:
    def __init__(self):
        self._db = get_db()

    def list_documents(self) -> List[Dict[str, Any]]:
        if self._db.is_connected():
            try:
                return list(self._db.knowledge_documents.find().sort("created_at", -1))
            except Exception:
                pass
        return []


# =============================================================================
# UNIFIED DATABASE ACCESS INTERFACE (DB)
# =============================================================================

class DatabaseService:
    """Unified container providing centralized access to all repositories."""
    def __init__(self):
        self.users = UserRepository()
        self.conversations = ConversationRepository()
        self.messages = MessageRepository()
        self.sessions = SessionRepository()
        self.leads = LeadRepository()
        self.properties = PropertyRepository()
        self.visits = VisitRepository()
        self.followups = FollowUpRepository()
        self.campaigns = CampaignRepository()
        self.campaign_recipients = CampaignRecipientRepository()
        self.outbound = OutboundMessageRepository()
        self.outbound_messages = self.outbound
        self.webhooks = WebhookRepository()
        self.events = EventRepository()
        self.vectors = VectorStoreManager()
        self.knowledge = KnowledgeRepository()
        self.customer_memory = CustomerMemoryRepository()
        self.sales_memory = SalesMemoryRepository()
        self.sales_events = SalesEventRepository()


DB = DatabaseService()

# Register module aliases for backward compatibility
import sys
sys.modules["database.mongodb"] = sys.modules[__name__]


