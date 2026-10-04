"""
Data models for Conversation Memory, Customer Memory, and Personalized Context.
"""

import re
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone


@dataclass
class FactOrInference:
    """
    Distinguishes explicit customer facts from inferred preferences.
    Explicit customer facts always take precedence over uncertain inferences.
    """
    value: Any
    source: str = "CUSTOMER"    # "CUSTOMER" (explicit) or "INFERRED"
    confidence: float = 1.0     # 1.0 for explicit statement, < 1.0 for inference
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "source": self.source,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FactOrInference":
        ts = data.get("timestamp")
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                ts = datetime.now(timezone.utc)
        return cls(
            value=data.get("value"),
            source=data.get("source", "CUSTOMER"),
            confidence=float(data.get("confidence", 1.0)),
            timestamp=ts or datetime.now(timezone.utc)
        )


@dataclass
class CustomerMemory:
    """
    Persistent first-class memory of a customer's stated requirements and preferences.
    """
    lead_id: str = ""
    phone: str = ""
    customer_summary: str = ""
    requirements: Dict[str, Any] = field(default_factory=lambda: {
        "city": None,
        "locality": None,
        "bhk": None,
        "budget_min": None,
        "budget_max": None,
        "purpose": "END_USE",
        "timeline": None
    })
    preferences: List[str] = field(default_factory=list)
    recommended_properties: List[str] = field(default_factory=list)
    objections: List[Dict[str, Any]] = field(default_factory=list)
    visit: Dict[str, Any] = field(default_factory=lambda: {
        "interest": False,
        "status": "NOT_BOOKED",
        "visit_id": None
    })
    explicit_facts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    inferred_preferences: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    history_log: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        if not self.lead_id and self.phone:
            self.lead_id = self.phone
        elif not self.phone and self.lead_id:
            self.phone = self.lead_id

    @property
    def bhk(self) -> List[str]:
        val = self.requirements.get("bhk")
        if isinstance(val, list):
            res = list(val)
        elif isinstance(val, str):
            res = [val]
        else:
            return []
        expanded = set(res)
        for item in res:
            m = re.match(r"^([1-5])\s*BHK$", str(item), re.IGNORECASE)
            if m:
                expanded.add(f"{m.group(1)}BHK")
                expanded.add(f"{m.group(1)} BHK")
        return list(expanded)

    @property
    def budget_max(self) -> Optional[float]:
        val = self.requirements.get("budget_max")
        return float(val) if val is not None else None

    @property
    def budget_min(self) -> Optional[float]:
        val = self.requirements.get("budget_min")
        return float(val) if val is not None else None

    @property
    def preferred_locality(self) -> str:
        return self.requirements.get("locality") or ""

    @property
    def city(self) -> str:
        return self.requirements.get("city") or ""

    @property
    def purpose(self) -> str:
        return self.requirements.get("purpose") or ""

    @property
    def timeline(self) -> str:
        return self.requirements.get("timeline") or ""

    @property
    def facts(self) -> Dict[str, Any]:
        """Provides dot-accessible fact objects with .is_explicit and .confidence."""
        class FactAccessor:
            def __init__(self, data: Dict[str, Any]):
                self.value = data.get("value")
                self.source = data.get("source", "CUSTOMER")
                self.confidence = float(data.get("confidence", 1.0))
                self.is_explicit = (self.source == "CUSTOMER" and self.confidence >= 0.9)

            def __repr__(self):
                return f"Fact({self.value}, is_explicit={self.is_explicit})"

        res = {}
        for k, v in self.explicit_facts.items():
            res[k] = FactAccessor(v if isinstance(v, dict) else {"value": v})
        # Map convenience keys
        if "budget_max" in res and "budget" not in res:
            res["budget"] = res["budget_max"]
        if "locality" in res and "location" not in res:
            res["location"] = res["locality"]
        return res

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat() if isinstance(self.created_at, datetime) else str(self.created_at)
        d["updated_at"] = self.updated_at.isoformat() if isinstance(self.updated_at, datetime) else str(self.updated_at)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CustomerMemory":
        data = data.copy()
        data.pop("_id", None)
        for k in ("created_at", "updated_at"):
            if k in data and isinstance(data[k], str):
                try:
                    data[k] = datetime.fromisoformat(data[k].replace("Z", "+00:00"))
                except Exception:
                    data[k] = datetime.now(timezone.utc)
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class SalesMemory:
    """
    Persistent sales state memory for a lead and conversation.
    """
    lead_id: str = ""
    phone: str = ""
    conversation_id: str = ""
    lead_temperature: str = "WARM"          # COLD, WARM, HOT
    sales_stage: str = "NEW"                # Sales state machine stage
    visit_readiness_score: int = 0          # 0-100
    last_sales_action: Optional[str] = None
    last_customer_objection: Optional[str] = None
    open_objections: List[str] = field(default_factory=list)
    visit_interest: bool = False
    visit_status: str = "NOT_BOOKED"        # NOT_BOOKED, BOOKED, COMPLETED, CANCELLED
    visit_pitched: bool = False
    recommended_properties: List[str] = field(default_factory=list)
    last_recommended_property: Optional[str] = None
    next_best_action: str = "DISCOVER_REQUIREMENT"
    action_reason: str = "INITIAL_GREETING"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        if not self.lead_id and self.phone:
            self.lead_id = self.phone
        elif not self.phone and self.lead_id:
            self.phone = self.lead_id

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat() if isinstance(self.created_at, datetime) else str(self.created_at)
        d["updated_at"] = self.updated_at.isoformat() if isinstance(self.updated_at, datetime) else str(self.updated_at)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SalesMemory":
        data = data.copy()
        data.pop("_id", None)
        for k in ("created_at", "updated_at"):
            if k in data and isinstance(data[k], str):
                try:
                    data[k] = datetime.fromisoformat(data[k].replace("Z", "+00:00"))
                except Exception:
                    data[k] = datetime.now(timezone.utc)
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class ConversationContext:
    """
    Assembled compact personalized context payload for Gemini and Sales Agent.
    """
    customer_memory: CustomerMemory
    sales_memory: SalesMemory
    recent_messages: List[Dict[str, Any]]
    relevant_history: List[Dict[str, Any]]
    property_context: str
    rag_context: str
    next_best_action: Dict[str, Any]
    conversation_summary: str
    current_message: str
