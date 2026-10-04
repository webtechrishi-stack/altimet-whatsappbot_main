"""
Conversion Service — tracks visit readiness score (0-100), lead temperature,
sales memory state, and logs audit events to the database.
"""

import re
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

from database import DB
from conversation.models import CustomerMemory, SalesMemory

logger = logging.getLogger(__name__)


class ConversionService:
    """
    Computes conversion scores and persists structured sales events.
    """

    def __init__(self):
        self.sales_memory_repo = DB.sales_memory
        self.sales_event_repo = DB.sales_events
        self.lead_repo = DB.leads
        self.conv_repo = DB.conversations

    @classmethod
    def calculate_readiness_score(
        cls,
        customer_memory: Any,
        sales_memory: Any = None,
        sales_stage: Any = None,
        recent_text: str = ""
    ) -> int:
        svc = cls()
        visit_agreed = False
        stage_str = sales_stage.value if hasattr(sales_stage, "value") else str(sales_stage or "")
        if stage_str in ("VISIT_BOOKED", "VISIT_COMPLETED") or getattr(sales_memory, "visit_status", "") in ("CONFIRMED", "BOOKED"):
            visit_agreed = True
        return svc.compute_visit_readiness(customer_memory, recent_text=recent_text, visit_agreed=visit_agreed)

    @classmethod
    def calculate_lead_temperature(
        cls,
        readiness_score: int,
        sales_stage: Any = None,
        recent_text: str = ""
    ) -> str:
        svc = cls()
        stage_str = sales_stage.value if hasattr(sales_stage, "value") else str(sales_stage or "")
        if stage_str in ("VISIT_BOOKED", "VISIT_COMPLETED", "VISIT_NEGOTIATING", "VISIT_PITCHED"):
            return "HOT"
        return svc.compute_lead_temperature(readiness_score, recent_text=recent_text)

    @classmethod
    def log_sales_event(
        cls,
        phone: str = "",
        conversation_id: str = "",
        event_type: str = "",
        previous_stage: str = "",
        new_stage: str = "",
        readiness_score: int = 0,
        lead_temperature: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        svc = cls()
        meta = metadata or {}
        if previous_stage: meta["previous_stage"] = previous_stage
        if new_stage: meta["new_stage"] = new_stage
        if readiness_score: meta["readiness_score"] = readiness_score
        if lead_temperature: meta["lead_temperature"] = lead_temperature
        return svc.sales_event_repo.log_sales_event(
            event_type=event_type,
            lead_id=phone,
            conversation_id=conversation_id,
            metadata=meta
        )

    def compute_visit_readiness(
        self,
        customer_memory: CustomerMemory,
        recent_text: str = "",
        visit_agreed: bool = False
    ) -> int:
        """
        Calculates multi-factor visit readiness score from 0 to 100.
        """
        score = 0
        req = customer_memory.requirements

        # 1. Stated requirements weights
        if req.get("budget_max"):
            score += 15
        if req.get("bhk"):
            score += 10
        if req.get("locality") or req.get("city"):
            score += 10
        if req.get("purpose"):
            score += 10
        if req.get("timeline"):
            score += 10

        # 2. Specific project interest
        if customer_memory.recommended_properties:
            score += 15

        # 3. Conversational signals in recent message
        lower = recent_text.lower()
        if re.search(r"\b(?:price|cost|how much|emi)\b", lower):
            score += 5
        if re.search(r"\b(?:brochure|pdf|layout|floor plan)\b", lower):
            score += 5
        if re.search(r"\b(?:where is|exact location|landmark|distance)\b", lower):
            score += 5
        if re.search(r"\b(?:available|ready|possession date|units left)\b", lower):
            score += 10
        if re.search(r"\b(?:visit|come|see|tour|appointment|weekend|saturday|sunday)\b", lower):
            score += 25

        if visit_agreed:
            score = max(score, 90)

        return min(score, 100)

    def compute_lead_temperature(self, readiness_score: int, recent_text: str = "") -> str:
        """
        Maps readiness score and explicit high-intent signals to COLD, WARM, or HOT.
        """
        lower = recent_text.lower()
        if any(w in lower for w in ["book", "visit", "confirm", "available", "token", "deposit", "saturday 4", "tomorrow"]):
            return "HOT"

        if readiness_score >= 70:
            return "HOT"
        elif readiness_score >= 35:
            return "WARM"
        return "COLD"

    def sync_sales_state(
        self,
        lead_id: str,
        conversation_id: str,
        sales_stage: str,
        next_best_action: str,
        visit_readiness: int,
        lead_temperature: str,
        objection: Optional[str] = None,
        recommended_property: Optional[str] = None
    ) -> SalesMemory:
        """
        Updates SalesMemory, Conversation, Lead, and logs sales events.
        """
        raw_sm = self.sales_memory_repo.get_by_lead_id(lead_id)
        if raw_sm:
            try:
                sm = SalesMemory.from_dict(raw_sm)
            except Exception:
                sm = SalesMemory(lead_id=str(lead_id), conversation_id=str(conversation_id))
        else:
            sm = SalesMemory(lead_id=str(lead_id), conversation_id=str(conversation_id))

        old_stage = sm.sales_stage

        sm.conversation_id = conversation_id
        sm.sales_stage = sales_stage
        sm.next_best_action = next_best_action
        sm.visit_readiness_score = visit_readiness
        sm.lead_temperature = lead_temperature

        if objection:
            sm.last_customer_objection = objection
            if objection not in sm.open_objections:
                sm.open_objections.append(objection)

        if recommended_property:
            sm.last_recommended_property = recommended_property

        sm.last_sales_action = next_best_action
        sm.updated_at = datetime.now(timezone.utc)

        # Persist SalesMemory
        self.sales_memory_repo.save_or_update(lead_id, sm.to_dict())

        # Update Conversation entity
        self.conv_repo.update_sales_state(
            conversation_id=conversation_id,
            sales_stage=sales_stage,
            last_sales_action=next_best_action,
            visit_readiness_score=visit_readiness
        )

        # Update Lead entity
        self.lead_repo.update(lead_id, {
            "sales_stage": sales_stage,
            "lead_temperature": lead_temperature
        })

        # Audit Event if stage changed
        if old_stage != sales_stage:
            self.sales_event_repo.log_sales_event(
                event_type="SALES_STAGE_CHANGED",
                lead_id=lead_id,
                conversation_id=conversation_id,
                metadata={
                    "old_stage": old_stage,
                    "new_stage": sales_stage,
                    "reason": next_best_action
                }
            )

        # Audit Next Best Action
        self.sales_event_repo.log_sales_event(
            event_type="NEXT_BEST_ACTION_DETERMINED",
            lead_id=lead_id,
            conversation_id=conversation_id,
            metadata={
                "action": next_best_action,
                "readiness": visit_readiness,
                "temperature": lead_temperature
            }
        )

        return sm
