"""
Deterministic Sales State Machine for ARIS WhatsApp Conversations.
Maintains clear conversation-level stages separate from CRM Lead stages.
"""

from enum import Enum
from typing import Dict, Any, Optional, Tuple, List


class ConversationSalesStage(str, Enum):
    NEW = "NEW"
    DISCOVERY = "DISCOVERY"
    QUALIFIED = "QUALIFIED"
    PROPERTY_RECOMMENDED = "PROPERTY_RECOMMENDED"
    ENGAGED = "ENGAGED"
    VISIT_PITCHED = "VISIT_PITCHED"
    VISIT_NEGOTIATING = "VISIT_NEGOTIATING"
    VISIT_BOOKED = "VISIT_BOOKED"
    VISIT_COMPLETED = "VISIT_COMPLETED"
    FOLLOW_UP = "FOLLOW_UP"
    WON = "WON"
    LOST = "LOST"
    OPTED_OUT = "OPTED_OUT"


class SalesStateMachine:
    """
    Evaluates and applies deterministic state transitions.
    Uses structured logic, never arbitrary ungrounded LLM strings.
    """

    ALLOWED_TRANSITIONS: Dict[ConversationSalesStage, List[ConversationSalesStage]] = {
        ConversationSalesStage.NEW: [
            ConversationSalesStage.DISCOVERY,
            ConversationSalesStage.QUALIFIED,
            ConversationSalesStage.PROPERTY_RECOMMENDED,
            ConversationSalesStage.LOST
        ],
        ConversationSalesStage.DISCOVERY: [
            ConversationSalesStage.QUALIFIED,
            ConversationSalesStage.PROPERTY_RECOMMENDED,
            ConversationSalesStage.ENGAGED,
            ConversationSalesStage.LOST
        ],
        ConversationSalesStage.QUALIFIED: [
            ConversationSalesStage.PROPERTY_RECOMMENDED,
            ConversationSalesStage.ENGAGED,
            ConversationSalesStage.VISIT_PITCHED,
            ConversationSalesStage.LOST
        ],
        ConversationSalesStage.PROPERTY_RECOMMENDED: [
            ConversationSalesStage.ENGAGED,
            ConversationSalesStage.VISIT_PITCHED,
            ConversationSalesStage.VISIT_NEGOTIATING,
            ConversationSalesStage.DISCOVERY,
            ConversationSalesStage.LOST
        ],
        ConversationSalesStage.ENGAGED: [
            ConversationSalesStage.VISIT_PITCHED,
            ConversationSalesStage.VISIT_NEGOTIATING,
            ConversationSalesStage.DISCOVERY,
            ConversationSalesStage.FOLLOW_UP,
            ConversationSalesStage.LOST
        ],
        ConversationSalesStage.VISIT_PITCHED: [
            ConversationSalesStage.VISIT_NEGOTIATING,
            ConversationSalesStage.VISIT_BOOKED,
            ConversationSalesStage.ENGAGED,
            ConversationSalesStage.FOLLOW_UP,
            ConversationSalesStage.LOST
        ],
        ConversationSalesStage.VISIT_NEGOTIATING: [
            ConversationSalesStage.VISIT_BOOKED,
            ConversationSalesStage.VISIT_PITCHED,
            ConversationSalesStage.ENGAGED,
            ConversationSalesStage.FOLLOW_UP
        ],
        ConversationSalesStage.VISIT_BOOKED: [
            ConversationSalesStage.VISIT_COMPLETED,
            ConversationSalesStage.VISIT_NEGOTIATING,
            ConversationSalesStage.FOLLOW_UP,
            ConversationSalesStage.WON,
            ConversationSalesStage.LOST
        ],
        ConversationSalesStage.VISIT_COMPLETED: [
            ConversationSalesStage.FOLLOW_UP,
            ConversationSalesStage.WON,
            ConversationSalesStage.LOST
        ],
        ConversationSalesStage.FOLLOW_UP: [
            ConversationSalesStage.ENGAGED,
            ConversationSalesStage.VISIT_PITCHED,
            ConversationSalesStage.VISIT_NEGOTIATING,
            ConversationSalesStage.VISIT_BOOKED,
            ConversationSalesStage.LOST
        ],
        ConversationSalesStage.WON: [],
        ConversationSalesStage.LOST: [
            ConversationSalesStage.NEW,
            ConversationSalesStage.DISCOVERY
        ],
        ConversationSalesStage.OPTED_OUT: [
            ConversationSalesStage.NEW,
            ConversationSalesStage.DISCOVERY,
            ConversationSalesStage.ENGAGED
        ],
    }

    @classmethod
    def can_transition(cls, from_stage: Any, to_stage: Any) -> bool:
        try:
            f_val = ConversationSalesStage(from_stage) if not isinstance(from_stage, ConversationSalesStage) else from_stage
            t_val = ConversationSalesStage(to_stage) if not isinstance(to_stage, ConversationSalesStage) else to_stage
            if t_val == ConversationSalesStage.OPTED_OUT:
                return True
            return t_val in cls.ALLOWED_TRANSITIONS.get(f_val, [])
        except Exception:
            return False

    @classmethod
    def evaluate_stage_progression(
        cls,
        current_stage: Any,
        user_message: str = "",
        has_properties: bool = False,
        visit_status: Optional[str] = None
    ) -> ConversationSalesStage:
        stage_str = current_stage.value if isinstance(current_stage, ConversationSalesStage) else str(current_stage)
        if stage_str == ConversationSalesStage.NEW.value and user_message:
            return ConversationSalesStage.DISCOVERY
        new_stage_str, _ = cls.evaluate_transition(
            current_stage=stage_str,
            customer_requirements={},
            visit_readiness=0,
            visit_status=visit_status or "NOT_BOOKED",
            intent="GENERAL",
            recommended_count=1 if has_properties else 0
        )
        try:
            return ConversationSalesStage(new_stage_str)
        except Exception:
            return ConversationSalesStage.NEW

    @classmethod
    def evaluate_transition(
        cls,
        current_stage: str,
        customer_requirements: Dict[str, Any],
        visit_readiness: int,
        visit_status: str = "NOT_BOOKED",
        intent: str = "GENERAL",
        recommended_count: int = 0
    ) -> Tuple[str, str]:
        """
        Determines if state should advance based on structured signals.
        Returns (new_stage, reason_code).
        """
        if isinstance(current_stage, list):
            stage = str(current_stage[0] if current_stage else "NEW")
        elif hasattr(current_stage, "value"):
            stage = str(current_stage.value)
        else:
            stage = str(current_stage or "NEW")
        stage = stage.upper()
        req = customer_requirements or {}

        # 0. Opt-Out intent (highest priority)
        if intent == "OPT_OUT":
            return ConversationSalesStage.OPTED_OUT.value, "CUSTOMER_OPTED_OUT"

        if stage == ConversationSalesStage.OPTED_OUT.value:
            if intent in ("RE_ENGAGE", "RETURNING_GREETING", "DISCOVER_REQUIREMENT", "PRICE_INQUIRY"):
                return ConversationSalesStage.NEW.value, "OPTED_OUT_CUSTOMER_REENGAGED"
            return ConversationSalesStage.OPTED_OUT.value, "MAINTAINING_OPTED_OUT"

        # 1. Booking confirmed
        if visit_status == "BOOKED":
            return ConversationSalesStage.VISIT_BOOKED.value, "VISIT_CONFIRMED"

        # 2. Agreed to visit / negotiating slot
        if intent in ("AGREE_VISIT", "BOOK_VISIT", "VISIT_NEGOTIATING"):
            return ConversationSalesStage.VISIT_NEGOTIATING.value, "CUSTOMER_AGREED_TO_VISIT"

        # 3. Already booked or completed
        if stage in (ConversationSalesStage.VISIT_BOOKED.value, ConversationSalesStage.VISIT_COMPLETED.value):
            return stage, "MAINTAINING_VISIT_STATE"

        # 4. Visit Pitched
        if intent == "PITCH_SITE_VISIT" or visit_readiness >= 75:
            if stage in (
                ConversationSalesStage.QUALIFIED.value,
                ConversationSalesStage.PROPERTY_RECOMMENDED.value,
                ConversationSalesStage.ENGAGED.value
            ):
                return ConversationSalesStage.VISIT_PITCHED.value, "READINESS_THRESHOLD_MET"

        # 5. Engaged (asking detailed questions, floor plans, pricing)
        if intent in ("PRICE_INQUIRY", "AMENITIES_INQUIRY", "FLOOR_PLAN_REQUEST", "BROCHURE_REQUEST"):
            if stage in (ConversationSalesStage.PROPERTY_RECOMMENDED.value, ConversationSalesStage.QUALIFIED.value):
                return ConversationSalesStage.ENGAGED.value, "CUSTOMER_ENGAGING_DETAILS"

        # 6. Property Recommended
        if recommended_count > 0 and stage in (ConversationSalesStage.NEW.value, ConversationSalesStage.DISCOVERY.value, ConversationSalesStage.QUALIFIED.value):
            return ConversationSalesStage.PROPERTY_RECOMMENDED.value, "INVENTORY_RECOMMENDED"

        # 7. Qualified check (budget or bhk or location known)
        bhk_known = bool(req.get("bhk"))
        budget_known = bool(req.get("budget_max"))
        loc_known = bool(req.get("locality") or req.get("city"))

        if (budget_known and (bhk_known or loc_known)) or (bhk_known and loc_known):
            if stage in (ConversationSalesStage.NEW.value, ConversationSalesStage.DISCOVERY.value):
                return ConversationSalesStage.QUALIFIED.value, "CORE_REQUIREMENTS_CAPTURED"

        # 8. Discovery check (any requirement provided)
        if (bhk_known or budget_known or loc_known) and stage == ConversationSalesStage.NEW.value:
            return ConversationSalesStage.DISCOVERY.value, "DISCOVERY_IN_PROGRESS"

        return stage, "NO_TRANSITION"
