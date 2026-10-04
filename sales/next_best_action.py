"""
Next Best Action Engine — evaluates customer intent, requirements, sales memory,
and readiness signals to determine the most effective next sales milestone.
"""

import re
import logging
from typing import Dict, Any, List, Optional

from sales.state_machine import ConversationSalesStage, SalesStateMachine
from sales.objection_engine import ObjectionEngine
from sales.visit_pitch_engine import VisitPitchEngine, PitchStrategy

logger = logging.getLogger(__name__)


class NextBestActionEngine:
    """
    Determines structured next action, preventing repetitive questions and
    ensuring natural sales progression toward a booked site visit.
    """

    VISIT_AGREEMENT_PATTERNS = [
        r"\b(?:ok(?:ay)?|yes|sure|done|let'?s visit|want to visit|book (?:a )?visit|schedule (?:a )?visit|plan a visit)\b",
        r"\b(?:can i visit|will visit|ready to visit|see the property|visit (?:on )?(?:saturday|sunday|tomorrow|today|this weekend))\b",
        r"\b(?:interested in visiting|arrange a visit)\b",
        r"\b(?:ha(?:a|an)?\s*(?:ye|is)?\s*(?:week|weekend)?|ha chalega|chalega|theek hai|sahi hai|dekhna hai|visit karna hai|aana hai|aana chahta hu|kab chalna hai)\b",
    ]

    IDENTITY_PATTERNS = [
        r"\b(?:who are you|who'?s this|who is this|kon ho|kaun ho|aap kaun ho|kon ho aap|aap kon|kahan se bol rahe|kis baare me)\b",
        r"\b(?:what is aris|who is aris|what does aris do|about aris|altimet|altimet ai|what do you do)\b",
        r"\b(?:whatsapp bot|ai bot|bot service|automation service|how does this bot work)\b",
    ]

    OPT_OUT_PATTERNS = [
        r"\b(?:don'?t (?:message|text|contact|call|msg)|stop (?:messaging|texting|messages|sending)|unsubscribe|leave me alone|do not (?:message|contact|call)|untill? i (?:msg|message|text)|no more messages|stop it)\b",
        r"\b(?:msg mat karo|message mat bhejo|mat bhejo|nahi chahiye|pareshan mat karo|aage se mat bhejna|ab message mat karna|mat karo message|call mat karo|aage se koi message|message mat karo)\b",
    ]

    GREETING_PATTERNS = [
        r"^(?:hi+|hello+|hey+|hii+|hola|namaste|good\s*(?:morning|afternoon|evening))\b"
    ]

    def __init__(self):
        self.objection_engine = ObjectionEngine()
        self.pitch_engine = VisitPitchEngine()

    def determine(
        self,
        customer_memory: Any,
        sales_memory: Any,
        recent_messages: List[Dict[str, Any]],
        current_message: str,
        property_context: str = "",
        visit_readiness_score: int = 0
    ) -> Dict[str, Any]:
        """
        Computes structured Next Best Action metadata.
        """
        text = (current_message or "").strip()
        normalized_text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
        lower = normalized_text.lower()

        req = getattr(customer_memory, "requirements", {}) or {}
        bhk = req.get("bhk")
        budget = req.get("budget_max")
        loc = req.get("locality") or req.get("city")
        purpose = req.get("purpose")

        stage = getattr(sales_memory, "sales_stage", "NEW")
        v_status = getattr(sales_memory, "visit_status", "NOT_BOOKED")
        recommended = getattr(customer_memory, "recommended_properties", []) or []

        # 0. Detect Opt-Out / DND ("Don't message me", "stop", "mat bhejo")
        if any(re.search(pat, lower, re.IGNORECASE) for pat in self.OPT_OUT_PATTERNS):
            return {
                "intent": "OPT_OUT",
                "sales_stage": "OPTED_OUT",
                "lead_temperature": "COLD",
                "objection": "UNSUBSCRIBE",
                "visit_readiness": 0,
                "next_best_action": "CONFIRM_OPT_OUT",
                "recommended_property": None,
                "reason_code": "CUSTOMER_REQUESTED_OPT_OUT"
            }

        # 0b. Detect Re-engagement from previously opted-out customer
        if stage == "OPTED_OUT":
            return {
                "intent": "RE_ENGAGE",
                "sales_stage": "NEW",
                "lead_temperature": "WARM",
                "objection": None,
                "visit_readiness": 20,
                "next_best_action": "RE_ENGAGE_WELCOME",
                "recommended_property": recommended[0] if recommended else None,
                "reason_code": "OPTED_OUT_CUSTOMER_REENGAGED"
            }

        # 0c. Detect Casual Greeting mid-conversation (avoid repeated bulky property/cab pitches)
        is_simple_greeting = any(re.search(pat, lower.strip(), re.IGNORECASE) for pat in self.GREETING_PATTERNS) and len(lower.strip().split()) <= 3
        if is_simple_greeting and (recommended or stage in ("PROPERTY_RECOMMENDED", "PROPERTY_SHORTLISTED", "VISIT_PITCHED", "ENGAGED")):
            return {
                "intent": "RETURNING_GREETING",
                "sales_stage": stage,
                "lead_temperature": "WARM",
                "objection": None,
                "visit_readiness": visit_readiness_score,
                "next_best_action": "CASUAL_CHECK_IN",
                "recommended_property": recommended[0] if recommended else None,
                "reason_code": "RETURNING_CUSTOMER_GREETING"
            }

        # 1. Detect Identity Query ("Who are you" / "What is ARIS" / "Altimet AI")
        if any(re.search(pat, lower, re.IGNORECASE) for pat in self.IDENTITY_PATTERNS):
            wants_contact = any(k in lower for k in ["contact", "email", "phone", "number", "reach", "call", "details", "team", "support", "hire"])
            return {
                "intent": "IDENTITY_INQUIRY",
                "sales_stage": stage,
                "lead_temperature": "WARM",
                "objection": None,
                "visit_readiness": visit_readiness_score,
                "next_best_action": "INTRODUCE_ADVISOR",
                "give_contact_info": wants_contact,
                "recommended_property": recommended[0] if recommended else None,
                "reason_code": "CUSTOMER_ASKED_IDENTITY"
            }

        # 2. Detect Human Advisor Request
        if any(k in lower for k in ["speak to human", "talk to agent", "call me", "human agent", "talk to human", "connect with executive"]):
            return {
                "intent": "HUMAN_HANDOFF",
                "sales_stage": stage,
                "lead_temperature": "HOT",
                "objection": None,
                "visit_readiness": visit_readiness_score,
                "next_best_action": "HANDOFF_HUMAN",
                "recommended_property": recommended[0] if recommended else None,
                "reason_code": "CUSTOMER_REQUESTED_HUMAN"
            }

        # 3. Check for Visit Agreement (BOOKING & SLOT NEGOTIATION MODE)
        agreed_to_visit = any(re.search(pat, lower, re.IGNORECASE) for pat in self.VISIT_AGREEMENT_PATTERNS)
        if agreed_to_visit or stage == ConversationSalesStage.VISIT_NEGOTIATING.value:
            # Check if customer gave a specific day/time
            has_time = any(w in lower for w in [
                "tomorrow", "today", "saturday", "sunday", "monday", "tuesday", "wednesday", "thursday", "friday",
                "kal", "parso", "shanivar", "ravivar", "morning", "afternoon", "evening",
                "am", "pm", "o'clock", "subah", "dopahar", "shaam", "10", "11", "12", "1", "2", "3", "4", "5", "6"
            ]) and not (lower in ["ha ye week", "ha", "haan", "chalega", "is week", "this week"])

            has_address = any(w in lower for w in ["wardha", "mihan", "besa", "nagar", "road", "street", "plot", "flat", "pune", "nagpur", "address", "ghr", "ghar", "home"])
            
            if not has_time:
                action = "NEGOTIATE_VISIT_TIME"
            elif not has_address:
                action = "OFFER_CAB"
            else:
                action = "CONFIRM_VISIT"

            return {
                "intent": "VISIT_BOOKING",
                "sales_stage": ConversationSalesStage.VISIT_NEGOTIATING.value,
                "lead_temperature": "HOT",
                "objection": None,
                "visit_readiness": max(visit_readiness_score, 85),
                "next_best_action": action,
                "recommended_property": recommended[0] if recommended else None,
                "reason_code": "CUSTOMER_AGREED_TO_VISIT"
            }

        # 3. Detect Objections
        detected_objection = self.objection_engine.detect_objection(text)
        if detected_objection:
            action_map = {
                "PRICE": "HANDLE_PRICE_OBJECTION",
                "TOO_EXPENSIVE": "HANDLE_PRICE_OBJECTION",
                "NEED_TO_THINK": "HANDLE_THINKING_OBJECTION",
                "SPOUSE_NOT_CONVINCED": "HANDLE_FAMILY_OBJECTION",
                "LOCATION": "HANDLE_LOCATION_OBJECTION",
                "TRUST": "HANDLE_TRUST_OBJECTION",
                "POSSESSION": "HANDLE_POSSESSION_OBJECTION",
                "FINANCING": "HANDLE_FINANCING_OBJECTION",
                "NO_TIME": "OFFER_CAB",
                "WANT_BROCHURE": "SEND_BROCHURE",
                "WANT_MORE_OPTIONS": "RECOMMEND_PROPERTY",
                "WANT_TO_COMPARE": "COMPARE_PROPERTIES"
            }
            return {
                "intent": "OBJECTION",
                "sales_stage": stage,
                "lead_temperature": "HOT" if detected_objection in ("PRICE", "SPOUSE_NOT_CONVINCED") else "WARM",
                "objection": detected_objection,
                "visit_readiness": visit_readiness_score,
                "next_best_action": action_map.get(detected_objection, "HANDLE_PRICE_OBJECTION"),
                "recommended_property": recommended[0] if recommended else None,
                "reason_code": f"OBJECTION_{detected_objection}"
            }

        # 4. Brochure Request
        if any(w in lower for w in ["brochure", "pdf", "master plan", "floor plan", "specifications"]):
            return {
                "intent": "BROCHURE_REQUEST",
                "sales_stage": stage,
                "lead_temperature": "WARM",
                "objection": None,
                "visit_readiness": max(visit_readiness_score, 50),
                "next_best_action": "SEND_BROCHURE",
                "recommended_property": recommended[0] if recommended else None,
                "reason_code": "CUSTOMER_REQUESTED_COLLATERAL"
            }

        # 5. Specific Property Inquiry (Price, Amenities, Possession)
        if any(w in lower for w in ["how much", "price", "cost", "emi", "amenities", "pool", "gym", "carpet", "possession"]):
            # If already qualified and high readiness, answer AND pitch visit
            if visit_readiness_score >= 65:
                pitch_strat = self.pitch_engine.select_strategy(req, [], previous_pitches=0)
                return {
                    "intent": "PROPERTY_SPECIFIC_INQUIRY",
                    "sales_stage": ConversationSalesStage.VISIT_PITCHED.value,
                    "lead_temperature": "HOT",
                    "objection": None,
                    "visit_readiness": visit_readiness_score,
                    "next_best_action": "PITCH_SITE_VISIT",
                    "pitch_strategy": pitch_strat.value,
                    "recommended_property": recommended[0] if recommended else None,
                    "reason_code": "HIGH_READINESS_CONVERT_INQUIRY"
                }
            return {
                "intent": "PROPERTY_SPECIFIC_INQUIRY",
                "sales_stage": ConversationSalesStage.ENGAGED.value,
                "lead_temperature": "WARM",
                "objection": None,
                "visit_readiness": visit_readiness_score,
                "next_best_action": "ANSWER_PROPERTY_QUESTION",
                "recommended_property": recommended[0] if recommended else None,
                "reason_code": "INVENTORY_DETAILS_REQUESTED"
            }

        # 6. If Qualified (Budget + Location + BHK known) -> Recommend or Pitch
        if bhk and (budget or loc):
            if not recommended:
                return {
                    "intent": "REQUIREMENT_MATCHED",
                    "sales_stage": ConversationSalesStage.PROPERTY_RECOMMENDED.value,
                    "lead_temperature": "WARM",
                    "objection": None,
                    "visit_readiness": visit_readiness_score,
                    "next_best_action": "RECOMMEND_PROPERTY",
                    "recommended_property": None,
                    "reason_code": "REQUIREMENTS_SUFFICIENT_FOR_RECOMMENDATION"
                }
            else:
                # Already recommended, customer is asking general question -> pitch visit
                pitch_strat = self.pitch_engine.select_strategy(req, [], previous_pitches=1)
                return {
                    "intent": "QUALIFIED_EXPLORATION",
                    "sales_stage": ConversationSalesStage.VISIT_PITCHED.value,
                    "lead_temperature": "HOT" if visit_readiness_score >= 60 else "WARM",
                    "objection": None,
                    "visit_readiness": visit_readiness_score,
                    "next_best_action": "PITCH_SITE_VISIT",
                    "pitch_strategy": pitch_strat.value,
                    "recommended_property": recommended[0],
                    "reason_code": "INVITE_QUALIFIED_LEAD_FOR_TOUR"
                }

        # 7. Progressive Discovery — ask for ONE missing piece naturally
        if not loc:
            return {
                "intent": "DISCOVERY",
                "sales_stage": ConversationSalesStage.DISCOVERY.value,
                "lead_temperature": "COLD",
                "objection": None,
                "visit_readiness": visit_readiness_score,
                "next_best_action": "ASK_LOCATION",
                "recommended_property": None,
                "reason_code": "LOCATION_MISSING"
            }
        elif not bhk:
            return {
                "intent": "DISCOVERY",
                "sales_stage": ConversationSalesStage.DISCOVERY.value,
                "lead_temperature": "COLD",
                "objection": None,
                "visit_readiness": visit_readiness_score,
                "next_best_action": "ASK_BHK",
                "recommended_property": None,
                "reason_code": "BHK_MISSING"
            }
        elif not budget:
            return {
                "intent": "DISCOVERY",
                "sales_stage": ConversationSalesStage.DISCOVERY.value,
                "lead_temperature": "COLD",
                "objection": None,
                "visit_readiness": visit_readiness_score,
                "next_best_action": "ASK_BUDGET",
                "recommended_property": None,
                "reason_code": "BUDGET_MISSING"
            }

        return {
            "intent": "GENERAL_INQUIRY",
            "sales_stage": stage,
            "lead_temperature": "WARM",
            "objection": None,
            "visit_readiness": visit_readiness_score,
            "next_best_action": "DISCOVER_REQUIREMENT",
            "recommended_property": None,
            "reason_code": "DEFAULT_CONSULTATIVE"
        }

    @classmethod
    def determine_nba(
        cls,
        current_stage: Any,
        customer_memory: Any,
        sales_memory: Any,
        last_user_message: str,
        recent_messages: Optional[List[Dict[str, Any]]] = None,
        property_context: str = "",
        visit_readiness_score: int = 0
    ) -> Dict[str, Any]:
        engine = cls()
        res = engine.determine(
            customer_memory=customer_memory,
            sales_memory=sales_memory,
            recent_messages=recent_messages or [],
            current_message=last_user_message,
            property_context=property_context,
            visit_readiness_score=visit_readiness_score
        )
        nba = res.get("next_best_action")
        if nba in ("ASK_LOCATION", "ASK_BHK", "ASK_BUDGET", "DISCOVER_REQUIREMENT"):
            res["action_type"] = "ASK_REQUIREMENT"
        elif nba in ("BOOK_SITE_VISIT", "OFFER_CAB", "CONFIRM_VISIT"):
            res["action_type"] = "COLLECT_VISIT_DETAILS"
        elif nba in (
            "HANDLE_PRICE_OBJECTION", "HANDLE_FAMILY_OBJECTION", "HANDLE_THINKING_OBJECTION",
            "HANDLE_LOCATION_OBJECTION", "HANDLE_TRUST_OBJECTION", "HANDLE_POSSESSION_OBJECTION",
            "HANDLE_FINANCING_OBJECTION"
        ):
            res["action_type"] = "RESOLVE_OBJECTION"
        elif nba == "PITCH_SITE_VISIT":
            res["action_type"] = "PITCH_SITE_VISIT"
        else:
            res["action_type"] = nba

        res["objection_category"] = res.get("objection")
        res["booking_mode_active"] = (
            res.get("intent") == "VISIT_BOOKING" or
            nba in ("BOOK_SITE_VISIT", "OFFER_CAB", "CONFIRM_VISIT")
        )
        if res.get("objection"):
            strat = engine.objection_engine.get_strategy(res["objection"])
            res["talking_points"] = [strat.get("guideline", "")]
        else:
            res["talking_points"] = []
        return res
