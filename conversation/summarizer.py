"""
Incremental conversation summarizer for ARIS CRM.
Updates concise conversational milestones without expensive re-generation on every trivial turn.
"""

import logging
from typing import List, Dict, Any, Optional
from database import DB

logger = logging.getLogger(__name__)


class ConversationSummarizer:
    """
    Maintains compact, incremental conversation summaries.
    """

    def __init__(self):
        self.conv_repo = DB.conversations

    def update_summary_if_needed(
        self,
        conversation_id: str,
        customer_memory: Any,
        sales_memory: Any,
        recent_messages: List[Dict[str, Any]],
        force: bool = False
    ) -> str:
        """
        Updates summary if message count crossed interval threshold or key state changed.
        """
        if not conversation_id:
            return ""

        conv = self.conv_repo.get_by_id(conversation_id)
        current_summary = conv.get("summary", "") if conv else ""
        msg_count = conv.get("message_count", len(recent_messages)) if conv else len(recent_messages)

        # Only update every 6 messages unless forced
        if not force and current_summary and (msg_count % 6 != 0):
            return current_summary

        try:
            summary = self._build_incremental_summary(customer_memory, sales_memory, recent_messages)
            if summary and summary != current_summary:
                self.conv_repo.update_summary(conversation_id, summary)
            return summary or current_summary
        except Exception as ex:
            logger.warning(f"[SUMMARIZER] Failed to update summary: {ex}")
            return current_summary

    def _build_incremental_summary(
        self,
        customer_memory: Any,
        sales_memory: Any,
        recent_messages: List[Dict[str, Any]]
    ) -> str:
        parts = []

        # Customer Requirements
        if customer_memory:
            req = getattr(customer_memory, "requirements", {}) or {}
            bhk = req.get("bhk")
            loc = req.get("locality") or req.get("city")
            b_max = req.get("budget_max")
            purpose = req.get("purpose")

            c_parts = []
            if bhk:
                c_parts.append(f"{bhk}")
            if loc:
                c_parts.append(f"in {loc}")
            if b_max:
                c_parts.append(f"around ₹{b_max}L")
            if purpose == "END_USE":
                c_parts.append("for self-use")
            elif purpose == "INVESTMENT":
                c_parts.append("for investment")

            if c_parts:
                parts.append(f"Customer seeking {' '.join(c_parts)}.")

            # Properties discussed
            props = getattr(customer_memory, "recommended_properties", []) or []
            if props:
                parts.append(f"Discussed properties: {', '.join(props[:2])}.")

            # Objections
            objs = [o.get("type") for o in getattr(customer_memory, "objections", []) if o.get("status") == "OPEN"]
            if objs:
                parts.append(f"Main concerns: {', '.join(objs)}.")

        # Sales Milestones
        if sales_memory:
            stage = getattr(sales_memory, "sales_stage", "NEW")
            v_status = getattr(sales_memory, "visit_status", "NOT_BOOKED")
            if v_status == "BOOKED":
                parts.append("Site visit is CONFIRMED.")
            elif stage in ("VISIT_PITCHED", "VISIT_NEGOTIATING"):
                parts.append("Site visit pitched; awaiting date/time confirmation.")
            elif stage == "QUALIFIED":
                parts.append("Lead qualified with budget and location matched.")

        if not parts:
            # Fallback to last customer message text
            for m in reversed(recent_messages):
                if m.get("sender_type") == "CUSTOMER" or m.get("role") == "user":
                    return f"Customer inquired: {m.get('text', '')[:120]}"

        return " ".join(parts)
