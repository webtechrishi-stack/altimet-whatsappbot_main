"""
Conversation Context Builder.
Constructs a compact, personalized, grounded context payload for Gemini and ARIS sales agent.
"""

import logging
from typing import Dict, Any, List, Optional

from database import DB
from conversation.models import ConversationContext, CustomerMemory, SalesMemory
from conversation.memory_service import CustomerMemoryService
from conversation.history_retriever import HistoryRetriever
from conversation.summarizer import ConversationSummarizer

logger = logging.getLogger(__name__)


class ConversationContextBuilder:
    """
    Assembles the structured context required for personalized sales generation.
    """

    def __init__(self):
        self.memory_service = CustomerMemoryService()
        self.history_retriever = HistoryRetriever()
        self.summarizer = ConversationSummarizer()
        self.msg_repo = DB.messages
        self.sales_memory_repo = DB.sales_memory
        self.property_repo = DB.properties

    def build(
        self,
        lead_id: str,
        conversation_id: str,
        current_message: str,
        next_best_action_data: Optional[Dict[str, Any]] = None,
        rag_context: str = "",
        focused_property_id: Optional[str] = None
    ) -> ConversationContext:
        """
        Builds complete ConversationContext object without loading full unbounded history.
        """
        # 1. Customer Memory
        customer_mem = self.memory_service.get_or_create_memory(lead_id)

        # 2. Sales Memory
        raw_sm = self.sales_memory_repo.get_by_lead_id(lead_id)
        if raw_sm:
            try:
                sales_mem = SalesMemory.from_dict(raw_sm)
            except Exception:
                sales_mem = SalesMemory(lead_id=str(lead_id), conversation_id=str(conversation_id))
        else:
            sales_mem = SalesMemory(lead_id=str(lead_id), conversation_id=str(conversation_id))

        # 3. Recent 10 messages
        recent_raw = self.msg_repo.get_last_messages(conversation_id=conversation_id, limit=10)
        recent_ids = [m.get("message_id") for m in recent_raw if m.get("message_id")]

        # 4. Relevant Historical Messages
        relevant_history = self.history_retriever.retrieve_relevant_turns(
            conversation_id=conversation_id,
            current_query=current_message,
            recent_message_ids=recent_ids,
            max_turns=3
        )

        # 5. Conversation Summary
        summary = self.summarizer.update_summary_if_needed(
            conversation_id=conversation_id,
            customer_memory=customer_mem,
            sales_memory=sales_mem,
            recent_messages=recent_raw
        )

        # 6. Verified Property Context
        property_context = self._build_property_context(
            customer_memory=customer_mem,
            focused_property_id=focused_property_id
        )

        nba = next_best_action_data or {
            "intent": "GENERAL_INQUIRY",
            "next_best_action": "DISCOVER_REQUIREMENT",
            "sales_stage": sales_mem.sales_stage,
            "lead_temperature": sales_mem.lead_temperature,
            "visit_readiness": sales_mem.visit_readiness_score,
            "objection": None,
            "recommended_property": None,
            "reason_code": "DEFAULT_FLOW"
        }

        return ConversationContext(
            customer_memory=customer_mem,
            sales_memory=sales_mem,
            recent_messages=recent_raw,
            relevant_history=relevant_history,
            property_context=property_context,
            rag_context=rag_context,
            next_best_action=nba,
            conversation_summary=summary,
            current_message=current_message
        )

    def _build_property_context(
        self,
        customer_memory: CustomerMemory,
        focused_property_id: Optional[str] = None
    ) -> str:
        """Constructs concise verified property specifications."""
        parts = []

        # Focused property if specified
        if focused_property_id:
            prop = self.property_repo.get_by_id(focused_property_id)
            if prop:
                parts.append(
                    f"PRIMARY FOCUSED PROPERTY:\n"
                    f"- Name: {prop.get('title')} ({prop.get('id')})\n"
                    f"- Location: {prop.get('locality')}, {prop.get('city')}\n"
                    f"- Configuration: {prop.get('bhk')}, Carpet Area: {prop.get('carpet_area')}\n"
                    f"- Price: {prop.get('price_display')}\n"
                    f"- Possession: {prop.get('possession_status')}\n"
                    f"- Amenities: {', '.join(prop.get('amenities', []))}\n"
                    f"- Highlights: {', '.join(prop.get('highlights', []))}"
                )

        # Top matching inventory based on requirements
        req = customer_memory.requirements
        city = req.get("city")
        bhk = req.get("bhk")
        b_max = req.get("budget_max")

        matching = self.property_repo.list_properties(
            city=city,
            bhk=bhk,
            max_budget_lakhs=b_max,
            limit=4
        )
        if matching:
            lines = ["MATCHING VERIFIED INVENTORY:"]
            for p in matching:
                if focused_property_id and p.get("id") == focused_property_id:
                    continue
                lines.append(f"• {p.get('title')} | {p.get('bhk')} in {p.get('locality')}, {p.get('city')} | {p.get('price_display')} [ID: {p.get('id')}]")
            if len(lines) > 1:
                parts.append("\n".join(lines))

        return "\n\n".join(parts) if parts else "Refer to standard verified project catalog."
