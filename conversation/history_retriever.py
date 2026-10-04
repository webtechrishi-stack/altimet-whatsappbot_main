"""
Relevant Historical Message Retriever.
Retrieves older salient conversation turns (past stated requirements, objections, questions)
without loading the entire conversation history into Gemini context.
Strictly kept separate from Property RAG.
"""

import re
import logging
from typing import List, Dict, Any, Optional
from database import DB

logger = logging.getLogger(__name__)


class HistoryRetriever:
    """
    Selectively retrieves relevant older conversation turns to supplement recent history.
    """

    KEYWORD_PATTERNS = [
        r"\b(?:budget|lakh|lakhs|cr|crore|stretch)\b",
        r"\b(?:bhk|bedroom|flat|apartment|villa)\b",
        r"\b(?:location|locality|mihan|manish nagar|besa|wardha|kharadi|baner)\b",
        r"\b(?:possession|ready|under construction|rera)\b",
        r"\b(?:visit|site visit|see property|appointment)\b",
        r"\b(?:expensive|discount|price|cost|emi|loan)\b",
        r"\b(?:brochure|floor plan|amenities)\b"
    ]

    def __init__(self):
        self.msg_repo = DB.messages

    def retrieve_relevant_turns(
        self,
        conversation_id: str,
        current_query: str = "",
        recent_message_ids: Optional[List[str]] = None,
        max_turns: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Finds up to max_turns historical messages from outside recent_message_ids
        that match key real estate topics or query terms.
        """
        if not conversation_id:
            return []

        recent_ids = set(recent_message_ids or [])

        # Fetch up to 50 older messages
        all_msgs = self.msg_repo.get_last_messages(conversation_id=conversation_id, limit=50)
        older_candidates = [m for m in all_msgs if m.get("message_id") not in recent_ids]

        if not older_candidates:
            return []

        scored_turns = []
        query_words = set(re.findall(r"\w+", (current_query or "").lower()))

        for m in older_candidates:
            text = (m.get("text") or "").lower()
            if len(text) < 5:
                continue

            score = 0
            # Check domain keyword patterns
            for pat in self.KEYWORD_PATTERNS:
                if re.search(pat, text, re.IGNORECASE):
                    score += 2

            # Check overlap with current user question
            msg_words = set(re.findall(r"\w+", text))
            overlap = len(query_words.intersection(msg_words))
            score += overlap * 3

            # Prioritize customer statements
            if m.get("sender_type") == "CUSTOMER" or m.get("role") == "user":
                score += 1

            if score > 0:
                scored_turns.append((score, m))

        # Sort descending by relevance score, then take up to max_turns
        scored_turns.sort(key=lambda x: x[0], reverse=True)
        selected = [item[1] for item in scored_turns[:max_turns]]

        # Re-sort chronologically
        return sorted(selected, key=lambda x: str(x.get("created_at", "")))
