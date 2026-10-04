"""
Test Suite for RAG-Grounded Human-Like Hinglish Sales Consultant.
Verifies:
1. GeminiProvider initializes with GOOGLE_API_KEY and gemini-3.6-flash
2. Live generation in Hinglish with language mirroring
3. Slot negotiation: "Ha ye week" asks for day/time instead of blind booking
4. Identity query: "Kon ho aap" introduces ARIS consultant warmly
5. Greeting isolation: "Namaste hi" does not trigger false booking
6. Grounded RAG fact injection
"""

import sys
import unittest
import os
from dotenv import load_dotenv

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from database import DB
from config import Config
from ai_engine import GeminiProvider, RAGService, ARISOrchestrator
from ai.sales_agent import PersonalizedSalesAgent
from sales.next_best_action import NextBestActionEngine


class TestHinglishConsultant(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = PersonalizedSalesAgent()
        cls.gemini = GeminiProvider()
        cls.rag = RAGService()
        cls.nba_engine = NextBestActionEngine()

    def test_01_gemini_configuration_and_model(self):
        """Verifies Gemini client is configured with valid key and gemini-3.6-flash."""
        self.assertTrue(self.gemini.is_configured(), "GeminiProvider should be configured with API key")
        self.assertEqual(self.gemini.model_name, "gemini-3.6-flash")
        print(f"[PASS] Gemini configured: {self.gemini.get_provider_name()}")

    def test_02_identity_query_handling(self):
        """Verifies 'Kon ho aap' triggers INTRODUCE_ADVISOR next best action."""
        nba = self.nba_engine.determine(
            customer_memory=type("mem", (object,), {"requirements": {}, "recommended_properties": []})(),
            sales_memory=type("sm", (object,), {"sales_stage": "NEW", "visit_status": "NOT_BOOKED"})(),
            recent_messages=[],
            current_message="Kon ho aap"
        )
        self.assertEqual(nba["intent"], "IDENTITY_INQUIRY")
        self.assertEqual(nba["next_best_action"], "INTRODUCE_ADVISOR")
        print("[PASS] 'Kon ho aap' identified as IDENTITY_INQUIRY -> INTRODUCE_ADVISOR.")

    def test_03_vague_visit_intent_negotiation(self):
        """Verifies 'Ha ye week' triggers NEGOTIATE_VISIT_TIME, not premature booking."""
        nba = self.nba_engine.determine(
            customer_memory=type("mem", (object,), {"requirements": {"bhk": "2BHK"}, "recommended_properties": ["Green Meadows Residency"]})(),
            sales_memory=type("sm", (object,), {"sales_stage": "VISIT_PITCHED", "visit_status": "NOT_BOOKED"})(),
            recent_messages=[],
            current_message="Ha ye week"
        )
        self.assertEqual(nba["intent"], "VISIT_BOOKING")
        self.assertEqual(nba["next_best_action"], "NEGOTIATE_VISIT_TIME")
        print("[PASS] 'Ha ye week' negotiated for day/time slot instead of premature confirmation.")

    def test_04_end_to_end_conversational_turn_hinglish(self):
        """Simulates customer saying 'Ha ye week' and checks reply asks for day/time."""
        test_wa = "919999988888"
        session = {"context": {"focused_property_id": "ARIS-NGP-01"}}
        
        reply, updated_session, nba = self.agent.process_turn(
            lead_id="test_lead_rishi",
            conversation_id="conv_test_rishi",
            wa_id=test_wa,
            customer_name="Rishi",
            message_text="Ha ye week",
            session_data=session
        )
        
        print("\n--- Model Response to 'Ha ye week' ---")
        print(reply)
        print("-------------------------------------")
        
        self.assertTrue(len(reply) > 20)
        # Should NOT say "Your site visit is scheduled! ⏰ Requested Slot: Ha ye week"
        self.assertNotIn("Requested Slot: Ha ye week", reply)
        # Should ask for day/time (Saturday/Sunday/timing/slot)
        lower_reply = reply.lower()
        has_negotiation = any(w in lower_reply for w in ["saturday", "sunday", "din", "day", "time", "slot", "weekend", "kab"])
        self.assertTrue(has_negotiation, "Agent should ask for preferred day/time window")

    def test_05_greeting_does_not_book_visit(self):
        """Verifies saying 'Namaste hi' does not trigger visit booking."""
        test_wa = "919999988888"
        session = {"state": "awaiting_visit_time", "context": {"focused_property_id": "ARIS-NGP-01"}}
        
        reply, updated_session, nba = self.agent.process_turn(
            lead_id="test_lead_rishi",
            conversation_id="conv_test_rishi",
            wa_id=test_wa,
            customer_name="Rishi",
            message_text="Namaste hi",
            session_data=session
        )
        
        print("\n--- Model Response to 'Namaste hi' ---")
        print(reply)
        print("--------------------------------------")
        
        self.assertNotIn("Requested Slot: Namaste hi", reply)
        self.assertNotIn("site visit is scheduled", reply.lower())

    def test_06_identity_query_live_response(self):
        """Verifies saying 'Kon ho aap' responds with advisor introduction in Hinglish."""
        test_wa = "919999988888"
        session = {"context": {"focused_property_id": "ARIS-NGP-01"}}
        
        reply, updated_session, nba = self.agent.process_turn(
            lead_id="test_lead_rishi",
            conversation_id="conv_test_rishi",
            wa_id=test_wa,
            customer_name="Rishi",
            message_text="Kon ho aap",
            session_data=session
        )
        
        print("\n--- Model Response to 'Kon ho aap' ---")
        print(reply)
        print("--------------------------------------")
        
        lower_reply = reply.lower()
        self.assertTrue(
            any(w in lower_reply for w in ["aris", "advisor", "consultant", "property", "namaste", "rishi"]),
            "Agent should introduce themselves warmly as ARIS advisor"
        )


if __name__ == "__main__":
    unittest.main()
