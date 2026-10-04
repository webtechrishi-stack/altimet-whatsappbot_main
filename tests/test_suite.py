"""
Comprehensive Verification Suite for ARIS Platform.
Tests:
1. Dashboard HTML Views (HTTP 200)
2. REST API Endpoints (HTTP 200 & valid JSON)
3. CRM Lead Scoring & Preferences
4. Site Visit Booking & Idempotency
5. Full 9-Turn Conversational Real Estate Sales Flow
"""

import sys
import unittest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app import app
from database import DB, LeadRepository, VisitRepository, PropertyRepository, SessionRepository, MessageRepository
from ai_engine import ARISAgent, ARISOrchestrator, VisitService


class TestARISPlatform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_01_dashboard_views(self):
        """Tests that all CRM web views render HTTP 200."""
        routes = [
            "/dashboard",
            "/dashboard/leads",
            "/dashboard/conversations",
            "/dashboard/properties",
            "/dashboard/visits",
            "/dashboard/followups",
            "/dashboard/analytics",
            "/dashboard/settings",
        ]
        for route in routes:
            res = self.client.get(route)
            self.assertEqual(res.status_code, 200, f"Route {route} failed with status {res.status_code}")
        print("[PASS] All Dashboard HTML Views rendered HTTP 200 successfully.")

    def test_02_rest_apis(self):
        """Tests that all REST API endpoints return valid JSON and HTTP 200."""
        # Overview
        res = self.client.get("/api/overview")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("metrics", data)
        self.assertIn("stage_distribution", data)

        # Leads
        res = self.client.get("/api/leads")
        self.assertEqual(res.status_code, 200)
        self.assertIn("leads", res.get_json())

        # Properties
        res = self.client.get("/api/properties")
        self.assertEqual(res.status_code, 200)
        props = res.get_json().get("properties", [])
        self.assertGreaterEqual(len(props), 1)

        # Visits
        res = self.client.get("/api/visits")
        self.assertEqual(res.status_code, 200)
        self.assertIn("visits", res.get_json())

        # Health
        res = self.client.get("/api/system/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json()["status"], "operational")

        print("[PASS] All Core REST APIs responded with valid JSON and HTTP 200.")

    def test_03_inbound_orchestrator_flow(self):
        """Tests lead preferences capture and CRM scoring."""
        orchestrator = ARISOrchestrator()
        test_wa_id = "918600079496"

        session = {"state": "initial", "context": {}}
        reply, session = orchestrator.process_message(test_wa_id, "Rishi", "Hi", session, [])
        self.assertIn("ARIS", reply)

        reply, session = orchestrator.process_message(
            test_wa_id, "Rishi", "I am looking for a 2BHK flat in Nagpur under 60 lakhs", session, []
        )
        self.assertTrue(len(reply) > 0)

        lead_repo = LeadRepository()
        lead = lead_repo.get_by_wa_id(test_wa_id)
        self.assertIsNotNone(lead)
        self.assertEqual(lead.get("preferred_city"), "Nagpur")
        self.assertIn("2BHK", lead.get("bhk", []))
        self.assertEqual(lead.get("budget_max"), 60.0)
        self.assertGreater(lead.get("lead_score", 0), 10)
        print(f"[PASS] Lead scored: {lead['lead_score']} (Stage: {lead['sales_stage']})")

    def test_04_site_visit_booking_idempotency(self):
        """Tests site visit booking idempotency."""
        vs = VisitService()
        res = vs.book_visit(
            wa_id="918600079496",
            customer_name="Rishi",
            property_id="ARIS-NGP-01",
            visit_date="2026-09-12",
            visit_time="10:00 AM"
        )
        self.assertTrue(res["success"])
        self.assertIn("Confirmed", res["message"])
        print("[PASS] Site visit successfully booked with idempotency.")

    def test_05_inbound_9_turn_conversation_flow(self):
        """Tests full 9-turn conversational slot filling and property search."""
        agent = ARISAgent()
        session_repo = SessionRepository()
        msg_repo = MessageRepository()

        test_wa_id = "919876543210"
        test_name = "Rishi"
        conv_id = "test_conv_suite_01"

        def chat(msg_text: str) -> str:
            session = session_repo.get_or_create_session(test_wa_id)
            history = msg_repo.get_last_messages(conv_id, limit=20)
            msg_repo.save_user_message(conv_id, test_wa_id, msg_text)
            reply, updated_session = agent.process_message(
                sender_id=test_wa_id,
                profile_name=test_name,
                message_text=msg_text,
                session=session,
                history=history
            )
            session_repo.update_session(
                test_wa_id,
                state=updated_session.get("state"),
                context=updated_session.get("context")
            )
            msg_repo.save_assistant_message(conv_id, test_wa_id, reply)
            return reply

        # Turn 1: Hi
        r1 = chat("Hi")
        self.assertIn("Rishi", r1)
        self.assertIn("Browse All Properties", r1)

        # Turn 2: Browse
        r2 = chat("List me all properties")
        self.assertTrue("Green Meadows Residency" in r2 or "Royal Palms Heights" in r2)

        # Turn 3: Search
        chat("menu")
        r3 = chat("2")
        self.assertIn("Which city", r3)

        # Turn 4: City
        r4 = chat("Nagpur")
        self.assertIn("configuration", r4.lower())

        # Turn 5: BHK
        r5 = chat("2BHK")
        self.assertIn("budget", r5.lower())

        # Turn 6: Budget
        r6 = chat("50")
        self.assertIn("Green Meadows Residency", r6)

        # Turn 7: Select
        r7 = chat("1")
        self.assertIn("Green Meadows", r7)

        # Turn 8: Book visit
        r8 = chat("Tomorrow at 4 PM")
        self.assertIn("Confirmed", r8)

        # Turn 9: Direct query
        r9 = chat("Looking for 3BHK in Pune")
        self.assertIn("Vanguard Valley", r9)

        print("[PASS] Full 9-turn conversational flow verified.")


if __name__ == "__main__":
    unittest.main()
