"""
Targeted Verification Suite for:
1. Opt-Out / DND Detection & MongoDB Persistence
2. Re-engagement Handling
3. Casual Returning Greeting (Preventing Repeated Property/Cab Pitches)
4. Smart 6-12h Follow-Up Eligibility, Quiet Hours, and Stage Nudges
5. Message Length & Tone Verification (Crisp & Concise)
"""

import sys
import unittest
from datetime import datetime, timezone, timedelta

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config import Config
from database import DB, Lead, Conversation, Message
from sales.state_machine import ConversationSalesStage, SalesStateMachine
from sales.next_best_action import NextBestActionEngine
from ai.prompts import ARISPromptManager
from conversation.models import ConversationContext, CustomerMemory, SalesMemory
from ai_engine import FollowUpService
from ai.sales_agent import PersonalizedSalesAgent


class TestSmartFollowUpAndOptOut(unittest.TestCase):

    def setUp(self):
        self.nba = NextBestActionEngine()
        self.fup_svc = FollowUpService()

    def test_01_opt_out_detection_patterns(self):
        """Test regex pattern detection for various English and Hinglish opt-out requests."""
        opt_out_phrases = [
            "Don’t message me again",
            "Untill i msg first",
            "Stop messaging me please",
            "unsubscribe",
            "msg mat karo",
            "message mat bhejo",
            "aage se mat bhejna",
            "nahi chahiye property",
            "leave me alone",
            "do not contact me"
        ]
        dummy_cm = CustomerMemory(lead_id="test_lead_1", requirements={"city": "Nagpur"})
        dummy_sm = SalesMemory(lead_id="test_lead_1", sales_stage="PROPERTY_RECOMMENDED")

        for phrase in opt_out_phrases:
            res = self.nba.determine(
                customer_memory=dummy_cm,
                sales_memory=dummy_sm,
                recent_messages=[],
                current_message=phrase
            )
            self.assertEqual(res["intent"], "OPT_OUT", f"Failed for phrase: '{phrase}'")
            self.assertEqual(res["next_best_action"], "CONFIRM_OPT_OUT")
            self.assertEqual(res["sales_stage"], "OPTED_OUT")

    def test_02_opt_out_database_persistence(self):
        """Test that opting out immediately updates Lead and Conversation in database."""
        import uuid
        test_wa = f"919999{uuid.uuid4().hex[:6]}"
        lead = DB.leads.get_or_create(test_wa, name="Rishi Test")
        lead_id = lead.get("lead_id")
        conv = DB.conversations.create_if_not_exists(test_wa, lead_id=lead_id)
        conv_id = conv.get("conversation_id")

        # Initial state should not be opted out
        self.assertFalse(lead.get("opted_out", False))

        # Perform opt out
        DB.leads.opt_out(test_wa)
        DB.conversations.set_opted_out(conv_id, True)

        updated_lead = DB.leads.get_by_id(lead_id)
        self.assertTrue(updated_lead.get("opted_out"))
        self.assertEqual(updated_lead.get("sales_stage"), "OPTED_OUT")

        updated_conv = DB.conversations.get_by_id(conv_id)
        self.assertIsNotNone(updated_conv)
        self.assertTrue(updated_conv.get("opted_out"))
        self.assertEqual(updated_conv.get("status"), "opted_out")

        # Now test re-engagement: customer reaches back out
        res = self.nba.determine(
            customer_memory=CustomerMemory(lead_id=lead_id),
            sales_memory=SalesMemory(lead_id=lead_id, sales_stage="OPTED_OUT"),
            recent_messages=[],
            current_message="Hello, can you help me find a 2BHK flat?"
        )
        self.assertEqual(res["intent"], "RE_ENGAGE")
        self.assertEqual(res["next_best_action"], "RE_ENGAGE_WELCOME")

        # Opt-in
        DB.leads.opt_in(test_wa)
        DB.conversations.set_opted_out(conv_id, False)

        re_opted_lead = DB.leads.get_by_id(lead_id)
        self.assertFalse(re_opted_lead.get("opted_out"))

    def test_03_returning_greeting_prevents_repeated_pitches(self):
        """Test that casual 'hi' or 'hello' mid-conversation triggers CASUAL_CHECK_IN instead of full property pitch."""
        dummy_cm = CustomerMemory(
            lead_id="test_lead_2",
            requirements={"city": "Nagpur", "locality": "Manish Nagar", "bhk": "2BHK"},
            recommended_properties=["Green Meadows Residency"]
        )
        dummy_sm = SalesMemory(lead_id="test_lead_2", sales_stage="PROPERTY_RECOMMENDED")

        greetings = ["Hii", "hi", "Hello", "hey!", "Namaste"]
        for g in greetings:
            res = self.nba.determine(
                customer_memory=dummy_cm,
                sales_memory=dummy_sm,
                recent_messages=[],
                current_message=g
            )
            self.assertEqual(res["intent"], "RETURNING_GREETING", f"Failed on greeting: '{g}'")
            self.assertEqual(res["next_best_action"], "CASUAL_CHECK_IN")

    def test_04_followup_quiet_hours(self):
        """Test quiet hours detection in Indian Standard Time (UTC+5:30)."""
        # 23:00 IST is 17:30 UTC -> quiet hours (True)
        night_utc = datetime(2026, 9, 19, 17, 30, tzinfo=timezone.utc)
        self.assertTrue(FollowUpService.is_quiet_hours(night_utc))

        # 04:00 IST is 22:30 UTC -> quiet hours (True)
        early_morning_utc = datetime(2026, 9, 19, 22, 30, tzinfo=timezone.utc)
        self.assertTrue(FollowUpService.is_quiet_hours(early_morning_utc))

        # 14:00 IST is 08:30 UTC -> daytime (False)
        day_utc = datetime(2026, 9, 19, 8, 30, tzinfo=timezone.utc)
        self.assertFalse(FollowUpService.is_quiet_hours(day_utc))

    def test_05_followup_eligibility_window(self):
        """Test that follow-ups strictly fire between 6 to 12 hours and respect rules."""
        # Use fixed daytime in UTC (11:00 AM IST = 05:30 UTC)
        now = datetime(2026, 9, 19, 5, 30, tzinfo=timezone.utc)

        # 1. Inactive for only 2 hours -> NOT eligible
        conv_2h = {
            "conversation_id": "conv_test_1",
            "status": "active",
            "sales_stage": "PROPERTY_RECOMMENDED",
            "last_ai_message_at": now - timedelta(hours=2),
            "last_customer_message_at": now - timedelta(hours=2, minutes=5),
            "followup_count": 0,
            "opted_out": False
        }
        self.assertFalse(self.fup_svc.is_eligible_for_followup(conv_2h, now_utc=now))

        # 2. Inactive for 8 hours (between 6 and 12h) -> ELIGIBLE
        conv_8h = {
            "conversation_id": "conv_test_2",
            "status": "active",
            "sales_stage": "PROPERTY_RECOMMENDED",
            "last_ai_message_at": now - timedelta(hours=8),
            "last_customer_message_at": now - timedelta(hours=8, minutes=5),
            "followup_count": 0,
            "opted_out": False
        }
        self.assertTrue(self.fup_svc.is_eligible_for_followup(conv_8h, now_utc=now))

        # 3. Customer replied AFTER last AI message -> NOT eligible (waiting for us to answer)
        conv_cust_replied = {
            "conversation_id": "conv_test_3",
            "status": "active",
            "sales_stage": "PROPERTY_RECOMMENDED",
            "last_ai_message_at": now - timedelta(hours=8),
            "last_customer_message_at": now - timedelta(hours=1),
            "followup_count": 0,
            "opted_out": False
        }
        self.assertFalse(self.fup_svc.is_eligible_for_followup(conv_cust_replied, now_utc=now))

        # 4. Opted-out conversation -> NOT eligible
        conv_opted_out = {
            "conversation_id": "conv_test_4",
            "status": "active",
            "sales_stage": "PROPERTY_RECOMMENDED",
            "last_ai_message_at": now - timedelta(hours=8),
            "followup_count": 0,
            "opted_out": True
        }
        self.assertFalse(self.fup_svc.is_eligible_for_followup(conv_opted_out, now_utc=now))

        # 5. Visit already booked -> NOT eligible
        conv_booked = {
            "conversation_id": "conv_test_5",
            "status": "active",
            "sales_stage": "VISIT_BOOKED",
            "last_ai_message_at": now - timedelta(hours=8),
            "followup_count": 0,
            "opted_out": False
        }
        self.assertFalse(self.fup_svc.is_eligible_for_followup(conv_booked, now_utc=now))

        # 6. Max followups reached (2) -> NOT eligible
        conv_max = {
            "conversation_id": "conv_test_6",
            "status": "active",
            "sales_stage": "PROPERTY_RECOMMENDED",
            "last_ai_message_at": now - timedelta(hours=8),
            "followup_count": 2,
            "opted_out": False
        }
        self.assertFalse(self.fup_svc.is_eligible_for_followup(conv_max, now_utc=now))

    def test_06_smart_followup_message_length_and_content(self):
        """Test that generated follow-up messages are concise (under 35 words) and contextual."""
        conv_pitched = {"sales_stage": "VISIT_PITCHED", "lead_id": "lead_x"}
        lead = {"name": "Rishi"}
        msg = self.fup_svc.generate_smart_followup_message(conv_pitched, lead)
        
        words = msg.split()
        self.assertLessEqual(len(words), 35, f"Message too long ({len(words)} words): {msg}")
        self.assertIn("Rishi", msg)
        self.assertTrue(any(w in msg.lower() for w in ["visit", "weekend", "look"]))


if __name__ == "__main__":
    unittest.main()
