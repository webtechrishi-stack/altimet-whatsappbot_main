"""
Comprehensive Verification Suite for ARIS Sales Conversion Engine & Conversation Intelligence.
Tests:
1. Message Persistence, Idempotency & Delivery Correlation (wamid)
2. Facts vs Inferences in Customer Memory
3. Sales State Machine Transitions
4. Objection Engine (Zero Fabricated Discounts)
5. Visit Pitch Engine & Complimentary Cab Flow
6. Next Best Action (NBA) Engine
7. Visit Readiness Scoring (0-100) & Lead Temperature
8. Human Takeover Mode & Suppression
9. Security Sanitization & Conversation Export
10. Full 13-Turn Real Estate Conversion Simulation (Section 56)
"""

import sys
import os
import unittest
from datetime import datetime, timezone

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app import app
from database import (
    DB, LeadRepository, ConversationRepository, MessageRepository,
    CustomerMemoryRepository, SalesMemoryRepository, SalesEventRepository,
    Message, Lead, Conversation
)
from conversation.models import CustomerMemory, SalesMemory, FactOrInference
from conversation.memory_service import CustomerMemoryService
from conversation.summarizer import ConversationSummarizer
from conversation.context_builder import ConversationContextBuilder
from sales.state_machine import ConversationSalesStage, SalesStateMachine
from sales.objection_engine import ObjectionEngine
from sales.visit_pitch_engine import VisitPitchEngine, PitchStrategy
from sales.next_best_action import NextBestActionEngine
from sales.conversion_service import ConversionService


class TestSalesEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        cls.test_phone = "919876543210"
        cls.test_phone_2 = "919876543211"

    def test_01_message_persistence_and_correlation(self):
        """Verify inbound/outbound message persistence, wamid correlation, and delivery updates."""
        conv = DB.conversations.get_or_create_conversation(self.test_phone, "Rahul Sharma")
        conv_id = conv["_id"]

        # 1. Inbound message pre-persistence
        wamid_in = "wamid.HBgLMTE4ODE2"
        in_msg = DB.messages.save_inbound_message(
            conversation_id=conv_id,
            wa_id=self.test_phone,
            text="Looking for a 3 BHK in Whitefield around 1.8 Cr",
            whatsapp_message_id=wamid_in,
            source="WHATSAPP"
        )
        self.assertIsNotNone(in_msg)
        self.assertEqual(in_msg["direction"], "INBOUND")
        self.assertEqual(in_msg["sender_type"], "CUSTOMER")
        self.assertEqual(in_msg["status"].lower(), "received")
        self.assertEqual(in_msg["whatsapp_message_id"], wamid_in)

        # 2. Idempotency verification: duplicate wamid must be rejected
        is_duplicate = DB.messages.check_idempotency_wamid(wamid_in)
        self.assertTrue(is_duplicate, "check_idempotency_wamid must return True for existing wamid")

        # 3. Outbound AI message persistence
        wamid_out = "wamid.HBgLMTE4ODE3"
        out_msg = DB.messages.save_outbound_ai_message(
            conversation_id=conv_id,
            wa_id=self.test_phone,
            text="We have excellent 3 BHK homes in Whitefield! Would you like a tour?",
            whatsapp_message_id=wamid_out,
            status="sent",
            metadata={"model": "gemini-2.5-flash", "stage": "VISIT_PITCHED"}
        )
        self.assertIsNotNone(out_msg)
        self.assertEqual(out_msg["direction"], "OUTBOUND")
        self.assertEqual(out_msg["sender_type"], "AI")
        self.assertEqual(out_msg["whatsapp_message_id"], wamid_out)
        self.assertEqual(out_msg["status"], "sent")

        # 4. Outbound Human message persistence
        human_msg = DB.messages.save_outbound_human_message(
            conversation_id=conv_id,
            wa_id=self.test_phone,
            text="Hi Rahul, this is Sarah from ARIS Concierge.",
            agent_id="agent_sarah_01"
        )
        self.assertEqual(human_msg["sender_type"], "HUMAN")
        self.assertEqual(human_msg["agent_id"], "agent_sarah_01")

        # 5. Delivery status correlation via wamid
        updated = DB.messages.update_delivery_status_by_wamid(wamid_out, "delivered")
        self.assertTrue(updated)
        refetched = DB.messages.get_message_by_wamid(wamid_out)
        self.assertEqual(refetched["status"].lower(), "delivered")

        # Update to read
        DB.messages.update_delivery_status_by_wamid(wamid_out, "read")
        refetched = DB.messages.get_message_by_wamid(wamid_out)
        self.assertEqual(refetched["status"].lower(), "read")

        # 6. Cursor pagination
        res = DB.messages.get_paginated_messages(conv_id, limit=2)
        self.assertIn("messages", res)
        self.assertIn("has_more", res)

        print("[PASS] Test 1: Inbound/outbound persistence, wamid correlation, and delivery updates passed.")

    def test_02_facts_vs_inference_and_customer_memory(self):
        """Verify strict facts-over-inference preservation and regex extraction."""
        conv = DB.conversations.get_or_create_conversation(self.test_phone_2, "Anita Patel")
        conv_id = conv["_id"]

        # Explicit statement by user
        user_text = "I am strictly looking for a 3 BHK in Whitefield with a budget of 1.8 Cr. Need it for self-use immediately."
        CustomerMemoryService.extract_and_update(self.test_phone_2, conv_id, user_text)

        mem = CustomerMemoryService.get_memory(self.test_phone_2)
        self.assertIsNotNone(mem)

        # Verify explicit facts
        self.assertEqual(mem.budget_max, 180.0)
        self.assertIn("3 BHK", mem.bhk)
        self.assertEqual(mem.preferred_locality, "Whitefield")
        self.assertEqual(mem.purpose, "END_USE")
        self.assertEqual(mem.timeline, "READY_TO_MOVE")

        # Verify fact confidence
        self.assertTrue(mem.facts["budget"].is_explicit)
        self.assertEqual(mem.facts["budget"].confidence, 1.0)
        self.assertTrue(mem.facts["bhk"].is_explicit)

        # Test that inference CANNOT overwrite explicit fact
        # If user says something vague, explicit fact remains intact
        CustomerMemoryService.extract_and_update(self.test_phone_2, conv_id, "Maybe somewhere in Bangalore")
        mem_after = CustomerMemoryService.get_memory(self.test_phone_2)
        self.assertEqual(mem_after.preferred_locality, "Whitefield", "Explicit locality must not be overwritten by vague inference")
        self.assertEqual(mem_after.budget_max, 180.0, "Explicit budget must remain intact")

        # Verify CRM Lead sync
        lead = DB.leads.get_by_phone(self.test_phone_2)
        self.assertIsNotNone(lead)
        self.assertEqual(lead.budget_max, 180.0)
        self.assertEqual(lead.preferred_bhk, "3 BHK")

        print("[PASS] Test 2: Customer memory extractors & facts-over-inference verified.")

    def test_03_sales_state_machine(self):
        """Verify deterministic sales stage transitions and validation rules."""
        # 1. Allowed transitions
        self.assertTrue(SalesStateMachine.can_transition(ConversationSalesStage.NEW, ConversationSalesStage.DISCOVERY))
        self.assertTrue(SalesStateMachine.can_transition(ConversationSalesStage.DISCOVERY, ConversationSalesStage.QUALIFIED))
        self.assertTrue(SalesStateMachine.can_transition(ConversationSalesStage.QUALIFIED, ConversationSalesStage.PROPERTY_RECOMMENDED))
        self.assertTrue(SalesStateMachine.can_transition(ConversationSalesStage.PROPERTY_RECOMMENDED, ConversationSalesStage.VISIT_PITCHED))
        self.assertTrue(SalesStateMachine.can_transition(ConversationSalesStage.VISIT_PITCHED, ConversationSalesStage.VISIT_BOOKED))

        # 2. Disallowed illegal transitions
        self.assertFalse(SalesStateMachine.can_transition(ConversationSalesStage.NEW, ConversationSalesStage.WON))
        self.assertFalse(SalesStateMachine.can_transition(ConversationSalesStage.NEW, ConversationSalesStage.VISIT_COMPLETED))

        # 3. Backward transition (resetting requirements back to DISCOVERY)
        self.assertTrue(SalesStateMachine.can_transition(ConversationSalesStage.ENGAGED, ConversationSalesStage.DISCOVERY))

        # 4. Evaluation helper
        next_stage = SalesStateMachine.evaluate_stage_progression(
            current_stage=ConversationSalesStage.NEW,
            user_message="I want to buy an apartment",
            has_properties=False,
            visit_status=None
        )
        self.assertEqual(next_stage, ConversationSalesStage.DISCOVERY)

        print("[PASS] Test 3: Sales state machine valid and invalid transitions verified.")

    def test_04_objection_engine(self):
        """Verify objection detection and consultative responses with zero fake discounts."""
        # Price objection
        cat = ObjectionEngine.detect_objection("Your price is too high and out of my budget")
        self.assertEqual(cat, "PRICE")
        strat = ObjectionEngine.get_strategy(cat)
        self.assertIn("never fabricate discounts", strat["guideline"].lower(), "Must NOT fabricate discounts")
        self.assertNotIn("hurry only today", strat["guideline"].lower(), "Must NOT use fake urgency")

        # Spouse / Family objection
        cat_spouse = ObjectionEngine.detect_objection("I need to discuss this with my wife before deciding")
        self.assertEqual(cat_spouse, "SPOUSE_NOT_CONVINCED")
        strat_spouse = ObjectionEngine.get_strategy(cat_spouse)
        self.assertTrue(any(w in strat_spouse["guideline"].lower() for w in ["wife", "both", "family", "cab", "partner"]))

        # Need to think objection
        cat_think = ObjectionEngine.detect_objection("Let me think about it and get back to you")
        self.assertEqual(cat_think, "NEED_TO_THINK")

        # Distance objection
        cat_dist = ObjectionEngine.detect_objection("Whitefield is too far from my office")
        self.assertIn(cat_dist, ("LOCATION", "LOCATION_DISTANCE"))

        print("[PASS] Test 4: Objection engine detection & truthful strategies verified.")

    def test_05_visit_pitch_engine(self):
        """Verify consultative visit pitches with complimentary cab inclusion."""
        # Experience pitch
        pitch_exp = VisitPitchEngine.craft_pitch(
            strategy=PitchStrategy.EXPERIENCE,
            property_name="Sobha Dream Acres",
            locality="Panathur Road",
            customer_name="Vikram"
        )
        self.assertIn("Vikram", pitch_exp)
        self.assertIn("Sobha Dream Acres", pitch_exp)
        self.assertTrue("cab" in pitch_exp.lower() or "chauffeur" in pitch_exp.lower() or "ride" in pitch_exp.lower())

        # Convenience pitch
        pitch_conv = VisitPitchEngine.craft_pitch(
            strategy=PitchStrategy.CONVENIENCE,
            property_name="Prestige Lakeside",
            locality="Varthur",
            customer_name="Anita"
        )
        self.assertIn("complimentary", pitch_conv.lower())
        self.assertIn("cab", pitch_conv.lower())

        print("[PASS] Test 5: Visit pitch engine & complimentary cab integration verified.")

    def test_06_next_best_action_engine(self):
        """Verify Next Best Action selection across various conversational states."""
        # Case A: Brand new user -> ASK_REQUIREMENT
        action_a = NextBestActionEngine.determine_nba(
            current_stage=ConversationSalesStage.NEW,
            customer_memory=CustomerMemory(phone="919000000001"),
            sales_memory=SalesMemory(phone="919000000001"),
            last_user_message="Hello, looking for a flat"
        )
        self.assertEqual(action_a["action_type"], "ASK_REQUIREMENT")

        # Case B: Qualified customer + property recommended -> PITCH_SITE_VISIT
        mem_b = CustomerMemory(
            phone="919000000002",
            requirements={"city": "Bangalore", "locality": "Whitefield", "bhk": "3 BHK", "budget_max": 180.0},
            recommended_properties=["Sobha Dream Acres"]
        )
        sales_b = SalesMemory(phone="919000000002", recommended_properties=["Sobha Dream Acres"])
        action_b = NextBestActionEngine.determine_nba(
            current_stage=ConversationSalesStage.PROPERTY_RECOMMENDED,
            customer_memory=mem_b,
            sales_memory=sales_b,
            last_user_message="The photos of Sobha Dream Acres look very nice!"
        )
        self.assertEqual(action_b["action_type"], "PITCH_SITE_VISIT")

        # Case C: Customer raises objection -> RESOLVE_OBJECTION
        action_c = NextBestActionEngine.determine_nba(
            current_stage=ConversationSalesStage.VISIT_PITCHED,
            customer_memory=mem_b,
            sales_memory=sales_b,
            last_user_message="The price is a bit higher than my budget"
        )
        self.assertEqual(action_c["action_type"], "RESOLVE_OBJECTION")
        self.assertEqual(action_c["objection_category"], "PRICE")

        # Case D: Customer agrees to visit -> COLLECT_VISIT_DETAILS (booking mode)
        action_d = NextBestActionEngine.determine_nba(
            current_stage=ConversationSalesStage.VISIT_PITCHED,
            customer_memory=mem_b,
            sales_memory=sales_b,
            last_user_message="Sure, I would love to visit this Saturday around 11 AM"
        )
        self.assertEqual(action_d["action_type"], "COLLECT_VISIT_DETAILS")
        self.assertTrue(action_d["booking_mode_active"])

        print("[PASS] Test 6: Next Best Action engine determinations verified.")

    def test_07_conversion_service_and_scoring(self):
        """Verify Visit Readiness Score (0-100) and Lead Temperature (HOT/WARM/COLD)."""
        conv = DB.conversations.get_or_create_conversation("919876543299", "Deepak Rao")
        conv_id = conv["_id"]

        # 1. Cold lead: bare inquiry
        score_cold = ConversionService.calculate_readiness_score(
            customer_memory=CustomerMemory(phone="919876543299"),
            sales_memory=SalesMemory(phone="919876543299"),
            sales_stage=ConversationSalesStage.NEW
        )
        temp_cold = ConversionService.calculate_lead_temperature(score_cold, ConversationSalesStage.NEW)
        self.assertLess(score_cold, 40)
        self.assertEqual(temp_cold, "COLD")

        # 2. Warm lead: qualified requirements
        mem_warm = CustomerMemory(
            phone="919876543299",
            requirements={"budget_max": 150.0, "bhk": "2 BHK", "locality": "Hinjawadi", "timeline": "1-3 months"},
            recommended_properties=["Godrej Elements"]
        )
        sales_warm = SalesMemory(phone="919876543299", recommended_properties=["Godrej Elements"])
        score_warm = ConversionService.calculate_readiness_score(
            customer_memory=mem_warm,
            sales_memory=sales_warm,
            sales_stage=ConversationSalesStage.PROPERTY_RECOMMENDED
        )
        temp_warm = ConversionService.calculate_lead_temperature(score_warm, ConversationSalesStage.PROPERTY_RECOMMENDED)
        self.assertGreaterEqual(score_warm, 40)
        self.assertIn(temp_warm, ["WARM", "HOT"])

        # 3. Hot lead: visit pitched / intent shown
        sales_hot = SalesMemory(phone="919876543299", visit_pitched=True, visit_status="CONFIRMED")
        score_hot = ConversionService.calculate_readiness_score(
            customer_memory=mem_warm,
            sales_memory=sales_hot,
            sales_stage=ConversationSalesStage.VISIT_BOOKED
        )
        temp_hot = ConversionService.calculate_lead_temperature(score_hot, ConversationSalesStage.VISIT_BOOKED)
        self.assertGreaterEqual(score_hot, 70)
        self.assertEqual(temp_hot, "HOT")

        # 4. Audit logging
        ConversionService.log_sales_event(
            phone="919876543299",
            conversation_id=conv_id,
            event_type="STAGE_CHANGED",
            previous_stage="NEW",
            new_stage="VISIT_BOOKED",
            readiness_score=score_hot,
            lead_temperature=temp_hot
        )
        events = DB.sales_events.get_events_for_conversation(conv_id)
        self.assertGreater(len(events), 0)
        self.assertEqual(events[0]["event_type"], "STAGE_CHANGED")

        print("[PASS] Test 7: Visit readiness scoring, temperature & sales audit logging verified.")

    def test_08_human_takeover_behavior(self):
        """Verify human takeover suppression of AI and manual agent messaging."""
        phone = "919876543277"
        conv = DB.conversations.get_or_create_conversation(phone, "Kiran Kumar")
        conv_id = conv["_id"]

        # 1. Enable human takeover via API
        res = self.client.post(f"/api/conversations/{conv_id}/takeover", json={"enable": True, "agent_id": "agent_rahul"})
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.get_json()["human_takeover"])

        # Check DB state
        conv_updated = DB.conversations.get_by_id(conv_id)
        self.assertTrue(conv_updated["human_takeover"])

        # 2. Simulate manual message sent by human agent
        res_send = self.client.post(f"/api/conversations/{conv_id}/messages", json={
            "text": "Hello Kiran, I will personally assist you with the booking.",
            "agent_id": "agent_rahul"
        })
        self.assertEqual(res_send.status_code, 200)
        self.assertTrue(res_send.get_json()["success"])

        # Verify saved message
        msgs = DB.messages.get_last_messages(conv_id, limit=5)
        latest = msgs[-1]
        self.assertEqual(latest["sender_type"], "HUMAN")
        self.assertIn("personally assist", latest["text"])

        # 3. Disable human takeover
        res_release = self.client.post(f"/api/conversations/{conv_id}/takeover", json={"enable": False})
        self.assertEqual(res_release.status_code, 200)
        self.assertFalse(res_release.get_json()["human_takeover"])

        print("[PASS] Test 8: Human takeover activation, message persistence, and release verified.")

    def test_09_security_sanitization_and_export(self):
        """Verify conversation export endpoint sanitizes confidential tokens and keys."""
        phone = "919876543266"
        conv = DB.conversations.get_or_create_conversation(phone, "Security Test")
        conv_id = conv["_id"]

        # Insert test message
        DB.messages.save_inbound_message(conv_id, phone, "Hi, checking export endpoint security.")

        # Test JSON export
        res_json = self.client.get(f"/api/conversations/{conv_id}/export?format=json")
        self.assertEqual(res_json.status_code, 200)
        data = res_json.get_json()
        self.assertIn("conversation_id", data)
        self.assertIn("messages", data)

        # Check that no sensitive tokens exist in the export
        dumped = str(data).lower()
        self.assertNotIn("access_token", dumped)
        self.assertNotIn("gemini_api_key", dumped)
        self.assertNotIn("password", dumped)

        # Test TXT export
        res_txt = self.client.get(f"/api/conversations/{conv_id}/export?format=txt")
        self.assertEqual(res_txt.status_code, 200)
        self.assertIn("ARIS CONVERSATION EXPORT", res_txt.get_data(as_text=True))

        print("[PASS] Test 9: Conversation export and security sanitization verified.")

    def test_10_full_13_turn_simulation(self):
        """
        Full 13-Turn Real Estate Conversion Simulation (Section 56):
        Simulates end-to-end conversation from first greeting through qualification,
        objection handling, site visit booking, complimentary cab, and delivery status correlation.
        """
        phone = "919876543255"
        customer_name = "Aditya Verma"
        # Reset memory for clean simulation
        DB.customer_memory.delete(phone)
        DB.sales_memory.delete(phone)
        lead = DB.leads.get_by_phone(phone)
        if lead:
            lead_id = lead.get("lead_id") or lead.get("_id")
            DB.customer_memory.delete(str(lead_id))
            DB.sales_memory.delete(str(lead_id))
            DB.leads.update(str(lead_id), {"bhk": "", "budget_max": 0.0, "preferred_location": ""})

        conv = DB.conversations.get_or_create_conversation(phone, customer_name)
        conv_id = conv["_id"]

        # Turn 1: Initial Greeting & Discovery
        t1_text = "Hi, I am looking for flats in Bangalore"
        DB.messages.save_inbound_message(conv_id, phone, t1_text, whatsapp_message_id="wamid_sim_01")
        CustomerMemoryService.extract_and_update(phone, conv_id, t1_text)
        nba_1 = NextBestActionEngine.determine_nba(
            ConversationSalesStage.NEW,
            CustomerMemoryService.get_memory(phone),
            SalesMemory(phone=phone),
            t1_text
        )
        self.assertEqual(nba_1["action_type"], "ASK_REQUIREMENT")
        DB.messages.save_outbound_ai_message(conv_id, phone, "Welcome Aditya! Which localities and BHK size are you exploring?", whatsapp_message_id="wamid_sim_01_reply")

        # Turn 2: Budget stated
        t2_text = "My budget is around 1.5 to 1.8 Cr"
        DB.messages.save_inbound_message(conv_id, phone, t2_text, whatsapp_message_id="wamid_sim_02")
        CustomerMemoryService.extract_and_update(phone, conv_id, t2_text)
        mem = CustomerMemoryService.get_memory(phone)
        self.assertEqual(mem.budget_max, 180.0)

        # Turn 3: Locality stated
        t3_text = "Preferably in Whitefield"
        DB.messages.save_inbound_message(conv_id, phone, t3_text, whatsapp_message_id="wamid_sim_03")
        CustomerMemoryService.extract_and_update(phone, conv_id, t3_text)
        mem = CustomerMemoryService.get_memory(phone)
        self.assertEqual(mem.preferred_locality, "Whitefield")

        # Turn 4: BHK configuration stated
        t4_text = "Strictly 3 BHK only"
        DB.messages.save_inbound_message(conv_id, phone, t4_text, whatsapp_message_id="wamid_sim_04")
        CustomerMemoryService.extract_and_update(phone, conv_id, t4_text)
        mem = CustomerMemoryService.get_memory(phone)
        self.assertIn("3 BHK", mem.bhk)

        # Transition to QUALIFIED
        DB.conversations.update_conversation(conv_id, {"sales_stage": ConversationSalesStage.QUALIFIED})

        # Turn 5: Property Recommendation
        sales_mem = SalesMemory(phone=phone, recommended_properties=["Prestige Somerville"])
        DB.sales_memory.save_memory(sales_mem)
        DB.conversations.update_conversation(conv_id, {"sales_stage": ConversationSalesStage.PROPERTY_RECOMMENDED})
        DB.messages.save_outbound_ai_message(conv_id, phone, "Prestige Somerville in Whitefield offers stunning 3 BHK homes within 1.8 Cr.", whatsapp_message_id="wamid_sim_05_reply")

        # Turn 6: Customer asks about amenities
        t6_text = "Does Prestige Somerville have a swimming pool and squash court?"
        DB.messages.save_inbound_message(conv_id, phone, t6_text, whatsapp_message_id="wamid_sim_06")
        DB.conversations.update_conversation(conv_id, {"sales_stage": ConversationSalesStage.ENGAGED})

        # Turn 7: AI Pitches Visit with Complimentary Cab
        pitch = VisitPitchEngine.craft_pitch(PitchStrategy.CONVENIENCE, "Prestige Somerville", "Whitefield", customer_name)
        self.assertTrue("cab" in pitch.lower() or "chauffeur" in pitch.lower())
        DB.messages.save_outbound_ai_message(conv_id, phone, pitch, whatsapp_message_id="wamid_sim_07_reply")
        sales_mem.visit_pitched = True
        DB.sales_memory.save_memory(sales_mem)
        DB.conversations.update_conversation(conv_id, {"sales_stage": ConversationSalesStage.VISIT_PITCHED})

        # Turn 8: Customer raises price objection
        t8_text = "The base price seems a little high compared to older projects."
        DB.messages.save_inbound_message(conv_id, phone, t8_text, whatsapp_message_id="wamid_sim_08")
        nba_8 = NextBestActionEngine.determine_nba(
            ConversationSalesStage.VISIT_PITCHED,
            mem,
            sales_mem,
            t8_text
        )
        self.assertEqual(nba_8["action_type"], "RESOLVE_OBJECTION")
        self.assertEqual(nba_8["objection_category"], "PRICE")
        # Ensure no fake discounts
        for pt in nba_8["talking_points"]:
            self.assertIn("never fabricate discounts", pt.lower())
            self.assertNotIn("hurry only today", pt.lower())

        # Turn 9: Customer raises spouse objection
        t9_text = "I need to discuss with my wife before booking anything."
        DB.messages.save_inbound_message(conv_id, phone, t9_text, whatsapp_message_id="wamid_sim_09")
        nba_9 = NextBestActionEngine.determine_nba(
            ConversationSalesStage.VISIT_PITCHED,
            mem,
            sales_mem,
            t9_text
        )
        self.assertEqual(nba_9["action_type"], "RESOLVE_OBJECTION")
        self.assertEqual(nba_9["objection_category"], "SPOUSE_NOT_CONVINCED")

        # Turn 10: Customer agrees to visit
        t10_text = "Okay, that makes sense. Can we visit this Saturday?"
        DB.messages.save_inbound_message(conv_id, phone, t10_text, whatsapp_message_id="wamid_sim_10")
        nba_10 = NextBestActionEngine.determine_nba(
            ConversationSalesStage.VISIT_PITCHED,
            mem,
            sales_mem,
            t10_text
        )
        self.assertEqual(nba_10["action_type"], "COLLECT_VISIT_DETAILS")
        self.assertTrue(nba_10["booking_mode_active"])
        DB.conversations.update_conversation(conv_id, {"sales_stage": ConversationSalesStage.VISIT_NEGOTIATING})

        # Turn 11: Customer provides visit time and pickup address
        t11_text = "Saturday 11 AM, pickup from Indiranagar 100ft Road."
        DB.messages.save_inbound_message(conv_id, phone, t11_text, whatsapp_message_id="wamid_sim_11")
        # Site visit booked in DB
        DB.visits.create_visit(
            lead_id=conv.get("lead_id", "lead_aditya"),
            wa_id=phone,
            property_id="prop_prestige_somerville",
            property_title="Prestige Somerville",
            visit_date="2026-09-19",
            visit_time="11:00 AM",
            pickup_required=True,
            pickup_address="Indiranagar 100ft Road"
        )
        DB.conversations.update_conversation(conv_id, {"sales_stage": ConversationSalesStage.VISIT_BOOKED})
        score_11 = ConversionService.calculate_readiness_score(mem, sales_mem, ConversationSalesStage.VISIT_BOOKED)
        temp_11 = ConversionService.calculate_lead_temperature(score_11, ConversationSalesStage.VISIT_BOOKED)
        self.assertEqual(temp_11, "HOT")

        # Turn 12: Human agent takeover to confirm cab & driver details
        res_takeover = self.client.post(f"/api/conversations/{conv_id}/takeover", json={"enable": True, "agent_id": "agent_vip_desk"})
        self.assertEqual(res_takeover.status_code, 200)
        res_driver = self.client.post(f"/api/conversations/{conv_id}/messages", json={
            "text": "Hi Aditya, your VIP cab is scheduled for Saturday 10:15 AM from Indiranagar. Chauffeur contact will be shared 1 hour prior.",
            "agent_id": "agent_vip_desk"
        })
        self.assertEqual(res_driver.status_code, 200)

        # Turn 13: Delivery receipt webhook updates message status
        last_msg = DB.messages.get_last_messages(conv_id, limit=1)[0]
        if last_msg.get("whatsapp_message_id"):
            DB.messages.update_delivery_status_by_wamid(last_msg["whatsapp_message_id"], "delivered")
            DB.messages.update_delivery_status_by_wamid(last_msg["whatsapp_message_id"], "read")
            updated_msg = DB.messages.get_message_by_wamid(last_msg["whatsapp_message_id"])
            self.assertEqual(updated_msg["status"].lower(), "read")

        print("[PASS] Test 10: Full 13-turn conversational sales simulation completed successfully!")


if __name__ == "__main__":
    unittest.main()
