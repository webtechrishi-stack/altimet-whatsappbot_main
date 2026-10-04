"""
Personalized Sales Agent — coordinates Next Best Action, Context Builder,
Gemini generation, and fallback to deterministic engine.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional

from database import DB
from conversation.models import ConversationContext
from conversation.context_builder import ConversationContextBuilder
from conversation.memory_service import CustomerMemoryService
from sales.next_best_action import NextBestActionEngine
from sales.conversion_service import ConversionService
from sales.state_machine import SalesStateMachine, ConversationSalesStage
from sales.objection_engine import ObjectionEngine
from sales.visit_pitch_engine import VisitPitchEngine, PitchStrategy
from ai.prompts import ARISPromptManager
from ai_engine import GeminiProvider, ARISAgent, RAGService, VisitService

logger = logging.getLogger(__name__)


class PersonalizedSalesAgent:
    """
    Main consultative conversational engine for ARIS.
    """

    def __init__(self):
        self.context_builder = ConversationContextBuilder()
        self.memory_service = CustomerMemoryService()
        self.nba_engine = NextBestActionEngine()
        self.conversion_service = ConversionService()
        self.prompt_manager = ARISPromptManager()
        self.gemini_provider = GeminiProvider()
        self.deterministic_fallback = ARISAgent()
        self.rag_service = RAGService()
        self.property_repo = DB.properties
        self.visit_service = VisitService()
        self.objection_engine = ObjectionEngine()

    def process_turn(
        self,
        lead_id: str,
        conversation_id: str,
        wa_id: str,
        customer_name: str,
        message_text: str,
        session_data: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        """
        Executes complete consultative turn:
        1. Extract & update customer memory (facts vs inference)
        2. Compute visit readiness score and lead temperature
        3. Evaluate Next Best Action
        4. Apply state machine transition
        5. Build compact context payload
        6. Generate response via Gemini (or fallback)
        7. Return (reply_text, updated_session, nba_metadata)
        """
        # Step 1: Update customer memory from message
        customer_mem = self.memory_service.update_from_message(
            lead_id=lead_id,
            message_text=message_text
        )

        # Step 2: Compute visit readiness and temperature
        readiness_score = self.conversion_service.compute_visit_readiness(
            customer_memory=customer_mem,
            recent_text=message_text
        )
        temperature = self.conversion_service.compute_lead_temperature(
            readiness_score=readiness_score,
            recent_text=message_text
        )

        # Step 3: Check property focus from session or message
        focused_prop_id = session_data.get("context", {}).get("focused_property_id")
        if not focused_prop_id:
            for p in self.property_repo.list_properties(limit=10):
                p_title = p.get("title", "").lower()
                p_id = p.get("id", "").lower()
                if p_title in message_text.lower() or p_id in message_text.lower():
                    focused_prop_id = p.get("id")
                    session_data.setdefault("context", {})["focused_property_id"] = focused_prop_id
                    break

        # Step 4: Retrieve relevant RAG excerpts if customer asked a factual question
        rag_context = ""
        if any(w in message_text.lower() for w in ["amenit", "spec", "price", "carpet", "possession", "rera", "floor", "payment", "brochure"]):
            rag_context = self.rag_service.retrieve_context(
                query=message_text,
                property_id=focused_prop_id,
                top_k=2
            )

        # Step 5: Determine Next Best Action
        raw_sm = DB.sales_memory.get_by_lead_id(lead_id)
        current_sm = DB.sales_memory.get_by_lead_id(lead_id)
        current_stage = current_sm.get("sales_stage", "NEW") if current_sm else "NEW"

        nba = self.nba_engine.determine(
            customer_memory=customer_mem,
            sales_memory=current_sm or type("obj", (object,), {"sales_stage": "NEW", "visit_status": "NOT_BOOKED"}),
            recent_messages=DB.messages.get_last_messages(conversation_id, limit=6),
            current_message=message_text,
            visit_readiness_score=readiness_score
        )

        # Step 6: Advance Sales State Machine
        new_stage, transition_reason = SalesStateMachine.evaluate_transition(
            current_stage=current_stage,
            customer_requirements=customer_mem.requirements,
            visit_readiness=readiness_score,
            intent=nba.get("intent", "GENERAL"),
            recommended_count=len(customer_mem.recommended_properties)
        )
        nba["sales_stage"] = new_stage

        # Step 7: Sync sales state to DB and log events
        sm_updated = self.conversion_service.sync_sales_state(
            lead_id=lead_id,
            conversation_id=conversation_id,
            sales_stage=new_stage,
            next_best_action=nba.get("next_best_action", "DISCOVER_REQUIREMENT"),
            visit_readiness=readiness_score,
            lead_temperature=temperature,
            objection=nba.get("objection"),
            recommended_property=nba.get("recommended_property")
        )

        # Step 8: Build Compact Conversation Context
        context = self.context_builder.build(
            lead_id=lead_id,
            conversation_id=conversation_id,
            current_message=message_text,
            next_best_action_data=nba,
            rag_context=rag_context,
            focused_property_id=focused_prop_id
        )

        # Step 9: Generate response with Gemini (if available)
        reply_text = None
        if self.gemini_provider.is_configured():
            try:
                system_prompt = self.prompt_manager.build_system_prompt(context)
                messages = []
                for h in context.recent_messages[-6:]:
                    role = "user" if h.get("sender_type") == "CUSTOMER" or h.get("role") == "user" else "assistant"
                    messages.append({"role": role, "content": h.get("text", "")})
                messages.append({"role": "user", "content": message_text})

                reply_text = self.gemini_provider.generate(
                    system_prompt=system_prompt,
                    messages=messages,
                    temperature=0.3
                )
            except Exception as ex:
                logger.error(f"[AI AGENT ERROR] Gemini generation error: {ex}")

        # Step 10: Graceful fallback to consultative human advisor engine if Gemini unavailable
        if not reply_text:
            logger.info(f"[AI AGENT] Using consultative sales advisor fallback for {conversation_id}")
            reply_text, session_data = self._generate_consultative_fallback(
                customer_name=customer_name,
                message_text=message_text,
                customer_mem=customer_mem,
                nba=nba,
                session_data=session_data,
                wa_id=wa_id,
                lead_id=lead_id,
                context=context
            )

        # Track recommended property if mentioned in reply
        for p in self.property_repo.list_properties(limit=10):
            if p.get("title", "").lower() in reply_text.lower():
                self.memory_service.record_recommended_property(lead_id, p.get("title"))

        return reply_text, session_data, nba

    def _generate_consultative_fallback(
        self,
        customer_name: str,
        message_text: str,
        customer_mem: Any,
        nba: Dict[str, Any],
        session_data: Dict[str, Any],
        wa_id: str,
        lead_id: str,
        context: ConversationContext
    ) -> Tuple[str, Dict[str, Any]]:
        text = message_text.strip()
        lower = text.lower()
        clean_name = customer_name.strip() or "there"

        # 0. Opt-Out / DND request
        if nba.get("next_best_action") == "CONFIRM_OPT_OUT" or any(w in lower for w in ["don't message", "stop", "mat bhejo", "untill i msg", "nahi chahiye", "msg mat karo"]):
            reply = (
                f"Bilkul {clean_name}, maine note kar liya hai! 🙏 "
                f"Aage se hum aapko koi message nahi karenge jab tak aap khud reach out na karein. "
                f"Have a great day ahead!"
            )
            return reply, session_data

        # 0b. Casual Check-in for returning greeting (avoids repeating full pitch)
        if nba.get("next_best_action") == "CASUAL_CHECK_IN":
            reply = (
                f"Hello {clean_name}! 👋 Great to hear from you. "
                f"How's everything going? Any questions on what we discussed, ya kuch aur explore karna tha?"
            )
            return reply, session_data

        # 0c. Re-engage welcome
        if nba.get("next_best_action") == "RE_ENGAGE_WELCOME":
            reply = (
                f"Hello {clean_name}! 👋 Great to connect with you again! "
                f"How can I help you with your property search today?"
            )
            return reply, session_data

        # 1. Explicit menu / reset commands
        if lower in ["menu", "restart", "reset", "help"]:
            return self.deterministic_fallback.process_message(
                sender_id=wa_id, profile_name=customer_name, message_text=message_text,
                session=session_data, history=context.recent_messages
            )

        # 2. Human advisor request
        if any(w in lower for w in ["talk to agent", "speak with advisor", "call me", "human agent", "talk to human", "speak to human", "advisor"]):
            reply = (
                f"👤 Thank you, {clean_name}! I have notified our senior property advisor.\n\n"
                f"They will reach out to you directly on WhatsApp (+{wa_id}) shortly to assist you personally.\n\n"
                f"In the meantime, feel free to ask me any questions about our projects or amenities! 🏡"
            )
            return reply, session_data

        # 3. Identity queries ("Kon ho aap", "Who are you", "Aap kaun ho")
        if any(w in lower for w in ["who are you", "who is this", "kon ho", "kaun ho", "aap kaun", "aap kon", "kahan se"]):
            reply = (
                f"Namaste {clean_name}! 🙏 Main ARIS Real Estate Advisory se aapka personal property consultant hoon.\n\n"
                f"Aapke property preferences ke according main aapko verified residential projects, floor plans aur honest pricing details provide karta hoon.\n\n"
                f"Saath hi hum aapke aur aapki family ke liye *complimentary doorstep VIP AC cab* arrange karte hain taaki aap bina kisi hassle ke site visit kar sakein! 🏡🚗\n\n"
                f"Batayein, aap kis location ya project ke baare mein jaanna chahenge?"
            )
            return reply, session_data

        # 4. Greetings ("hi", "hello", "hey", "hii", "namaste", "namaste hi")
        is_greeting = self.deterministic_fallback._is_greeting(lower) or any(w in lower for w in ["namaste", "pranam", "radhe radhe", "ram ram"])
        if is_greeting and not any(w in lower for w in ["visit", "book", "cab", "bhk", "price", "budget"]):
            session_data.pop("state", None)  # Clear any stuck awaiting_visit_time
            req = customer_mem.requirements if hasattr(customer_mem, "requirements") else {}
            has_prior = req.get("bhk") or req.get("city") or req.get("budget_max")
            if has_prior:
                details = []
                if req.get("bhk"):
                    details.append(f"{req['bhk']}")
                if req.get("locality") or req.get("city"):
                    details.append(f"in {req.get('locality') or req.get('city')}")
                if req.get("budget_max"):
                    details.append(f"under ₹{req['budget_max']}L")
                prior_str = " ".join(details)
                reply = (
                    f"Hello {clean_name}! 👋 Great to hear from you again from *ARIS Real Estate*.\n\n"
                    f"Aapke preferences ({prior_str}) ke mutabik mere paas verified options hain. "
                    f"Kya aap matching projects dekhna chahenge, ya hum complimentary VIP cab ke saath site visit plan karein?"
                )
            else:
                reply = (
                    f"Namaste {clean_name}! 👋 Welcome to *ARIS Real Estate Advisory*.\n\n"
                    f"Main Nagpur aur Pune mein verified dream homes find karne mein aapki help kar sakta hoon. "
                    f"Aap kis locality, BHK configuration ya budget range mein search kar rahe hain? 🏡"
                )
            return reply, session_data

        # 5. Site visit agreement with vague timing ("ha ye week", "weekend par", "ha chalega")
        is_vague_time = any(w in lower for w in ["ha ye week", "ye week", "is week", "this week", "weekend", "is weekend", "chalega", "ha", "haa"])
        has_specific_time = any(w in lower for w in [
            "tomorrow", "today", "pm", "am", "morning", "afternoon", "evening",
            "saturday", "sunday", "monday", "tuesday", "wednesday", "thursday", "friday",
            "kal", "parso", "shanivar", "ravivar", "subah", "shaam"
        ])

        if is_vague_time and not has_specific_time:
            session_data["state"] = "awaiting_visit_time"
            focused_prop_id = session_data.get("context", {}).get("focused_property_id") or "ARIS-NGP-01"
            prop = self.property_repo.get_by_id(focused_prop_id) or {}
            prop_name = prop.get("title", "Green Meadows")
            reply = (
                f"Bahut badhiya, {clean_name}! 🏡\n\n"
                f"*{prop_name}* ke visit ke liye is week kaun sa din convenient rahega—*Saturday ya Sunday*? "
                f"Aur aap *Morning (11 AM)* ya *Afternoon (4 PM)* slot prefer karenge?\n\n"
                f"Hum aapke doorstep se *complimentary private AC cab* arrange karwa denge! 🚗"
            )
            return reply, session_data

        # 6. Site visit booking: Specific time response (e.g. "tomorrow 4 pm", "saturday morning", "sunday 11 am")
        is_awaiting_time = session_data.get("state") == "awaiting_visit_time"
        if (is_awaiting_time and has_specific_time) or (has_specific_time and any(w in lower for w in ["visit", "come", "book", "reach", "at", "time", "slot"])):
            focused_prop_id = session_data.get("context", {}).get("focused_property_id") or "ARIS-NGP-01"
            prop = self.property_repo.get_by_id(focused_prop_id) or {}
            prop_title = prop.get("title", "Selected Property")
            prop_loc = f"{prop.get('locality', '')}, {prop.get('city', '')}".strip(", ")
            date_str, time_str = self.deterministic_fallback._parse_visit_datetime(text)
            try:
                self.visit_service.book_visit(
                    wa_id=wa_id,
                    customer_name=clean_name,
                    property_id=focused_prop_id,
                    visit_date=date_str,
                    visit_time=time_str,
                    requested_time_raw=text,
                    cab_required=True
                )
            except Exception as ex:
                logger.warning(f"[VISIT BOOKING FALLBACK] {ex}")

            confirmation = (
                f"🎉 *Wonderful, {clean_name}! Your site visit is scheduled!*\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"🏡 *Project:* {prop_title}\n"
                f"📍 *Location:* {prop_loc or 'Nagpur'}\n"
                f"⏰ *Requested Slot:* {text}\n"
                f"🚗 *Doorstep VIP Cab:* Confirmed (Private AC Cab)\n"
                f"━━━━━━━━━━━━━━━━━━━━\n\n"
                f"Our dedicated project host will call you shortly to confirm your preferred pickup address and share the chauffeur details.\n\n"
                f"We look forward to giving you and your family a wonderful experience!"
            )
            session_data["state"] = "visit_scheduled"
            return confirmation, session_data

        # 7. Site visit request without time ("book a visit", "site visit", "want to visit")
        if any(w in lower for w in ["site visit", "book visit", "want to visit", "schedule visit", "visit the property"]):
            focused_prop_id = session_data.get("context", {}).get("focused_property_id")
            prop = self.property_repo.get_by_id(focused_prop_id) if focused_prop_id else None
            prop_name = prop.get("title", "our featured project") if prop else "the project"
            session_data["state"] = "awaiting_visit_time"
            reply = (
                f"We would be delighted to host you at *{prop_name}*! 🏡\n\n"
                f"Our site visit includes a *complimentary private AC cab* that will pick you and your family up from your home and drop you back—completely on us with zero obligation.\n\n"
                f"Which day and time works best for you? (For example, *Tomorrow 4 PM* or *Saturday morning*)"
            )
            return reply, session_data

        # 5. Objections detected
        objection = nba.get("objection")
        if objection:
            if objection == "PRICE":
                reply = (
                    f"I completely understand, {clean_name}. Budget is one of the most critical factors in finding the right home.\n\n"
                    f"What our clients appreciate about this project is the maximum carpet efficiency and transparent pricing with zero hidden charges. "
                    f"We also have flexible bank-approved installment plans.\n\n"
                    f"If you'd like, we can arrange a quick, zero-obligation tour with our complimentary doorstep cab so you can inspect the space and layout firsthand. Would this weekend suit you?"
                )
                return reply, session_data
            elif objection == "NEED_TO_THINK":
                reply = (
                    f"Take all the time you need, {clean_name}! Buying a home is a big milestone, and there is absolutely no rush.\n\n"
                    f"Is there any specific detail—like the floor layout, connectivity, or payment milestones—that I can share to help you evaluate? I'm here to help whenever you're ready."
                )
                return reply, session_data
            elif objection == "SPOUSE_NOT_CONVINCED":
                reply = (
                    f"That makes total sense, {clean_name}! Finding the right home is always a family decision.\n\n"
                    f"Why not bring your family along for a relaxed visit? We'll arrange a complimentary doorstep AC cab to bring everyone comfortably to the site and back. Would Saturday or Sunday be convenient?"
                )
                return reply, session_data
            elif objection == "LOCATION":
                reply = (
                    f"Good question about the location, {clean_name}! The project has direct connectivity to the main arterial roads and upcoming metro access, keeping commute times very manageable.\n\n"
                    f"Experiencing the actual drive often gives the best feel. Our complimentary private cab can pick you up from your doorstep so you can test the commute comfortably. Would that be helpful?"
                )
                return reply, session_data

        # 7. Property Search / Browse / Specific BHK or Budget
        req = customer_mem.requirements if hasattr(customer_mem, "requirements") else {}
        search_city = req.get("city")
        search_bhk = req.get("bhk")
        search_budget = req.get("budget_max")

        has_search_intent = any(w in lower for w in ["bhk", "flat", "apartment", "property", "properties", "options", "nagpur", "pune", "lakh", "budget", "manish nagar", "dharampeth", "besa", "kharadi", "baner", "show", "list"])
        if has_search_intent or search_city or search_bhk or search_budget:
            matched = self.property_repo.list_properties(
                city=search_city,
                bhk=search_bhk,
                max_budget_lakhs=search_budget,
                limit=3
            )
            if not matched and search_city:
                matched = self.property_repo.list_properties(city=search_city, limit=2)
            if not matched:
                matched = self.property_repo.list_properties(limit=2)

            if matched:
                p = matched[0]
                session_data.setdefault("context", {})["focused_property_id"] = p.get("id")
                amenities = p.get("amenities", [])
                amenities_str = ", ".join(amenities[:3]) if amenities else "Gated Community, Security, Parking"

                reply = (
                    f"Hi {clean_name}! Here is an excellent property that matches what you're looking for:\n\n"
                    f"🏡 *{p.get('title')}*\n"
                    f"📍 *Location:* {p.get('locality')}, {p.get('city')}\n"
                    f"📐 *Configuration:* {p.get('bhk')} ({p.get('area_sqft', '')} sq.ft)\n"
                    f"💰 *Price:* ₹{p.get('price_lakhs')} Lakhs ({p.get('status', 'Ready to Move')})\n"
                    f"✨ *Key Highlights:* {amenities_str}\n\n"
                    f"🚗 *Complimentary VIP Cab Service:*\n"
                    f"We provide a private AC cab to pick you and your family up directly from your doorstep and bring you back home—completely complimentary with zero obligation.\n\n"
                    f"Would you like to visit this week? What day or time works best for you?"
                )
                session_data["state"] = "awaiting_visit_time"
                return reply, session_data

        # 8. General Consultative Assistance
        reply = (
            f"Hi {clean_name}! I'd be glad to assist you with your property search in Nagpur or Pune.\n\n"
            f"Could you let me know which locality you prefer, your preferred configuration (1BHK, 2BHK, 3BHK, or Villa), "
            f"and your comfortable budget range? I will handpick the best verified options for you! 🏡"
        )
        return reply, session_data
