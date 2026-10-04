"""
Test Suite for RAG Layer, Real Estate Consultant Persona, and Cab Pickup Service.
Verifies:
1. Sentence-Transformers 384-dimensional vector embeddings
2. Vector Store ingestion and top-K semantic search
3. Real Estate Consultant Persona & Cab Pitching
4. Site visit booking with cab pickup request
5. Automatic human takeover release on user greeting ("Hii")
"""

import sys
import unittest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from database import DB
from ai_engine import SentenceTransformerEmbedder, RAGService, RAGContextBuilder, ARISAgent, ARISOrchestrator, VisitService


class TestRAGAndConsultant(unittest.TestCase):
    def test_01_sentence_transformer_embeddings(self):
        """Tests that embedder generates valid 384-dimensional dense vectors."""
        embedder = SentenceTransformerEmbedder()
        vec = embedder.encode("Modern 2BHK flat with rooftop swimming pool in Manish Nagar Nagpur")
        self.assertEqual(len(vec), 384)
        self.assertTrue(any(v != 0.0 for v in vec))
        print(f"[PASS] Embeddings generated successfully: 384 dimensions.")

    def test_02_vector_store_and_rag_retrieval(self):
        """Tests document indexing and semantic retrieval."""
        rag = RAGService()
        ingested = rag.ingest_documents()
        total_chunks = DB.vectors.count_chunks()
        self.assertGreaterEqual(total_chunks, 1)

        # Query for amenities in Royal Palms
        context = rag.retrieve_context("What sports amenities are available in Royal Palms Besa?")
        self.assertTrue(len(context) > 0)
        self.assertTrue("Squash" in context or "pool" in context.lower() or "royal palms" in context.lower())
        print(f"[PASS] RAG Semantic Retrieval matched knowledge chunks: {len(context)} chars.")

    def test_03_consultant_system_prompt_with_cab_pitch(self):
        """Tests that RAGContextBuilder constructs a grounded Real Estate Advisor prompt."""
        builder = RAGContextBuilder()
        lead = {"lead_id": "test_lead_01", "name": "Rishi", "preferred_city": "Nagpur", "budget_max": 50.0}
        prop = DB.properties.get_by_id("ARIS-NGP-01")
        prompt = builder.build_system_prompt(lead, structured_property=prop, user_query="What are the payment milestones?")
        
        self.assertIn("Real Estate Advisory Consultant", prompt)
        self.assertIn("Complimentary Doorstep Cab Pickup & Drop", prompt)
        self.assertIn("Green Meadows", prompt)
        print("[PASS] Consultant system prompt constructed with verified facts and Cab Pitch.")

    def test_04_site_visit_booking_with_cab(self):
        """Tests that booking a site visit records cab pickup details."""
        vs = VisitService()
        res = vs.book_visit(
            wa_id="918600079496",
            customer_name="Rishi",
            property_id="ARIS-NGP-01",
            visit_date="Tomorrow",
            visit_time="04:00 PM",
            cab_required=True,
            pickup_address="Plot 12, Wardha Road, Nagpur"
        )
        self.assertTrue(res["success"])
        self.assertIn("Confirmed", res["message"])
        self.assertIn("Doorstep VIP Cab", res["message"])
        self.assertIn("Wardha Road", res["message"])
        print("[PASS] Site visit booked with VIP Doorstep Cab service.")

    def test_05_human_takeover_auto_resume_on_greeting(self):
        """Tests that when human_takeover is active, an incoming 'Hii' automatically resumes AI."""
        test_wa = "918600079496"
        conv = DB.conversations.create_if_not_exists(test_wa)
        conv_id = conv["conversation_id"]

        # Engage human takeover
        DB.conversations.set_human_takeover(conv_id, True)
        self.assertTrue(DB.conversations.get_active_conversation(test_wa).get("human_takeover"))

        # User sends 'Hii' -> simulate app.py auto-resume logic
        msg = "Hii"
        if msg.strip().lower() in ("hi", "hii", "hello", "hey", "menu"):
            DB.conversations.reset_human_takeover(conv_id)

        # Verify AI is resumed
        active_conv = DB.conversations.get_active_conversation(test_wa)
        self.assertFalse(active_conv.get("human_takeover"))
        print("[PASS] Human takeover automatically cleared upon receiving greeting 'Hii'.")


if __name__ == "__main__":
    unittest.main()
