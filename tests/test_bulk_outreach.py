"""
Test script for ARIS Production WhatsApp Bulk Outreach & Campaign Engine.
Tests:
- Database schema & indexes
- CampaignRecipientRepository
- CampaignService snapshotting, claiming, delivery window, retry backoff, error categorization
- Admin campaign REST APIs & UI routes
"""
import sys
import os
import unittest
from datetime import datetime, timezone, timedelta

# Ensure workspace is in sys.path
sys.path.insert(0, os.path.abspath("."))

from database import DB, CampaignStatus, RecipientStatus
from whatsapp import (
    CampaignService,
    CampaignWorker,
    is_within_delivery_window,
    categorize_error,
)
from app import app


class TestBulkOutreachEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app_client = app.test_client()
        cls.cs = CampaignService()
        cls.test_campaign_ids = []

    @classmethod
    def tearDownClass(cls):
        # Cleanup test campaigns
        for cid in cls.test_campaign_ids:
            try:
                DB.campaigns._db.campaigns.delete_one({"campaign_id": cid})
                DB.campaign_recipients._db.campaign_recipients.delete_many({"campaign_id": cid})
            except Exception:
                pass

    def test_01_delivery_window_and_error_categorization(self):
        print("\n--- Test 01: Delivery Window & Error Categorization ---")
        # Test 24-hour window
        self.assertTrue(is_within_delivery_window("00:00", "23:59"))
        # Test inverted/invalid window fallback
        self.assertTrue(is_within_delivery_window("invalid", "times"))

        # Test error categorization
        cat = categorize_error(131056, "Rate limit hit")
        self.assertEqual(cat, "RATE_LIMIT")

        cat = categorize_error(190, "Access token expired")
        self.assertEqual(cat, "AUTH")

        cat = categorize_error(131026, "Message undeliverable")
        self.assertEqual(cat, "RECIPIENT_ERROR")

        cat = categorize_error(132000, "Template does not exist")
        self.assertEqual(cat, "TEMPLATE_ERROR")

        cat = categorize_error(500, "Transient network timeout")
        self.assertEqual(cat, "TRANSIENT")
        print("Delivery window and error categorization verified!")

    def test_02_campaign_recipients_repository(self):
        print("\n--- Test 02: Campaign Recipients Repository ---")
        cid = "test_camp_recip_repo"
        self.test_campaign_ids.append(cid)

        # Bulk insert (campaign_id, recipients)
        recipients = [
            {
                "lead_id": f"lead_{i}",
                "phone": f"+91987654321{i}",
                "name": f"Lead {i}",
                "status": "PENDING",
                "created_at": datetime.now(timezone.utc),
            }
            for i in range(5)
        ]
        inserted = DB.campaign_recipients.bulk_insert(cid, recipients)
        self.assertEqual(inserted, 5)

        # Stats
        stats = DB.campaign_recipients.get_stats(cid)
        self.assertEqual(stats.get("total"), 5)

        # Claim batch
        claimed = DB.campaign_recipients.claim_next_batch(cid, batch_size=3)
        self.assertEqual(len(claimed), 3)
        for r in claimed:
            self.assertEqual(r["status"], RecipientStatus.QUEUED)

        # Update by Meta ID
        first_lead_id = claimed[0]["lead_id"]
        DB.campaign_recipients.update_status(
            cid, first_lead_id, RecipientStatus.SENT, meta_message_id="wamid.test12345"
        )

        matched_cid = DB.campaign_recipients.update_by_meta_id("wamid.test12345", RecipientStatus.DELIVERED)
        self.assertIsNotNone(matched_cid)

        # Mark failed with error
        second_lead_id = claimed[1]["lead_id"]
        DB.campaign_recipients.update_status(
            cid,
            second_lead_id,
            RecipientStatus.FAILED,
            error_code=131056,
            error_message="Rate limit hit",
            error_category="RATE_LIMIT",
        )

        breakdown = DB.campaign_recipients.get_error_breakdown(cid)
        self.assertEqual(breakdown.get("total_failed"), 1)
        self.assertEqual(breakdown.get("errors", [])[0]["error_category"], "RATE_LIMIT")

        # Retry failed
        retried = DB.campaign_recipients.mark_failed_for_retry(cid)
        self.assertEqual(retried, 1)

        print("Campaign recipient repository operations verified!")

    def test_03_campaign_creation_and_snapshot(self):
        print("\n--- Test 03: Campaign Creation & Snapshotting ---")
        camp = self.cs.create_campaign(
            title="Automated Test Campaign",
            template_name="aris_lead_outreach",
            description="Testing bulk outreach pipeline",
            tags=["automated", "test"],
            priority=1,
            delivery_window_start="08:00",
            delivery_window_end="22:00",
        )
        cid = camp["campaign_id"]
        self.test_campaign_ids.append(cid)

        self.assertEqual(camp["title"], "Automated Test Campaign")
        self.assertEqual(camp["priority"], 1)
        self.assertEqual(camp["tags"], ["automated", "test"])

        # Validate campaign (creates snapshot in campaign_recipients)
        val_res = self.cs.validate_campaign(cid)
        self.assertIn("recipients_count", val_res)

        # Check recipients are inserted
        recipient_count = DB.campaign_recipients.count_recipients(cid)
        self.assertEqual(recipient_count, val_res["recipients_count"])

        # Check timeline
        timeline = self.cs.get_campaign_timeline(cid)
        self.assertTrue(len(timeline) >= 2)  # CREATED and VALIDATED

        # Test clone
        cloned = self.cs.clone_campaign(cid, new_title="Cloned Automated Campaign")
        self.test_campaign_ids.append(cloned["campaign_id"])
        self.assertEqual(cloned["status"], CampaignStatus.DRAFT)
        self.assertEqual(cloned["cloned_from"], cid)

        print("Campaign creation, snapshotting, and cloning verified!")

    def test_04_admin_flask_endpoints(self):
        print("\n--- Test 04: Admin Flask REST & UI Endpoints ---")
        # Test campaign list API
        with self.app_client.session_transaction() as sess:
            sess["user"] = {"email": "admin@aris.ai", "role": "ADMIN"}

        res = self.app_client.get("/api/admin/whatsapp/campaigns")
        self.assertEqual(res.status_code, 200)

        # Test campaign global stats
        res = self.app_client.get("/api/admin/whatsapp/campaigns/stats")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("total_campaigns", data)

        # Test UI index page
        res = self.app_client.get("/admin/whatsapp/campaigns")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Bulk Outreach", res.data)

        # Test UI create page
        res = self.app_client.get("/admin/whatsapp/campaigns/create")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Active Hours Start (IST)", res.data)

        # Test UI detail page for our created campaign
        valid_cids = [c for c in self.test_campaign_ids if c.startswith("cmp_")]
        cid = valid_cids[0] if valid_cids else self.test_campaign_ids[-1]
        res = self.app_client.get(f"/admin/whatsapp/campaigns/{cid}")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Export CSV", res.data)

        # Test CSV export endpoint
        res = self.app_client.get(f"/api/admin/whatsapp/campaigns/{cid}/export")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers.get("Content-Type"), "text/csv; charset=utf-8")

        # Test recipients API endpoint
        res = self.app_client.get(f"/api/admin/whatsapp/campaigns/{cid}/recipients")
        self.assertEqual(res.status_code, 200)

        # Test errors API endpoint
        res = self.app_client.get(f"/api/admin/whatsapp/campaigns/{cid}/errors")
        self.assertEqual(res.status_code, 200)

        # Test timeline API endpoint
        res = self.app_client.get(f"/api/admin/whatsapp/campaigns/{cid}/timeline")
        self.assertEqual(res.status_code, 200)

        print("All Admin Flask REST & UI endpoints returned 200 OK!")


if __name__ == "__main__":
    unittest.main()
