import unittest
from datetime import datetime, timezone
from database import DB, CampaignStatus
from whatsapp import CampaignService

class TestCampaignEditAndAudiencePersistence(unittest.TestCase):
    def setUp(self):
        self.cs = CampaignService()
        # Fetch 4 existing leads from DB
        all_leads = DB.leads.list_leads(filter_query={"opted_out": {"$ne": True}}, limit=10)
        self.assertTrue(len(all_leads) >= 4, f"Expected at least 4 test leads, found {len(all_leads)}")
        self.lead_ids = [l["lead_id"] for l in all_leads[:4]]

    def test_selected_leads_persistence_and_integrity(self):
        # 1. Create a campaign with exactly 4 selected leads
        camp = self.cs.create_campaign(
            title="Test 4 Leads Selection",
            template_name="aris_lead_outreach",
            audience_type="SELECTED_LEADS",
            selected_lead_ids=self.lead_ids,
            created_by="test_admin@aris.ai"
        )
        camp_id = camp["campaign_id"]

        # 2. Verify campaign record in database strictly retained audience_type & selected_lead_ids
        saved = DB.campaigns.get_by_id(camp_id)
        self.assertIsNotNone(saved)
        self.assertEqual(saved.get("audience_type"), "SELECTED_LEADS")
        self.assertEqual(saved.get("selected_lead_ids"), self.lead_ids)
        self.assertEqual(len(saved.get("selected_lead_ids")), 4)

        # 3. Verify get_eligible_audience returns exactly 4 leads, NOT all 13
        eligible = self.cs.get_eligible_audience(
            city=saved.get("target_city"),
            stage=saved.get("target_stage"),
            min_score=saved.get("min_lead_score", 0),
            bhk_filter=saved.get("bhk_filter"),
            audience_type=saved.get("audience_type", "FILTER"),
            selected_lead_ids=saved.get("selected_lead_ids"),
        )
        self.assertEqual(len(eligible), 4, f"Expected exactly 4 leads, but got {len(eligible)}!")

        # 4. Update campaign: reduce selected leads to 2
        new_selection = self.lead_ids[:2]
        updated = self.cs.update_campaign(
            camp_id,
            {
                "title": "Test 4 Leads Selection (Updated to 2)",
                "selected_lead_ids": new_selection
            },
            updated_by="test_admin@aris.ai"
        )
        self.assertEqual(updated.get("selected_lead_ids"), new_selection)
        self.assertEqual(len(updated.get("selected_lead_ids")), 2)
        self.assertEqual(updated.get("eligible_count"), 2)

        # 5. Remove 1 lead using remove_lead_from_campaign
        lead_to_remove = new_selection[0]
        after_remove = self.cs.remove_lead_from_campaign(camp_id, lead_to_remove, removed_by="test_admin@aris.ai")
        self.assertEqual(len(after_remove.get("selected_lead_ids")), 1)
        self.assertEqual(after_remove.get("selected_lead_ids"), [new_selection[1]])

        # 6. Safety guard: Cannot alter audience while RUNNING
        DB.campaigns.update_campaign(camp_id, {"status": CampaignStatus.RUNNING})
        with self.assertRaises(ValueError) as ctx:
            self.cs.update_campaign(camp_id, {"selected_lead_ids": self.lead_ids})
        self.assertIn("while campaign is RUNNING", str(ctx.exception))

if __name__ == "__main__":
    unittest.main()
