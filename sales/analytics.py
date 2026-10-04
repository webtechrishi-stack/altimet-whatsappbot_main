"""
Sales Funnel Analytics & Conversion Metrics service.
Computes funnel stage conversions, durations, and milestone timestamps.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from database import DB


class SalesAnalyticsService:
    """
    Computes real-time sales funnel metrics from MongoDB.
    """

    STAGES = [
        "NEW",
        "DISCOVERY",
        "QUALIFIED",
        "PROPERTY_RECOMMENDED",
        "ENGAGED",
        "VISIT_PITCHED",
        "VISIT_BOOKED",
        "VISIT_COMPLETED",
        "WON"
    ]

    def __init__(self):
        self.conv_repo = DB.conversations
        self.lead_repo = DB.leads
        self.visit_repo = DB.visits
        self.sales_event_repo = DB.sales_events

    def get_funnel_metrics(self) -> Dict[str, Any]:
        """
        Calculates conversion funnel counts and drop-off percentages.
        """
        all_convs = self.conv_repo.list_conversations(limit=2000)
        total_leads = len(all_convs) or 1

        stage_counts = {s: 0 for s in self.STAGES}
        for c in all_convs:
            st = c.get("sales_stage", "NEW").upper()
            if st in stage_counts:
                stage_counts[st] += 1
            elif "VISIT" in st:
                stage_counts["VISIT_PITCHED"] += 1
            else:
                stage_counts["NEW"] += 1

        # Calculate cumulative funnel progress
        funnel_progression = []
        cumulative = len(all_convs)
        for s in self.STAGES:
            count = stage_counts.get(s, 0)
            funnel_progression.append({
                "stage": s,
                "current_count": count,
                "percentage_of_total": round((count / total_leads) * 100, 1)
            })

        # Booked visits count
        confirmed_visits = self.visit_repo.count_visits({"status": "CONFIRMED"}) if hasattr(self.visit_repo, "count_visits") else len(self.visit_repo.list_visits(limit=500))

        return {
            "total_conversations": len(all_convs),
            "stage_breakdown": stage_counts,
            "funnel_progression": funnel_progression,
            "confirmed_visits": confirmed_visits,
            "conversion_rate_to_visit": round((confirmed_visits / total_leads) * 100, 2)
        }
