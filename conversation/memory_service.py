"""
Customer Memory Service — extracts, updates, and persists customer requirements,
strictly distinguishing explicit customer facts from inferred preferences.
"""

import re
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

from database import DB
from conversation.models import CustomerMemory, FactOrInference

logger = logging.getLogger(__name__)


class CustomerMemoryService:
    """
    Manages persistent Customer Memory for leads, extracting and updating
    requirements with strict explicit-facts-over-inference rules.
    """

    KNOWN_CITIES = ["Nagpur", "Pune", "Mumbai", "Bangalore", "Hyderabad", "Delhi"]
    KNOWN_LOCALITIES = [
        # Nagpur
        "MIHAN", "Manish Nagar", "Besa", "Wardha Road", "Dharampeth",
        "Civil Lines", "Ramdaspeth", "Somalwada", "Chatrapati Square",
        "Beltarodi", "Shankarpur", "Trimurti Nagar", "Pratap Nagar",
        # Pune
        "Kharadi", "Baner", "Wakad", "Hinjewadi", "Viman Nagar",
        "Kothrud", "Hadapsar", "Aundh", "Balewadi", "Bavdhan",
        # Bangalore
        "Whitefield", "Sarjapur", "Electronic City", "Bellandur", "Indiranagar",
        "Koramangala", "HSR Layout", "Hebbal", "Panathur Road", "Varthur", "Yelahanka"
    ]

    @classmethod
    def get_memory(cls, phone_or_lead_id: str) -> Optional[CustomerMemory]:
        svc = cls()
        lead = DB.leads.get_by_phone(phone_or_lead_id)
        lead_id = lead.get("lead_id") if isinstance(lead, dict) else (getattr(lead, "id", None) if lead else phone_or_lead_id)
        return svc.get_or_create_memory(str(lead_id or phone_or_lead_id))

    @classmethod
    def extract_and_update(cls, phone_or_lead_id: str, conversation_id: str, message_text: str) -> CustomerMemory:
        svc = cls()
        lead = DB.leads.get_by_phone(phone_or_lead_id)
        lead_id = lead.get("lead_id") if isinstance(lead, dict) else (getattr(lead, "id", None) if lead else phone_or_lead_id)
        mem = svc.update_from_message(str(lead_id or phone_or_lead_id), message_text)
        if hasattr(mem, "phone") and not mem.phone:
            mem.phone = phone_or_lead_id
        return mem

    def __init__(self):
        self.memory_repo = DB.customer_memory
        self.lead_repo = DB.leads

    def get_or_create_memory(self, lead_id: str) -> CustomerMemory:
        """Loads existing memory or initializes new record."""
        if not lead_id:
            return CustomerMemory(lead_id="unknown")

        data = self.memory_repo.get_by_lead_id(lead_id)
        if data:
            try:
                return CustomerMemory.from_dict(data)
            except Exception as ex:
                logger.warning(f"[MEMORY] Could not parse memory dict for {lead_id}: {ex}")

        # Try populating from lead record if available
        lead = self.lead_repo.get_by_id(lead_id)
        mem = CustomerMemory(lead_id=str(lead_id))
        if lead:
            if lead.get("preferred_city"):
                mem.requirements["city"] = lead["preferred_city"]
                mem.explicit_facts["city"] = FactOrInference(lead["preferred_city"], "CUSTOMER", 1.0).to_dict()
            if lead.get("preferred_location"):
                mem.requirements["locality"] = lead["preferred_location"]
                mem.explicit_facts["locality"] = FactOrInference(lead["preferred_location"], "CUSTOMER", 1.0).to_dict()
            if lead.get("bhk"):
                mem.requirements["bhk"] = lead["bhk"]
                mem.explicit_facts["bhk"] = FactOrInference(lead["bhk"], "CUSTOMER", 1.0).to_dict()
            if lead.get("budget_max"):
                mem.requirements["budget_max"] = float(lead["budget_max"])
                mem.explicit_facts["budget_max"] = FactOrInference(float(lead["budget_max"]), "CUSTOMER", 1.0).to_dict()
            if lead.get("budget_min"):
                mem.requirements["budget_min"] = float(lead["budget_min"])
                mem.explicit_facts["budget_min"] = FactOrInference(float(lead["budget_min"]), "CUSTOMER", 1.0).to_dict()

        self.memory_repo.save_or_update(lead_id, mem.to_dict())
        return mem

    def update_from_message(
        self,
        lead_id: str,
        message_text: str,
        existing_memory: Optional[CustomerMemory] = None
    ) -> CustomerMemory:
        """
        Extracts new facts & preferences from customer message and merges them safely.
        Never throws unhandled exceptions that block customer response.
        """
        try:
            memory = existing_memory or self.get_or_create_memory(lead_id)
            text = (message_text or "").strip()
            lower = text.lower()

            extracted_facts: Dict[str, Any] = {}
            extracted_inferences: Dict[str, Any] = {}

            # 1. City extraction
            city = self._extract_city(text)
            if city:
                extracted_facts["city"] = city

            # 2. Locality extraction
            locality = self._extract_locality(text)
            if locality:
                extracted_facts["locality"] = locality

            # 3. BHK extraction
            bhk = self._extract_bhk(text)
            if bhk:
                extracted_facts["bhk"] = bhk

            # 4. Budget extraction
            b_min, b_max = self._extract_budget(text)
            if b_max is not None:
                extracted_facts["budget_max"] = b_max
            if b_min is not None:
                extracted_facts["budget_min"] = b_min

            # 5. Purpose extraction
            purpose = self._extract_purpose(text)
            if purpose:
                extracted_facts["purpose"] = purpose

            # 6. Timeline extraction
            timeline = self._extract_timeline(text)
            if timeline:
                extracted_facts["timeline"] = timeline

            # 7. Inferred price sensitivity
            if any(w in lower for w in ["expensive", "too high", "negotiable", "discount", "budget friendly", "cheaper"]):
                extracted_inferences["price_sensitivity"] = "HIGH"

            # 8. Preferences extraction (amenities, view, etc.)
            for pref in ["swimming pool", "gym", "garden", "balcony", "metro", "vastu", "clubhouse", "security"]:
                if pref in lower and pref not in memory.preferences:
                    memory.preferences.append(pref)

            # Apply updates with strict explicit fact precedence
            changed = False
            for k, val in extracted_facts.items():
                old_val = memory.requirements.get(k)
                if old_val != val:
                    memory.requirements[k] = val
                    memory.explicit_facts[k] = FactOrInference(
                        value=val,
                        source="CUSTOMER",
                        confidence=1.0,
                        timestamp=datetime.now(timezone.utc)
                    ).to_dict()
                    memory.history_log.append({
                        "field": k,
                        "old_value": old_val,
                        "new_value": val,
                        "source": "CUSTOMER",
                        "text_snippet": text[:80],
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                    changed = True

            # Inferences only set if explicit fact is not established
            for k, val in extracted_inferences.items():
                if k not in memory.explicit_facts:
                    memory.inferred_preferences[k] = FactOrInference(
                        value=val,
                        source="INFERRED",
                        confidence=0.75,
                        timestamp=datetime.now(timezone.utc)
                    ).to_dict()
                    changed = True

            # Update customer summary
            memory.customer_summary = self._generate_summary(memory)
            memory.updated_at = datetime.now(timezone.utc)

            # Persist to database
            self.memory_repo.save_or_update(lead_id, memory.to_dict())

            # Sync to CRM Lead entity
            self._sync_to_lead(lead_id, memory)

            return memory

        except Exception as ex:
            logger.error(f"[MEMORY ERROR] Failed to update customer memory for {lead_id}: {ex}")
            return existing_memory or self.get_or_create_memory(lead_id)

    def record_objection(self, lead_id: str, objection_type: str, details: str = "") -> None:
        """Records an objection in customer memory."""
        try:
            mem = self.get_or_create_memory(lead_id)
            obj = {
                "type": str(objection_type).upper(),
                "status": "OPEN",
                "details": details,
                "recorded_at": datetime.now(timezone.utc).isoformat()
            }
            # Avoid duplicate open objections of same type
            existing = [o for o in mem.objections if o.get("type") == obj["type"] and o.get("status") == "OPEN"]
            if not existing:
                mem.objections.append(obj)
                mem.updated_at = datetime.now(timezone.utc)
                self.memory_repo.save_or_update(lead_id, mem.to_dict())
        except Exception as ex:
            logger.error(f"[MEMORY ERROR] Could not record objection: {ex}")

    def resolve_objection(self, lead_id: str, objection_type: str) -> None:
        """Marks an objection as RESOLVED in customer memory."""
        try:
            mem = self.get_or_create_memory(lead_id)
            ot = str(objection_type).upper()
            updated = False
            for o in mem.objections:
                if o.get("type") == ot and o.get("status") == "OPEN":
                    o["status"] = "RESOLVED"
                    o["resolved_at"] = datetime.now(timezone.utc).isoformat()
                    updated = True
            if updated:
                mem.updated_at = datetime.now(timezone.utc)
                self.memory_repo.save_or_update(lead_id, mem.to_dict())
        except Exception as ex:
            logger.error(f"[MEMORY ERROR] Could not resolve objection: {ex}")

    def record_recommended_property(self, lead_id: str, property_title_or_id: str) -> None:
        """Tracks properties recommended to this customer."""
        try:
            mem = self.get_or_create_memory(lead_id)
            clean = str(property_title_or_id).strip()
            if clean and clean not in mem.recommended_properties:
                mem.recommended_properties.append(clean)
                mem.updated_at = datetime.now(timezone.utc)
                self.memory_repo.save_or_update(lead_id, mem.to_dict())
        except Exception as ex:
            logger.error(f"[MEMORY ERROR] Could not record recommended property: {ex}")

    def record_visit_interest(self, lead_id: str, interested: bool = True, status: str = "INTERESTED") -> None:
        """Tracks customer interest in visiting a property."""
        try:
            mem = self.get_or_create_memory(lead_id)
            mem.visit["interest"] = bool(interested)
            mem.visit["status"] = status
            mem.updated_at = datetime.now(timezone.utc)
            self.memory_repo.save_or_update(lead_id, mem.to_dict())
        except Exception as ex:
            logger.error(f"[MEMORY ERROR] Could not update visit interest: {ex}")

    # =========================================================================
    # Extraction Helpers
    # =========================================================================

    def _extract_city(self, text: str) -> Optional[str]:
        for c in self.KNOWN_CITIES:
            if re.search(r"\b" + re.escape(c) + r"\b", text, re.IGNORECASE):
                return c
        return None

    def _extract_locality(self, text: str) -> Optional[str]:
        for loc in self.KNOWN_LOCALITIES:
            if re.search(r"\b" + re.escape(loc) + r"\b", text, re.IGNORECASE):
                return loc
        return None

    def _extract_bhk(self, text: str) -> Optional[str]:
        match = re.search(r"\b([1-5])\s*(?:bhk|bedroom|bed)\b", text, re.IGNORECASE)
        if match:
            return f"{match.group(1)}BHK"
        if re.search(r"\bvilla\b", text, re.IGNORECASE):
            return "Villa"
        if re.search(r"\bpenthouse\b", text, re.IGNORECASE):
            return "Penthouse"
        return None

    def _extract_budget(self, text: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Extracts budget values in Lakhs (e.g. '70 lakh' -> 70.0, '1.2 cr' -> 120.0).
        Supports stretch budgets (e.g. 'stretch to 75' -> 75.0).
        """
        # 1. Crore range pattern e.g. "1.5 to 1.8 Cr", "1.5 - 1.8 Cr"
        cr_range = re.search(r"(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s*(?:cr|crore)", text, re.IGNORECASE)
        if cr_range:
            return float(cr_range.group(1)) * 100.0, float(cr_range.group(2)) * 100.0

        # 2. Single Crore pattern e.g. "1.8 Cr"
        cr_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:cr|crore)", text, re.IGNORECASE)
        if cr_match:
            val = float(cr_match.group(1)) * 100.0  # Convert Cr to Lakhs
            return None, val

        # 2. Stretch pattern e.g., "stretch to 75", "can go up to 80"
        stretch_match = re.search(r"(?:stretch|go up to|budget is|maximum|upto|under)\s*(?:to\s*)?(\d+(?:\.\d+)?)\s*(?:lakh|lakhs|l)?", text, re.IGNORECASE)
        if stretch_match:
            try:
                val = float(stretch_match.group(1))
                if 10.0 <= val <= 1000.0:
                    return None, val
            except Exception:
                pass

        # 3. Standard Lakhs range e.g. "60 to 75 lakh", "60-75L"
        range_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s*(?:lakh|lakhs|l)?", text, re.IGNORECASE)
        if range_match:
            try:
                low = float(range_match.group(1))
                high = float(range_match.group(2))
                if 10.0 <= low <= high <= 1000.0:
                    return low, high
            except Exception:
                pass

        # 4. Standard single Lakhs mention e.g. "70 lakh", "70L"
        single_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:lakh|lakhs|l)\b", text, re.IGNORECASE)
        if single_match:
            try:
                val = float(single_match.group(1))
                if 10.0 <= val <= 1000.0:
                    return None, val
            except Exception:
                pass

        return None, None

    def _extract_purpose(self, text: str) -> Optional[str]:
        lower = text.lower()
        if any(w in lower for w in ["self use", "self-use", "own use", "family", "live", "personal", "stay"]):
            return "END_USE"
        if any(w in lower for w in ["investment", "rental", "rent", "invest", "roi", "returns"]):
            return "INVESTMENT"
        return None

    def _extract_timeline(self, text: str) -> Optional[str]:
        lower = text.lower()
        if any(w in lower for w in ["immediate", "ready to move", "ready possession", "immediately"]):
            return "READY_TO_MOVE"
        if any(w in lower for w in ["under construction", "next year", "6 months", "1 year", "2026", "2027"]):
            return "UNDER_CONSTRUCTION"
        return None

    def _generate_summary(self, mem: CustomerMemory) -> str:
        req = mem.requirements
        parts = []
        if req.get("bhk"):
            parts.append(str(req["bhk"]))
        if req.get("locality"):
            parts.append(f"in {req['locality']}")
        elif req.get("city"):
            parts.append(f"in {req['city']}")

        if req.get("budget_max"):
            if req.get("budget_min"):
                parts.append(f"budget ₹{req['budget_min']}L–₹{req['budget_max']}L")
            else:
                parts.append(f"budget around ₹{req['budget_max']}L")

        purpose_str = "self-use" if req.get("purpose") == "END_USE" else ("investment" if req.get("purpose") == "INVESTMENT" else "")
        if purpose_str:
            parts.append(f"for {purpose_str}")

        if mem.recommended_properties:
            parts.append(f"Interested in: {', '.join(mem.recommended_properties[:2])}")

        open_objs = [o.get("type") for o in mem.objections if o.get("status") == "OPEN"]
        if open_objs:
            parts.append(f"Open concerns: {', '.join(open_objs)}")

        if not parts:
            return "Customer inquiring about available residential inventory."
        return "Customer looking for " + " ".join(parts) + "."

    def _sync_to_lead(self, lead_id: str, mem: CustomerMemory) -> None:
        """Syncs normalized requirements to CRM Lead entity."""
        try:
            req = mem.requirements
            updates: Dict[str, Any] = {}
            if req.get("city"):
                updates["preferred_city"] = req["city"]
            if req.get("locality"):
                updates["preferred_location"] = req["locality"]
            if req.get("bhk"):
                updates["bhk"] = req["bhk"]
            if req.get("budget_max"):
                updates["budget_max"] = float(req["budget_max"])
            if req.get("budget_min"):
                updates["budget_min"] = float(req["budget_min"])
            if req.get("purpose"):
                updates["investment_intent"] = req["purpose"]
            if req.get("timeline"):
                updates["timeline"] = req["timeline"]

            if updates:
                self.lead_repo.update(lead_id, updates)
                if getattr(mem, "phone", None):
                    self.lead_repo.update_by_wa_id(mem.phone, updates)
        except Exception as ex:
            logger.warning(f"[MEMORY] Lead sync warning: {ex}")
