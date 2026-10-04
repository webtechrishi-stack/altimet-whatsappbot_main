"""
Campaign Lead Import & File Parsing Service.
Supports Excel (.xlsx, .xls) and CSV file uploads for personalized WhatsApp campaigns.
Extracts customer contact information, requirements (BHK, budget, location, purpose),
validates phone numbers, synchronizes to CRM Leads and Customer Memory, and enables
targeted broadcast messaging.
"""

import io
import csv
import re
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timezone

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

from database import DB
from conversation.memory_service import CustomerMemoryService

logger = logging.getLogger(__name__)


class CampaignLeadImporter:
    """
    Parses, validates, and imports leads from Excel or CSV files.
    """

    HEADER_MAPPINGS = {
        "name": [
            "name", "fullname", "full_name", "customer_name", "client_name",
            "lead_name", "contact_name", "first_name", "client"
        ],
        "phone": [
            "phone", "mobile", "number", "whatsapp", "wa_id", "contact",
            "contact_number", "phone_number", "cell", "mobile_number"
        ],
        "city": [
            "city", "location", "preferred_city", "town", "target_city"
        ],
        "locality": [
            "locality", "area", "suburb", "address", "street", "preferred_locality"
        ],
        "bhk": [
            "bhk", "configuration", "preferred_bhk", "config", "type", "flat_type"
        ],
        "budget": [
            "budget", "budget_max", "max_budget", "price", "budget_lakhs", "budget_in_lakhs"
        ],
        "purpose": [
            "purpose", "intent", "buying_purpose", "use", "end_use"
        ],
        "notes": [
            "notes", "remarks", "comment", "comments", "description"
        ]
    }

    @classmethod
    def clean_header_key(cls, header: str) -> str:
        h = re.sub(r"[^a-zA-Z0-9_]", "", str(header).strip().lower().replace(" ", "_"))
        for standard_key, aliases in cls.HEADER_MAPPINGS.items():
            if h in aliases or any(a in h for a in aliases):
                return standard_key
        return h

    @classmethod
    def normalize_phone(cls, raw: Any) -> Tuple[str, bool]:
        """
        Cleans phone number and validates it for WhatsApp (E.164 without +).
        Defaults to Indian country code (91) for 10-digit mobile numbers.
        """
        s = str(raw or "").strip()
        # Remove floating point from Excel reading integer as float (e.g. 919876543210.0)
        if s.endswith(".0"):
            s = s[:-2]
        digits = re.sub(r"\D", "", s)

        if not digits:
            return "", False

        # If 10 digits starting with 6, 7, 8, 9 -> assume India (91)
        if len(digits) == 10 and digits[0] in "6789":
            digits = "91" + digits

        # Standard Indian 12-digit number (91 + 10 digits)
        if len(digits) == 12 and digits.startswith("91") and digits[2] in "6789":
            return digits, True

        # International numbers between 10 and 15 digits
        if 10 <= len(digits) <= 15:
            return digits, True

        return digits, False

    @classmethod
    def normalize_budget(cls, raw: Any) -> Optional[float]:
        """
        Normalizes budget values to Lakhs (e.g. '65', '1.2 Cr', '85L').
        """
        if raw is None or raw == "":
            return None
        try:
            if isinstance(raw, (int, float)):
                return float(raw)
            s = str(raw).lower().strip()
            num_match = re.search(r"(\d+(?:\.\d+)?)", s)
            if not num_match:
                return None
            val = float(num_match.group(1))
            if "cr" in s or "crore" in s:
                val = val * 100.0
            return round(val, 2)
        except Exception:
            return None

    @classmethod
    def normalize_bhk(cls, raw: Any) -> Optional[str]:
        if not raw:
            return None
        s = str(raw).upper().strip()
        m = re.search(r"([1-5])\s*BHK", s)
        if m:
            return f"{m.group(1)}BHK"
        if "VILLA" in s:
            return "Villa"
        if "PLOT" in s:
            return "Plot"
        return s

    @classmethod
    def parse_file(cls, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """
        Parses Excel (.xlsx, .xls) or CSV bytes and returns preview + statistics.
        """
        fn = filename.lower()
        rows_data: List[Dict[str, Any]] = []

        if fn.endswith(".csv"):
            try:
                text_content = file_bytes.decode("utf-8-sig")
            except UnicodeDecodeError:
                text_content = file_bytes.decode("latin-1", errors="replace")

            reader = csv.reader(io.StringIO(text_content))
            rows = list(reader)
            if not rows:
                return {"total_rows": 0, "valid_count": 0, "invalid_count": 0, "leads": [], "columns": []}

            headers = [cls.clean_header_key(h) for h in rows[0]]
            raw_data_rows = rows[1:]

            for r in raw_data_rows:
                if not any(cell.strip() for cell in r if cell):
                    continue
                row_dict = {}
                for idx, h in enumerate(headers):
                    val = r[idx].strip() if idx < len(r) else ""
                    row_dict[h] = val
                rows_data.append(row_dict)

        elif fn.endswith((".xlsx", ".xls")):
            wb = openpyxl.load_workbook(io.BytesIO(file_bytes), data_only=True)
            sheet = wb.active
            all_rows = list(sheet.iter_rows(values_only=True))
            if not all_rows:
                return {"total_rows": 0, "valid_count": 0, "invalid_count": 0, "leads": [], "columns": []}

            headers = [cls.clean_header_key(h or "") for h in all_rows[0]]
            raw_data_rows = all_rows[1:]

            for r in raw_data_rows:
                if not any(cell is not None and str(cell).strip() for cell in r):
                    continue
                row_dict = {}
                for idx, h in enumerate(headers):
                    val = r[idx] if idx < len(r) else None
                    if val is not None:
                        row_dict[h] = str(val).strip()
                    else:
                        row_dict[h] = ""
                rows_data.append(row_dict)
        else:
            raise ValueError("Unsupported file format. Please upload an Excel (.xlsx, .xls) or CSV (.csv) file.")

        # Process and normalize rows
        parsed_leads = []
        valid_count = 0
        invalid_count = 0

        for idx, row in enumerate(rows_data, 1):
            raw_name = row.get("name") or f"Lead {idx}"
            raw_phone = row.get("phone") or ""
            norm_phone, is_valid = cls.normalize_phone(raw_phone)

            city = row.get("city") or ""
            locality = row.get("locality") or ""
            bhk = cls.normalize_bhk(row.get("bhk"))
            budget = cls.normalize_budget(row.get("budget"))
            purpose = row.get("purpose") or ""
            notes = row.get("notes") or ""

            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1

            lead_entry = {
                "row_index": idx,
                "name": raw_name,
                "raw_phone": raw_phone,
                "phone": norm_phone,
                "is_valid": is_valid,
                "city": city,
                "locality": locality,
                "bhk": bhk,
                "budget_max": budget,
                "purpose": purpose,
                "notes": notes,
                "status": "VALID" if is_valid else "INVALID_PHONE"
            }
            parsed_leads.append(lead_entry)

        return {
            "total_rows": len(parsed_leads),
            "valid_count": valid_count,
            "invalid_count": invalid_count,
            "leads": parsed_leads,
            "columns": list(cls.HEADER_MAPPINGS.keys())
        }

    @classmethod
    def import_and_persist(cls, leads: List[Dict[str, Any]], campaign_title: str = "") -> List[str]:
        """
        Creates or updates leads in MongoDB and synchronizes CustomerMemory.
        Returns list of lead_ids created/updated.
        """
        lead_ids: List[str] = []
        memory_svc = CustomerMemoryService()

        for item in leads:
            if not item.get("is_valid") and not item.get("phone"):
                continue

            phone = str(item.get("phone", "")).strip()
            if not phone:
                continue

            name = item.get("name") or "Valued Client"
            city = item.get("city") or "Nagpur"
            locality = item.get("locality") or ""
            bhk = item.get("bhk") or "2BHK"
            budget = item.get("budget_max")
            purpose = item.get("purpose") or "Self-Use"

            # 1. Upsert into CRM Leads collection
            lead = DB.leads.get_or_create(wa_id=phone, name=name, phone=phone)
            lead_id = lead.get("lead_id")

            updates = {
                "name": name,
                "preferred_city": city,
                "sales_stage": "NEW",
                "lead_score": max(lead.get("lead_score", 0), 25),
                "source": "CAMPAIGN_IMPORT",
                "consent_given": True,
                "marketing_opt_in": True,
                "updated_at": datetime.now(timezone.utc)
            }
            if locality:
                updates["locality"] = locality
            if bhk:
                updates["bhk"] = [bhk] if not isinstance(bhk, list) else bhk
            if budget:
                updates["budget_max"] = budget
            if purpose:
                updates["purpose"] = purpose

            DB.leads.update(lead_id, updates)

            # 2. Update CustomerMemory with explicit facts
            try:
                memory_svc.save_explicit_facts(
                    lead_id=lead_id,
                    wa_id=phone,
                    facts={
                        "name": name,
                        "city": city,
                        "locality": locality,
                        "bhk": bhk,
                        "budget_max": budget,
                        "purpose": purpose
                    }
                )
            except Exception as ex:
                logger.warning(f"[IMPORT MEMORY WARNING] {ex}")

            lead_ids.append(lead_id)

        return lead_ids

    @classmethod
    def generate_sample_template(cls) -> bytes:
        """
        Generates an Excel template with proper formatting, column headers,
        and example data rows to help users prepare their import files.
        """
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "ARIS Leads"

        headers = [
            "Customer Name",
            "WhatsApp Number",
            "Preferred City",
            "Locality / Area",
            "BHK Required",
            "Budget in Lakhs",
            "Buying Purpose",
            "Notes / Remarks"
        ]

        sample_rows = [
            ["Rishi Sharma", "918600079496", "Nagpur", "Manish Nagar", "2BHK", 65, "Self-Use", "Interested in ready-to-move flats"],
            ["Dr. Anita Patel", "919876543210", "Nagpur", "Dharampeth", "3BHK", 120, "Self-Use", "Prefers gated society with lift"],
            ["Vikram Deshmukh", "919822334455", "Pune", "Kharadi", "2BHK", 78, "Investment", "Looking for rental yield near IT park"],
            ["Meera Kulkarni", "919766554433", "Pune", "Baner", "3BHK", 150, "Self-Use", "Spouse interested in weekend visit"],
        ]

        # Styling
        header_fill = PatternFill(start_color="0F172A", end_color="0F172A", fill_type="solid")
        header_font = Font(name="Segoe UI", size=11, bold=True, color="FFFFFF")
        data_font = Font(name="Segoe UI", size=10, color="000000")
        thin_border = Border(
            left=Side(style='thin', color='CBD5E1'),
            right=Side(style='thin', color='CBD5E1'),
            top=Side(style='thin', color='CBD5E1'),
            bottom=Side(style='thin', color='CBD5E1')
        )

        ws.append(headers)
        for col_num, h in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col_num)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = thin_border

        for r in sample_rows:
            ws.append(r)

        for row in ws.iter_rows(min_row=2, max_row=len(sample_rows) + 1, min_col=1, max_col=len(headers)):
            for cell in row:
                cell.font = data_font
                cell.border = thin_border
                cell.alignment = Alignment(vertical="center")

        # Auto column widths
        col_widths = [22, 20, 18, 20, 16, 18, 18, 35]
        for idx, width in enumerate(col_widths, 1):
            col_letter = openpyxl.utils.get_column_letter(idx)
            ws.column_dimensions[col_letter].width = width

        ws.row_dimensions[1].height = 28
        for r_idx in range(2, len(sample_rows) + 2):
            ws.row_dimensions[r_idx].height = 22

        buf = io.BytesIO()
        wb.save(buf)
        buf.seek(0)
        return buf.getvalue()
