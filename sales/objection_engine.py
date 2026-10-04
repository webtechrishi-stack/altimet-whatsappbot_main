"""
Objection Engine — detects customer objections and selects consultative, truthful resolution strategies.
Strictly prohibits fabricated discounts, artificial urgency, or unverified scarcity.
"""

import re
from typing import Dict, Any, Optional, Tuple, List


class ObjectionEngine:
    """
    Detects and categorizes customer objections, mapping them to truthful value-oriented strategies.
    """

    PATTERNS: Dict[str, List[str]] = {
        "PRICE": [
            r"\b(?:expensive|too high|costly|price is high|out of budget|budget tight|overpriced|steep)\b",
            r"\b(?:discount|any offer|best price|less price|negotiable|reduce price)\b",
            r"\b(?:higher than (?:my )?budget|over (?:my )?budget)\b",
            r"\bprice\b.*?\b(?:high|expensive|steep|costly)\b"
        ],
        "NEED_TO_THINK": [
            r"\b(?:need to think|let me think|will think about it|give me time|have to consider)\b",
            r"\b(?:will check and tell|will let you know|will get back|need some time)\b"
        ],
        "SPOUSE_NOT_CONVINCED": [
            r"\b(?:discuss|talk|ask)\b.*?\b(?:wife|husband|spouse|family|parents|father|mother|partner)\b",
            r"\b(?:family (?:is )?not (?:convinced|sure)|need family approval)\b"
        ],
        "LOCATION": [
            r"\b(?:too far|location is far|distance is high|far from (?:city|office|station|metro))\b",
            r"\b(?:connectivity (?:is )?poor|isolated area|outer area)\b"
        ],
        "NOT_READY": [
            r"\b(?:not ready (?:yet|now)|planning (?:after|next year)|buying later|just looking)\b",
            r"\b(?:not in a hurry|few months later)\b"
        ],
        "WANT_MORE_OPTIONS": [
            r"\b(?:more options|other projects|other properties|anything else|different options)\b",
            r"\b(?:show me more|more flats)\b"
        ],
        "WANT_BROCHURE": [
            r"\b(?:send brochure|share brochure|brochure pdf|project details pdf|send pdf)\b",
            r"\b(?:send catalog|floor plan pdf)\b"
        ],
        "WANT_TO_COMPARE": [
            r"\b(?:comparing with|better than|vs\b|versus|which one is better)\b"
        ],
        "TRUST": [
            r"\b(?:rera approved|builder reputation|legal issue|clear title|fraud|reliable builder)\b",
            r"\b(?:completion certificate|quality of construction)\b"
        ],
        "POSSESSION": [
            r"\b(?:possession delay|when will i get possession|delayed|handover date)\b"
        ],
        "FINANCING": [
            r"\b(?:home loan|loan eligibility|interest rate|bank loan|emi options|sbi approval)\b"
        ],
        "NO_TIME": [
            r"\b(?:no time|very busy|travelling|out of town|tied up|can't visit now)\b"
        ]
    }

    STRATEGIES: Dict[str, Dict[str, str]] = {
        "PRICE": {
            "strategy": "VALUE_CLARITY_AND_ALTERNATIVES",
            "guideline": (
                "Acknowledge budget sensitivity warmly. Highlight verified value (carpet area efficiency, "
                "amenities, location connectivity) that justifies the pricing. If budget is strictly firm, "
                "mention a verified alternative configuration or flexible bank-approved payment milestones. "
                "NEVER fabricate discounts or claim fake urgent price drops."
            )
        },
        "NEED_TO_THINK": {
            "strategy": "DISCOVER_UNCERTAINTY",
            "guideline": (
                "Respectfully acknowledge their thoughtful approach. Gently inquire what specific aspect "
                "(budget, layout, location, or possession timeline) they'd like more clarity on. "
                "Do NOT pressure the customer."
            )
        },
        "SPOUSE_NOT_CONVINCED": {
            "strategy": "INVOLVE_FAMILY_CONVENIENCE",
            "guideline": (
                "Validate that home buying is a shared family decision. Offer to share complete project specs, "
                "walkthrough videos, or suggest bringing the family for a relaxed, zero-obligation weekend visit "
                "with complimentary AC doorstep cab pickup so everyone can experience the space firsthand."
            )
        },
        "LOCATION": {
            "strategy": "INFRASTRUCTURE_FACTS",
            "guideline": (
                "Provide verified travel times to key hubs (metro stations, IT corridors, airport, hospitals). "
                "Explain upcoming infrastructure developments without making unsupported ROI claims."
            )
        },
        "NOT_READY": {
            "strategy": "LOW_PRESSURE_NURTURING",
            "guideline": (
                "Reassure the client there is zero rush. Offer to keep them updated on major milestones "
                "and share the verified brochure so they have the facts ready whenever their timeline matures."
            )
        },
        "WANT_MORE_OPTIONS": {
            "strategy": "EXPLORE_CATALOG",
            "guideline": (
                "Present 2 alternative verified projects matching their city and budget. Ask which specific trade-off "
                "(more carpet area vs closer locality) matters most to them."
            )
        },
        "WANT_BROCHURE": {
            "strategy": "BROCHURE_PLUS_CONSULTATION",
            "guideline": (
                "Confirm that the project brochure and master layout are ready. Highlight key sections "
                "(floor plan layouts, payment schedules, clubhouse amenities) and ask which aspect to focus on."
            )
        },
        "WANT_TO_COMPARE": {
            "strategy": "OBJECTIVE_COMPARISON",
            "guideline": (
                "Provide a clear, objective comparison on carpet area, density, amenities, and price per sq.ft. "
                "based strictly on verified data without badmouthing competitors."
            )
        },
        "TRUST": {
            "strategy": "VERIFIED_CREDENTIALS",
            "guideline": (
                "Cite official MahaRERA registration numbers, clear bank pre-approvals (SBI, HDFC, ICICI), "
                "and the developer's track record of timely delivery."
            )
        },
        "POSSESSION": {
            "strategy": "MILESTONES_AND_RERA_TIMELINE",
            "guideline": (
                "State the exact RERA possession timeline and current construction status (e.g. structure completed, "
                "finishing in progress). Never give unverified earlier dates."
            )
        },
        "FINANCING": {
            "strategy": "FINANCIAL_ASSISTANCE",
            "guideline": (
                "Explain available home loan assistance with leading nationalized banks, customized EMI calculations, "
                "and construction-linked payment schedules."
            )
        },
        "NO_TIME": {
            "strategy": "VIP_CONVENIENCE_CAB",
            "guideline": (
                "Offer complete schedule flexibility: 'We can arrange our complimentary private AC cab to pick you up "
                "directly from your office or home at whatever hour or weekend works best for you!'"
            )
        }
    }

    @classmethod
    def detect_objection(cls, text: str) -> Optional[str]:
        """Detects if incoming text matches any known objection category."""
        clean = (text or "").lower()
        for category, patterns in cls.PATTERNS.items():
            for pat in patterns:
                if re.search(pat, clean, re.IGNORECASE):
                    return category
        return None

    @classmethod
    def get_strategy(cls, objection_type: str) -> Dict[str, str]:
        """Retrieves structured strategy for an objection."""
        ot = (objection_type or "").upper()
        return cls.STRATEGIES.get(ot, {
            "strategy": "CONSULTATIVE_CLARIFICATION",
            "guideline": "Address the customer's question directly with verified facts and empathy."
        })
