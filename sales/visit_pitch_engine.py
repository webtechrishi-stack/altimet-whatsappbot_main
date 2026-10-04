"""
Adaptive Site Visit Pitch Engine.
Chooses the most consultative, contextual strategy to invite customers for a physical tour
without repeating the same generic pitch on every message.
"""

from enum import Enum
from typing import Dict, Any, Optional, List


class PitchStrategy(str, Enum):
    EXPERIENCE = "EXPERIENCE"          # Space & layout reality vs photos
    COMPARISON = "COMPARISON"          # In-person clarity between options
    CONVENIENCE = "CONVENIENCE"        # Complimentary VIP Doorstep Cab pickup
    LOW_COMMITMENT = "LOW_COMMITMENT"  # Zero-obligation, exploratory visit


class VisitPitchEngine:
    """
    Selects tailored visit pitch approach based on customer context and signals.
    """

    STRATEGY_DESCRIPTIONS: Dict[PitchStrategy, str] = {
        PitchStrategy.EXPERIENCE: (
            "Explain that ceiling heights, natural ventilation, carpet area dimensions, and finishes "
            "are best experienced in person rather than through brochure diagrams."
        ),
        PitchStrategy.COMPARISON: (
            "If the client is evaluating two options or localities, suggest a combined tour "
            "to clearly see the difference in neighborhood, connectivity, and clubhouse scale."
        ),
        PitchStrategy.CONVENIENCE: (
            "Highlight our complimentary doorstep AC cab service: 'We arrange a private AC cab from your home/office "
            "directly to the site and back for you and your family, completely on us!'"
        ),
        PitchStrategy.LOW_COMMITMENT: (
            "Reassure the client that a site visit is purely exploratory with zero obligation to buy: "
            "'A quick 30-minute walkthrough gives you complete peace of mind with zero pressure.'"
        )
    }

    def select_strategy(
        self,
        customer_requirements: Dict[str, Any],
        open_objections: List[str],
        previous_pitches: int = 0,
        comparing_multiple: bool = False
    ) -> PitchStrategy:
        """
        Determines the optimal pitch angle based on customer state.
        """
        # 1. If comparing multiple properties
        if comparing_multiple:
            return PitchStrategy.COMPARISON

        # 2. If objection is NO_TIME or SPOUSE_NOT_CONVINCED -> Convenience
        if any(o in open_objections for o in ["NO_TIME", "SPOUSE_NOT_CONVINCED"]):
            return PitchStrategy.CONVENIENCE

        # 3. If objection is NEED_TO_THINK or NOT_READY -> Low Commitment
        if any(o in open_objections for o in ["NEED_TO_THINK", "NOT_READY", "PRICE"]):
            return PitchStrategy.LOW_COMMITMENT

        # 4. Cycle strategies across multiple turns so ARIS never sounds robotic
        if previous_pitches % 3 == 0:
            return PitchStrategy.EXPERIENCE
        elif previous_pitches % 3 == 1:
            return PitchStrategy.CONVENIENCE
        else:
            return PitchStrategy.LOW_COMMITMENT

    def get_pitch_directive(self, strategy: PitchStrategy) -> str:
        """Returns concise prompt instruction for the chosen pitch strategy."""
        desc = self.STRATEGY_DESCRIPTIONS.get(strategy, "")
        return f"SITE VISIT PITCH STRATEGY [{strategy.value}]: {desc}"

    @classmethod
    def craft_pitch(
        cls,
        strategy: PitchStrategy,
        property_name: str = "",
        locality: str = "",
        customer_name: str = ""
    ) -> str:
        name_prefix = f"Hi {customer_name}, " if customer_name else ""
        prop_info = f" at {property_name}" if property_name else ""
        loc_info = f" in {locality}" if locality else ""

        if strategy == PitchStrategy.EXPERIENCE:
            return (
                f"{name_prefix}Photos and brochures only tell half the story. Experiencing the natural light, ceiling height, "
                f"and open spaces{prop_info}{loc_info} in person gives true clarity. We would love to arrange an exclusive site tour "
                f"with our complimentary private chauffeur cab to pick you up and drop you back home comfortably."
            )
        elif strategy == PitchStrategy.COMPARISON:
            return (
                f"{name_prefix}Comparing layouts and neighborhoods is so much easier in person. We can coordinate an easy walkthrough "
                f"of{prop_info}{loc_info} along with comparable options, complete with our complimentary doorstep cab service."
            )
        elif strategy == PitchStrategy.CONVENIENCE:
            return (
                f"{name_prefix}We know visiting properties takes time from your busy week, so we make it effortless! We provide a complimentary "
                f"private doorstep AC cab pickup and drop for you and your family directly to{prop_info}{loc_info}."
            )
        else:
            return (
                f"{name_prefix}A quick 30-minute walkthrough{prop_info}{loc_info} is completely exploratory with zero pressure or commitment. "
                f"To make it effortless, our complimentary private cab will pick you up and drop you back."
            )
