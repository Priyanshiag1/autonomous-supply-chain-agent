"""
Autonomous Demand-Sensing: Root-Cause Intelligence Engine & Semantic Guardrails
-------------------------------------------------------------------------------
Implements:
1. Tier 1: Deterministic Calendar & SNAP Event Lookup
2. Step 4b: Category-Event Semantic Relevance Guardrail (Spurious Correlation Defense)
3. Tier 2: Statistical Pattern Inference (B2B Bulk Impulse vs. Promotional Plateau)
4. Tier 3: Executive Intelligence Synthesis (with Deterministic Fallback & Gemini LLM)
"""

import os
from typing import Dict, Any, Optional
import pandas as pd
from dotenv import load_dotenv

# Load local .env if present
load_dotenv()

class RootCauseIntelligenceEngine:
    def __init__(self, affinity_threshold: float = 0.50):
        self.affinity_threshold = affinity_threshold
        
        # Category-Event Semantic Affinity Matrix (0.0 to 1.0)
        # Defines realistic causal relationships between retail categories and events
        self.affinity_matrix: Dict[str, Dict[str, float]] = {
            "SNAP": {
                "FOODS": 1.00,      # SNAP welfare is legally restricted to food only
                "HOUSEHOLD": 0.00,  # Legally barred
                "HOBBIES": 0.00     # Legally barred
            },
            "SuperBowl": {
                "FOODS": 0.95,      # Game day party snacks, dips, beverages
                "HOUSEHOLD": 0.10,
                "HOBBIES": 0.20
            },
            "Thanksgiving": {
                "FOODS": 0.95,      # Holiday feast groceries, baking
                "HOUSEHOLD": 0.40,
                "HOBBIES": 0.15
            },
            "Christmas": {
                "HOBBIES": 0.95,    # Toys, games, gift crafts
                "FOODS": 0.90,      # Feast groceries
                "HOUSEHOLD": 0.30
            },
            "NewYear": {
                "FOODS": 0.90,      # Party snacks, cold beverages
                "HOUSEHOLD": 0.20,
                "HOBBIES": 0.25
            },
            "ValentinesDay": {
                "FOODS": 0.90,      # Chocolates, confectionery
                "HOBBIES": 0.85,    # Greeting cards, gift toys
                "HOUSEHOLD": 0.10
            },
            "StPatricksDay": {
                "FOODS": 0.85,      # Beverages, themed party foods
                "HOUSEHOLD": 0.15,
                "HOBBIES": 0.30
            },
            "Easter": {
                "FOODS": 0.90,      # Candy, eggs, baking
                "HOBBIES": 0.80,    # Easter baskets, craft items
                "HOUSEHOLD": 0.15
            },
            "MemorialDay": {
                "FOODS": 0.85,      # Outdoor BBQ, grilling meats, soda
                "HOUSEHOLD": 0.20,
                "HOBBIES": 0.35     # Outdoor recreation
            },
            "IndependenceDay": {
                "FOODS": 0.90,      # 4th of July grilling & party snacks
                "HOUSEHOLD": 0.20,
                "HOBBIES": 0.40
            },
            "LaborDay": {
                "FOODS": 0.85,      # Holiday weekend groceries
                "HOUSEHOLD": 0.20,
                "HOBBIES": 0.30
            },
            "Halloween": {
                "FOODS": 0.95,      # Candy, confectionery
                "HOBBIES": 0.85,    # Costumes, novelty toys
                "HOUSEHOLD": 0.20
            }
        }

    def evaluate_root_cause(
        self,
        anomaly_record: Dict[str, Any],
        calendar_row: pd.Series,
        cross_state_spiked: bool = False
    ) -> Dict[str, Any]:
        """
        Determines the root cause for an anomaly with semantic guardrails.
        """
        cat_id = anomaly_record['cat_id']
        state_id = anomaly_record['state_id']
        actual_sales = anomaly_record['actual_sales']
        baseline_mean = anomaly_record['baseline_mean']
        z_score = anomaly_record['z_score']
        status = anomaly_record['status']
        streak_day = anomaly_record.get('streak_day', 1)
        
        evt_name = str(calendar_row.get('event_name_1', ''))
        snap_flag = int(calendar_row.get(f"snap_{state_id}", 0))
        
        has_calendar_event = (evt_name != '' and evt_name != 'nan' and evt_name != 'None')
        has_snap = (snap_flag == 1)
        
        tier_level = "TIER_2_STATISTICAL"
        classification = "UNMARKED_ANOMALY"
        primary_reason = ""
        guardrail_audit = "None"
        guardrail_passed = True
        
        # --- ZERO DEMAND GUARD ---
        if actual_sales == 0:
            tier_level = "TIER_2_STATISTICAL"
            classification = "ZERO_DEMAND_INACTIVE"
            primary_reason = f"Zero POS sales recorded today against baseline of {baseline_mean:.1f} units."
            guardrail_passed = True
            guardrail_audit = "None"

        # --- TIER 1 & STEP 4B: CALENDAR LOOKUP + SEMANTIC GUARDRAIL ---
        elif has_calendar_event:
            # Check affinity of this calendar event with product category
            event_affinities = self.affinity_matrix.get(evt_name, {"FOODS": 0.5, "HOUSEHOLD": 0.5, "HOBBIES": 0.5})
            affinity = event_affinities.get(cat_id, 0.3)
            
            if affinity >= self.affinity_threshold:
                tier_level = "TIER_1_CALENDAR"
                classification = f"CALENDAR_EVENT_{evt_name.upper()}"
                primary_reason = f"Surge driven by {evt_name} holiday purchasing in {cat_id} (Category Affinity: {affinity:.2f})."
                guardrail_passed = True
            else:
                guardrail_passed = False
                guardrail_audit = f"Calendar Event '{evt_name}' rejected: Coincidental holiday incompatible with category '{cat_id}' (Affinity: {affinity:.2f} < {self.affinity_threshold})."
                
        # If calendar event was missing or rejected, check SNAP welfare (if FOODS)
        if actual_sales > 0 and (not guardrail_passed or (not has_calendar_event and has_snap)):
            snap_affinity = self.affinity_matrix["SNAP"].get(cat_id, 0.0)
            if has_snap and snap_affinity >= self.affinity_threshold:
                tier_level = "TIER_1_CALENDAR"
                classification = "SNAP_WELFARE_DISBURSEMENT"
                if actual_sales >= 300: # Mega-spike indicator
                    primary_reason = f"{state_id} monthly SNAP welfare disbursement cycle; likely compounded by institutional wholesale reorder on the same date."
                else:
                    primary_reason = f"{state_id} state SNAP welfare food-stamp disbursement cycle driving high category footfall."
                guardrail_passed = True
            elif has_snap and snap_affinity < self.affinity_threshold:
                guardrail_audit = f"SNAP Welfare Flag rejected: Welfare disbursement legally restricted to food; incompatible with '{cat_id}'."
                guardrail_passed = False

        # --- TIER 2: STATISTICAL HEURISTIC INFERENCE (WHEN CALENDAR EMPTY OR REJECTED) ---
        if actual_sales > 0 and (tier_level == "TIER_2_STATISTICAL" or not guardrail_passed):
            tier_level = "TIER_2_STATISTICAL"
            
            # Duration & Velocity Fingerprint
            if status == "PROVISIONAL_ALERT" or streak_day == 1:
                classification = "B2B_WHOLESALE_IMPULSE"
                shape_reason = "High-velocity single-day impulse shock; characteristic of institutional B2B distributor restocking."
            elif "PLATEAU" in status or streak_day >= 3:
                classification = "PROMOTIONAL_OR_VIRAL_PLATEAU"
                shape_reason = "Multi-day sustained elevation; characteristic of unannounced store promotional campaign or local viral adoption."
            else:
                classification = "MULTI_DAY_ELEVATED_SURGE"
                shape_reason = "Multi-day elevated demand velocity; active demand wave under surveillance."
                
            # Cross-State Scope
            if cross_state_spiked:
                geo_reason = "National multi-region velocity shift observed across companion states."
            else:
                geo_reason = f"Hyperlocal regional shock isolated exclusively to {state_id} distribution zone."
                
            primary_reason = f"{shape_reason} ({geo_reason})"

        # --- TIER 3: EXECUTIVE BRIEFING SYNTHESIS (GEMINI OR DETERMINISTIC) ---
        executive_summary = self._synthesize_executive_summary(
            sku_id=anomaly_record.get('sku_id', 'UNKNOWN_SKU'),
            cat_id=cat_id,
            state_id=state_id,
            date_str=calendar_row.get('date', 'Unknown Date'),
            actual_sales=actual_sales,
            baseline_mean=baseline_mean,
            z_score=z_score,
            primary_reason=primary_reason,
            guardrail_audit=guardrail_audit,
            classification=classification
        )

        return {
            'tier_level': tier_level,
            'classification': classification,
            'primary_reason': primary_reason,
            'guardrail_passed': guardrail_passed,
            'guardrail_audit': guardrail_audit,
            'executive_summary': executive_summary
        }

    def _synthesize_executive_summary(
        self,
        sku_id: str,
        cat_id: str,
        state_id: str,
        date_str: str,
        actual_sales: float,
        baseline_mean: float,
        z_score: float,
        primary_reason: str,
        guardrail_audit: str,
        classification: str
    ) -> str:
        """
        Generates a 2-sentence executive summary.
        Uses Gemini LLM if GEMINI_API_KEY is configured in .env; otherwise falls back to deterministic rule generator.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        
        # If Gemini API key is present and non-empty, attempt LLM generation
        if api_key and api_key.strip() != "":
            try:
                from google import genai
                client = genai.Client(api_key=api_key)
                prompt = (
                    f"You are Agent 1 (Senior Demand Sensing Intelligence AI) in an enterprise supply-chain system.\n"
                    f"Synthesize a strict 2-sentence executive intelligence briefing for a detected retail demand shock:\n"
                    f"• Date: {date_str} | State: {state_id} | SKU: {sku_id} (Category: {cat_id})\n"
                    f"• Demand: {actual_sales:.0f} units vs Expected Baseline: {baseline_mean:.1f} units (Z-Score: {z_score:+.2f})\n"
                    f"• Causal Diagnosis: {primary_reason}\n"
                    f"• Guardrail Check: {guardrail_audit}\n"
                    f"Requirements: Write exactly 2 crisp sentences. Mention the exact numbers and operational root cause."
                )
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt
                )
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                # Fallback on network or API issue
                pass

        # Deterministic Executive Synthesis (Instant, 100% Reliable Fallback)
        if actual_sales == 0:
            return f"Passive surveillance active for {sku_id} in {state_id} ({date_str}): Zero POS demand recorded today against baseline of {baseline_mean:.1f} units. No demand anomaly detected."
            
        pct_change = ((actual_sales - baseline_mean) / max(baseline_mean, 1.0)) * 100
        sign = "+" if pct_change >= 0 else ""
        
        sentence1 = (
            f"Demand shock detected for {sku_id} in {state_id} ({date_str}), registering {actual_sales:.0f} units "
            f"({sign}{pct_change:.1f}% vs baseline {baseline_mean:.1f}, Z-score: {z_score:+.2f})."
        )
        sentence2 = f"Root-cause diagnosis attributes this surge to: {primary_reason}"
        
        if guardrail_audit != "None":
            sentence2 += f" [Guardrail Note: {guardrail_audit}]"
            
        return f"{sentence1} {sentence2}"
