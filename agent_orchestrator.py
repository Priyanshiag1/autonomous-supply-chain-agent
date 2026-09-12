"""
Autonomous Demand-Sensing & Inventory Rebalancing Prototype
Module: agent_orchestrator.py
-----------------------------------------------------------
Implements Step 7:
1. Agent 1: Demand Detective (Sensing, Anomaly Detection & Causal Diagnosis)
2. Agent 2: Inventory Operator (Physical Accounting, Dual-Runway, Transfers, MRP Netting)
3. Inter-Agent Communication Message Protocol (Structured JSON Handshake & Live Dialogue)
"""

import json
from typing import Dict, Any, List, Optional
import pandas as pd

from anomaly_engine import AnomalyDetectionEngine
from reason_engine import RootCauseIntelligenceEngine
from inventory_engine import WarehouseInventoryEngine

class DemandDetectiveAgent:
    """
    Agent 1: Senior Demand Sensing Intelligence AI
    Monitors daily Point-of-Sale time series data, flags statistical anomalies,
    runs semantic guardrails against calendar/policy events, and synthesizes root-cause hypotheses.
    """
    def __init__(self, warmup_days: int = 7, affinity_threshold: float = 0.50):
        self.anomaly_engine = AnomalyDetectionEngine(warmup_days=warmup_days)
        self.reason_engine = RootCauseIntelligenceEngine(affinity_threshold=affinity_threshold)

    def analyze_day(
        self,
        day_index: int,
        sku_id: str,
        state_id: str,
        cat_id: str,
        historical_sales: List[float],
        calendar_df: pd.DataFrame,
        cross_state_spikes: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluates Day t demand. If an anomaly is detected, synthesizes a structured JSON payload.
        """
        day_tag = f"d_{day_index}"
        
        # 1. Run Anomaly Detection Engine
        res_df = self.anomaly_engine.analyze_series(
            historical_sales[:day_index],
            calendar_df,
            sku_id,
            state_id,
            cat_id
        )
        
        day_rows = res_df[res_df['day'] == day_tag]
        if day_rows.empty:
            return None
            
        anom_row = day_rows.iloc[0]
        status = anom_row['status']
        
        actual_sales = float(anom_row['actual_sales'])
        z_score = float(anom_row['z_score'])
        baseline = float(anom_row['baseline_mean'])
        
        # Only dispatch active inter-agent alert payload if there is a real demand event:
        # 1. Significant positive surge: z_score >= 2.0 and actual_sales > 0
        # 2. Confirmed structural plateau: "PLATEAU" in status and actual_sales > 0
        # 3. Severe sudden drop from active baseline: z_score <= -2.5 and baseline >= 5.0 and actual_sales == 0
        is_surge = (z_score >= 2.0 and actual_sales > 0)
        is_plateau = ("PLATEAU" in status and actual_sales > 0)
        is_severe_drop = (z_score <= -2.5 and baseline >= 5.0 and actual_sales == 0)
        
        if not (is_surge or is_plateau or is_severe_drop):
            return None
            
        cal_rows = calendar_df[calendar_df['d'] == day_tag]
        cal_row = cal_rows.iloc[0] if not cal_rows.empty else pd.Series()
        
        # 2. Run Root Cause Intelligence Engine (with Semantic Guardrails)
        anom_dict = anom_row.to_dict()
        anom_dict['sku_id'] = sku_id
        anom_dict['cat_id'] = cat_id
        anom_dict['state_id'] = state_id
        diagnosis = self.reason_engine.evaluate_root_cause(anom_dict, cal_row, cross_state_spiked=cross_state_spikes)
        
        # 3. Construct Standardized Agent 1 JSON Payload
        payload = {
            "sender": "Agent 1 (Demand Detective)",
            "recipient": "Agent 2 (Inventory Operator)",
            "timestamp_day": day_tag,
            "date": str(cal_row.get('date', '2011-01-01')),
            "sku_id": sku_id,
            "category": cat_id,
            "state_id": state_id,
            "metrics": {
                "actual_demand": int(anom_row['actual_sales']),
                "baseline_mean": round(float(anom_row['baseline_mean']), 2),
                "z_score": round(float(anom_row['z_score']), 2),
                "status": status,
                "streak_day": int(anom_row.get('streak_day', 1))
            },
            "root_cause_diagnosis": {
                "tier_level": diagnosis['tier_level'],
                "classification": diagnosis['classification'],
                "guardrail_passed": diagnosis['guardrail_passed'],
                "guardrail_audit": diagnosis['guardrail_audit'],
                "primary_reason": diagnosis['primary_reason']
            },
            "executive_summary": diagnosis['executive_summary']
        }
        return payload


class InventoryOperatorAgent:
    """
    Agent 2: Inventory & Supply Chain Operator AI
    Listens for Agent 1 demand anomaly payloads, queries the live SQLite database,
    evaluates physical stock evolution, computes dual-runway brackets, and commits rebalancing actions.
    """
    def __init__(self, inventory_engine: WarehouseInventoryEngine):
        self.inv_engine = inventory_engine

    def handle_anomaly(self, agent1_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Receives Agent 1 payload, performs physical accounting in SQLite, and formulates response payload.
        """
        sku_id = agent1_payload['sku_id']
        state_id = agent1_payload['state_id']
        day_tag = agent1_payload['timestamp_day']
        day_index = int(day_tag.replace("d_", ""))
        
        metrics = agent1_payload['metrics']
        actual_demand = metrics['actual_demand']
        baseline_mean = metrics['baseline_mean']
        z_score = metrics['z_score']
        status = metrics['status']
        
        # 1. Process physical accounting in SQLite database
        inv_result = self.inv_engine.process_daily_demand(
            day_index=day_index,
            sku_id=sku_id,
            state_id=state_id,
            actual_demand=actual_demand,
            anomaly_status=status,
            baseline_mean=baseline_mean,
            z_score=z_score
        )
        
        # 2. Query in-transit pipeline stock for Texas
        with self.inv_engine.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COALESCE(SUM(quantity), 0) FROM inbound_shipments
                WHERE sku_id = ? AND state_id = ? AND status = 'IN_TRANSIT' AND arrival_day > ?
            """, (sku_id, state_id, day_index))
            pipeline_qty = cursor.fetchone()[0]
            
            # Fetch active shipments
            cursor.execute("""
                SELECT shipment_id, quantity, shipment_type, arrival_day, source_location
                FROM inbound_shipments
                WHERE sku_id = ? AND state_id = ? AND status = 'IN_TRANSIT' AND arrival_day > ?
            """, (sku_id, state_id, day_index))
            active_shipments = [dict(r) for r in cursor.fetchall()]

        # 3. Construct Conversational Dialogue Response
        dialogue_statement = self._generate_dialogue_statement(inv_result, pipeline_qty, status)
        
        # 4. Construct Standardized Agent 2 JSON Response Payload
        response_payload = {
            "sender": "Agent 2 (Inventory Operator)",
            "recipient": "Agent 1 (Demand Detective)",
            "timestamp_day": day_tag,
            "sku_id": sku_id,
            "state_id": state_id,
            "warehouse_accounting": {
                "opening_stock": inv_result['opening_stock'],
                "inbound_received_today": inv_result['inbound_received'],
                "actual_demand": inv_result['actual_demand'],
                "fulfilled_demand": inv_result['fulfilled_demand'],
                "closing_stock": inv_result['closing_stock'],
                "unmet_demand_backlog": inv_result['unmet_demand'],
                "stockout_occurred": inv_result['stockout_occurred']
            },
            "risk_and_runway": {
                "alert_tier": inv_result['alert_tier'],
                "runway_impulse_days": inv_result['runway_impulse_days'],
                "runway_sustained_days": inv_result['runway_sustained_days'],
                "pipeline_stock_in_transit": pipeline_qty
            },
            "active_inbound_pipeline": active_shipments,
            "decisions_and_actions": inv_result['action_taken'].split(" | "),
            "conversational_dialogue": dialogue_statement
        }
        return response_payload

    def _generate_dialogue_statement(self, inv_res: Dict[str, Any], pipeline_qty: int, status: str) -> str:
        """Generates conversational dialogue text for the live dashboard stream."""
        if inv_res['stockout_occurred']:
            return (
                f"ALERT CONFIRMED: Stockout incident active! Fulfilled {inv_res['fulfilled_demand']} units, "
                f"inventory clamped to 0 with {inv_res['unmet_demand']} unmet orders. "
                f"{inv_res['action_taken']}"
            )
        elif status == "PROVISIONAL_ALERT":
            return (
                f"PROVISIONAL ALERT ACKNOWLEDGED: Sustained runway is {inv_res['runway_sustained_days']:.1f} days "
                f"(Impulse: {inv_res['runway_impulse_days']:.1f} days). "
                f"Closing stock at {inv_res['closing_stock']} units. {inv_res['action_taken']}"
            )
        elif inv_res['alert_tier'] == "RECOVERED_BELOW_SAFETY_BUFFER":
            return (
                f"RECOVERY AUDIT: Stockout resolved today (Unmet=0). Closing stock at {inv_res['closing_stock']} units "
                f"(Below safety buffer). Committed structural replenishment."
            )
        else:
            return (
                f"AUDIT COMPLETE: Stock level nominal at {inv_res['closing_stock']} units. "
                f"Demand fully satisfied within standard safety parameters."
            )


class MultiAgentSystemOrchestrator:
    """
    Message Bus & Orchestration Controller:
    Dispatches time-series signals to Agent 1, routes anomaly payloads to Agent 2,
    and records the complete structured handshake history.
    """
    def __init__(self, inventory_engine: WarehouseInventoryEngine, warmup_days: int = 7):
        self.agent1 = DemandDetectiveAgent(warmup_days=warmup_days)
        self.agent2 = InventoryOperatorAgent(inventory_engine)
        self.conversation_history: List[Dict[str, Any]] = []

    def process_day(
        self,
        day_index: int,
        sku_id: str,
        state_id: str,
        cat_id: str,
        historical_sales: List[float],
        calendar_df: pd.DataFrame,
        cross_state_spikes: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Executes end-to-end multi-agent handshake for Day t.
        """
        # Step A: Agent 1 senses and evaluates
        a1_payload = self.agent1.analyze_day(
            day_index=day_index,
            sku_id=sku_id,
            state_id=state_id,
            cat_id=cat_id,
            historical_sales=historical_sales,
            calendar_df=calendar_df,
            cross_state_spikes=cross_state_spikes
        )
        
        if not a1_payload:
            return None
            
        # Step B: Route payload over message bus to Agent 2
        a2_payload = self.agent2.handle_anomaly(a1_payload)
        
        # Step C: Bundle handshake
        handshake_record = {
            "day": f"d_{day_index}",
            "sku_id": sku_id,
            "state_id": state_id,
            "agent_1_outbound": a1_payload,
            "agent_2_inbound_response": a2_payload
        }
        self.conversation_history.append(handshake_record)
        return handshake_record
