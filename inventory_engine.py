"""
Autonomous Demand-Sensing & Inventory Rebalancing Prototype
Module: inventory_engine.py
-----------------------------------------------------------
Implements Step 6:
1. Active SQLite Inventory Ledger (inventory.db)
2. Physical Warehouse Accounting (Min-Fulfillment, Zero-Clamping, Unmet Demand)
3. Dual-Runway Safe-Side Bracket (Impulse Runway vs Sustained Runway)
4. Multi-Echelon Inter-Warehouse Transfer & Emergency Expedited Factory Orders
"""

import sqlite3
import os
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd

class WarehouseInventoryEngine:
    def __init__(self, db_path: str = "inventory.db"):
        self.db_path = db_path
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        """Returns SQLite connection with row dictionary access."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self, reset: bool = False):
        """Initializes production-grade SQLite schema."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if reset:
                cursor.execute("DROP TABLE IF EXISTS inventory_daily_ledger")
                cursor.execute("DROP TABLE IF EXISTS inbound_shipments")
                cursor.execute("DROP TABLE IF EXISTS warehouse_inventory")

            # Table 1: Primary Warehouse Inventory Balance
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS warehouse_inventory (
                sku_id TEXT,
                state_id TEXT,
                current_stock INTEGER CHECK(current_stock >= 0),
                safety_stock INTEGER,
                lead_time_days INTEGER,
                reorder_batch_size INTEGER,
                last_updated_day INTEGER DEFAULT 0,
                PRIMARY KEY (sku_id, state_id)
            );
            """)

            # Table 2: Inbound Purchase Orders & Inter-Warehouse Transfers
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS inbound_shipments (
                shipment_id TEXT PRIMARY KEY,
                sku_id TEXT,
                state_id TEXT,
                order_day INTEGER,
                arrival_day INTEGER,
                quantity INTEGER,
                shipment_type TEXT, -- 'EMERGENCY_FACTORY_PO', 'INTER_WAREHOUSE_TRANSFER', 'STANDARD_PO'
                status TEXT,        -- 'IN_TRANSIT', 'DELIVERED'
                source_location TEXT -- 'FACTORY', 'CA_WAREHOUSE', 'TX_WAREHOUSE', etc.
            );
            """)

            # Table 3: Daily Physical Inventory Audit Ledger
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS inventory_daily_ledger (
                day_index INTEGER,
                sku_id TEXT,
                state_id TEXT,
                opening_stock INTEGER,
                inbound_received INTEGER,
                actual_demand INTEGER,
                fulfilled_demand INTEGER,
                unmet_demand INTEGER,
                closing_stock INTEGER,
                stockout_occurred INTEGER, -- 1 for True, 0 for False
                runway_impulse_days REAL,
                runway_sustained_days REAL,
                action_taken TEXT,
                PRIMARY KEY (day_index, sku_id, state_id)
            );
            """)
            conn.commit()

    def seed_initial_inventory(self, m5_df: pd.DataFrame, hero_skus: List[str]):
        """
        Seeds initial stock balances for all Hero SKUs across all companion states (CA, TX, WI).
        Uses real historical sales metrics to calibrate initial stock and safety buffers:
        - Opening Stock: ~10-14 days of baseline sales
        - Safety Stock: ~3-5 days of baseline sales
        - Lead Time: 3 days (Factory), 1 day (Inter-warehouse transfer)
        - Reorder Batch: ~14-21 days of baseline sales
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Extract unique item_ids from hero SKUs (e.g. FOODS_3_090, FOODS_2_285)
            unique_item_prefixes = set()
            for sku in hero_skus:
                parts = sku.replace("_validation", "").split("_")
                item_prefix = "_".join(parts[:3])
                unique_item_prefixes.add(item_prefix)
            
            for item_prefix in unique_item_prefixes:
                # Find all state variants (CA, TX, WI) for this item
                matching_rows = m5_df[m5_df['item_id'] == item_prefix]
                    
                for _, row in matching_rows.iterrows():
                    sku_id = row['id']
                    state_id = row['state_id']
                    
                    # Calculate baseline mean from first 30 days
                    d_cols = [f"d_{i}" for i in range(1, 31)]
                    first_month_sales = [float(row[c]) for c in d_cols]
                    avg_daily_demand = max(float(pd.Series(first_month_sales).mean()), 3.0)
                    
                    # Calibrate warehouse stock levels
                    # Ensure realistic inventory for high-variance testing
                    initial_stock = int(round(avg_daily_demand * 12))
                    safety_stock = int(round(avg_daily_demand * 3.5))
                    lead_time = 3
                    reorder_batch = int(round(avg_daily_demand * 14))
                    
                    cursor.execute("""
                    INSERT OR REPLACE INTO warehouse_inventory 
                    (sku_id, state_id, current_stock, safety_stock, lead_time_days, reorder_batch_size, last_updated_day)
                    VALUES (?, ?, ?, ?, ?, ?, 0)
                    """, (sku_id, state_id, initial_stock, safety_stock, lead_time, reorder_batch))
            
            conn.commit()

    def reset_scenario_baseline(self, sku_id: str, state_id: str, day_index: int, m5_df: Optional[pd.DataFrame] = None):
        """
        Idempotent simulation baseline reset for Digital Twin historical replay.
        Ensures scrubbing or reloading a day always begins from canonical pre-event state,
        preventing state drift or cumulative mutation across browser refreshes.
        """
        parts = sku_id.replace("_validation", "").split("_")
        prefix = "_".join(parts[:3])
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Clear existing active shipments and daily ledger records for this item prefix
            cursor.execute("DELETE FROM inbound_shipments WHERE sku_id LIKE ?", (f"{prefix}%",))
            cursor.execute("DELETE FROM inventory_daily_ledger WHERE sku_id LIKE ?", (f"{prefix}%",))
            
            # Scenario 1: SuperBowl Day 9 (FOODS_3_090_TX)
            if "FOODS_3_090" in sku_id:
                cursor.execute("""
                    INSERT OR REPLACE INTO warehouse_inventory
                    (sku_id, state_id, current_stock, safety_stock, lead_time_days, reorder_batch_size, last_updated_day)
                    VALUES (?, 'TX', 349, 102, 3, 407, ?)
                """, (sku_id, day_index - 1))
                cursor.execute("""
                    INSERT OR REPLACE INTO warehouse_inventory
                    (sku_id, state_id, current_stock, safety_stock, lead_time_days, reorder_batch_size, last_updated_day)
                    VALUES (?, 'CA', 350, 100, 1, 407, ?)
                """, (sku_id.replace("_TX_", "_CA_"), day_index - 1))
                cursor.execute("""
                    INSERT OR REPLACE INTO warehouse_inventory
                    (sku_id, state_id, current_stock, safety_stock, lead_time_days, reorder_batch_size, last_updated_day)
                    VALUES (?, 'WI', 200, 80, 1, 407, ?)
                """, (sku_id.replace("_TX_", "_WI_"), day_index - 1))

            # Scenario 2: 86x Mega SNAP Outlier (FOODS_2_285_TX, Day 98)
            elif "FOODS_2_285" in sku_id:
                cursor.execute("""
                    INSERT OR REPLACE INTO warehouse_inventory
                    (sku_id, state_id, current_stock, safety_stock, lead_time_days, reorder_batch_size, last_updated_day)
                    VALUES (?, 'TX', 93, 27, 3, 108, ?)
                """, (sku_id, day_index - 1))
                cursor.execute("""
                    INSERT OR REPLACE INTO warehouse_inventory
                    (sku_id, state_id, current_stock, safety_stock, lead_time_days, reorder_batch_size, last_updated_day)
                    VALUES (?, 'CA', 300, 40, 1, 108, ?)
                """, (sku_id.replace("_TX_", "_CA_"), day_index - 1))
                cursor.execute("""
                    INSERT OR REPLACE INTO warehouse_inventory
                    (sku_id, state_id, current_stock, safety_stock, lead_time_days, reorder_batch_size, last_updated_day)
                    VALUES (?, 'WI', 200, 40, 1, 108, ?)
                """, (sku_id.replace("_TX_", "_WI_"), day_index - 1))

            # Scenario 3: Toys & Crafts SNAP Defense (HOBBIES_1_209_TX, Day 126)
            elif "HOBBIES_1_209" in sku_id:
                cursor.execute("""
                    INSERT OR REPLACE INTO warehouse_inventory
                    (sku_id, state_id, current_stock, safety_stock, lead_time_days, reorder_batch_size, last_updated_day)
                    VALUES (?, 'TX', 36, 10, 3, 42, ?)
                """, (sku_id, day_index - 1))
                cursor.execute("""
                    INSERT OR REPLACE INTO warehouse_inventory
                    (sku_id, state_id, current_stock, safety_stock, lead_time_days, reorder_batch_size, last_updated_day)
                    VALUES (?, 'CA', 69, 20, 1, 42, ?)
                """, (sku_id.replace("_TX_", "_CA_"), day_index - 1))
                cursor.execute("""
                    INSERT OR REPLACE INTO warehouse_inventory
                    (sku_id, state_id, current_stock, safety_stock, lead_time_days, reorder_batch_size, last_updated_day)
                    VALUES (?, 'WI', 50, 20, 1, 42, ?)
                """, (sku_id.replace("_TX_", "_WI_"), day_index - 1))

            # Scenario 4 & 5: Cleaners Single-Day Impulse & Multi-Day Plateau (HOUSEHOLD_2_440_TX)
            elif "HOUSEHOLD_2_440" in sku_id:
                cursor.execute("""
                    INSERT OR REPLACE INTO warehouse_inventory
                    (sku_id, state_id, current_stock, safety_stock, lead_time_days, reorder_batch_size, last_updated_day)
                    VALUES (?, 'TX', 36, 10, 3, 42, 339)
                """, (sku_id,))
                cursor.execute("""
                    INSERT OR REPLACE INTO warehouse_inventory
                    (sku_id, state_id, current_stock, safety_stock, lead_time_days, reorder_batch_size, last_updated_day)
                    VALUES (?, 'CA', 36, 10, 1, 42, 339)
                """, (sku_id.replace("_TX_", "_CA_"),))
                cursor.execute("""
                    INSERT OR REPLACE INTO warehouse_inventory
                    (sku_id, state_id, current_stock, safety_stock, lead_time_days, reorder_batch_size, last_updated_day)
                    VALUES (?, 'WI', 36, 10, 1, 42, 339)
                """, (sku_id.replace("_TX_", "_WI_"),))

            # Generic Custom SKU Explorer (Any of the 9,147 SKUs in Walmart M5)
            else:
                base_demand = 10.0
                if m5_df is not None and not m5_df.empty:
                    m_row = m5_df[m5_df['id'] == sku_id]
                    if not m_row.empty:
                        d_cols = [f"d_{i}" for i in range(1, min(31, max(2, day_index)))]
                        if d_cols:
                            base_demand = max(3.0, float(pd.Series([float(m_row.iloc[0][c]) for c in d_cols]).mean()))
                
                stock = int(round(base_demand * 12))
                ss = int(round(base_demand * 3.5))
                batch = int(round(base_demand * 14))
                
                parts = sku_id.replace("_validation", "").split("_")
                prefix = "_".join(parts[:3])
                store_num = parts[4] if len(parts) > 4 else "1"
                
                for st in ['TX', 'CA', 'WI']:
                    target_sku = f"{prefix}_{st}_{store_num}_validation"
                    cursor.execute("""
                        INSERT OR REPLACE INTO warehouse_inventory
                        (sku_id, state_id, current_stock, safety_stock, lead_time_days, reorder_batch_size, last_updated_day)
                        VALUES (?, ?, ?, ?, 3, ?, ?)
                    """, (target_sku, st, stock, ss, batch, day_index - 1))
            
            conn.commit()

        # If evaluating a day during the plateau progression (340 to 343), replay prior days
        if 340 <= day_index <= 343 and "HOUSEHOLD_2_440" in sku_id:
            plateau_sequence = [
                (340, 24, "PROVISIONAL_ALERT", 2.8, 4.2),
                (341, 42, "ELEVATED_SURGE_DAY_2", 2.8, 7.8),
                (342, 31, "CONFIRMED_STRUCTURAL_PLATEAU", 2.8, 5.6)
            ]
            for d_i, dem_i, st_i, b_i, z_i in plateau_sequence:
                if d_i < day_index:
                    self.process_daily_demand(
                        day_index=d_i,
                        sku_id=sku_id,
                        state_id=state_id,
                        actual_demand=dem_i,
                        anomaly_status=st_i,
                        baseline_mean=b_i,
                        z_score=z_i
                    )

    def get_stock_record(self, sku_id: str, state_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves active warehouse stock record."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM warehouse_inventory WHERE sku_id = ? AND state_id = ?",
                (sku_id, state_id)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def process_daily_demand(
        self,
        day_index: int,
        sku_id: str,
        state_id: str,
        actual_demand: int,
        anomaly_status: str,
        baseline_mean: float,
        z_score: float
    ) -> Dict[str, Any]:
        """
        Executes physical inventory accounting and dual-runway evaluation for Day t:
        1. Inbound receipt of arriving purchase orders.
        2. Fulfillment: Fulfilled = min(Stock_effective, Demand).
        3. Zero-clamping: Current_Stock = max(0, Stock_effective - Demand).
        4. Unmet orders: Unmet = max(0, Demand - Stock_effective).
        5. Stockout determination: Stockout_Occurred = (Unmet > 0).
        6. Dual-Runway calculation on provisional days.
        7. Automated replenishment / cross-warehouse dispatch when runway <= lead time.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Fetch current warehouse record
            cursor.execute(
                "SELECT * FROM warehouse_inventory WHERE sku_id = ? AND state_id = ?",
                (sku_id, state_id)
            )
            wh_row = cursor.fetchone()
            if not wh_row:
                raise ValueError(f"SKU {sku_id} in {state_id} not found in warehouse_inventory.")
            
            opening_stock = wh_row['current_stock']
            safety_stock = wh_row['safety_stock']
            lead_time = wh_row['lead_time_days']
            reorder_batch = wh_row['reorder_batch_size']
            
            # 2. Check and receive inbound shipments arriving today
            cursor.execute("""
                SELECT shipment_id, quantity FROM inbound_shipments
                WHERE sku_id = ? AND state_id = ? AND arrival_day = ? AND status = 'IN_TRANSIT'
            """, (sku_id, state_id, day_index))
            arrived_shipments = cursor.fetchall()
            
            inbound_received = sum(s['quantity'] for s in arrived_shipments)
            
            # Update arrived shipments status to DELIVERED
            for s in arrived_shipments:
                cursor.execute(
                    "UPDATE inbound_shipments SET status = 'DELIVERED' WHERE shipment_id = ?",
                    (s['shipment_id'],)
                )
            
            # 3. Physical Inventory Accounting
            effective_opening = opening_stock + inbound_received
            fulfilled_demand = min(effective_opening, actual_demand)
            closing_stock = max(0, effective_opening - actual_demand)
            unmet_demand = max(0, actual_demand - effective_opening)
            stockout_occurred = (unmet_demand > 0)
            
            # 4. Dual-Runway Safe-Side Calculation
            effective_base = max(baseline_mean, 0.5)
            effective_spike = max(float(actual_demand), 0.5)
            
            # 4. Dual-Runway Safe-Side Calculation
            effective_base = max(baseline_mean, 0.5)
            effective_spike = max(float(actual_demand), 0.5)
            
            runway_impulse = closing_stock / effective_base
            runway_sustained = closing_stock / effective_spike
            
            # 5. Pipeline Stock Tracking (Orders currently in-transit arriving in future)
            cursor.execute("""
                SELECT COALESCE(SUM(quantity), 0) FROM inbound_shipments
                WHERE sku_id = ? AND state_id = ? AND status = 'IN_TRANSIT' AND arrival_day > ?
            """, (sku_id, state_id, day_index))
            pipeline_qty = cursor.fetchone()[0]
            
            # 6. Autonomous Inventory Operator Decision (Agent 2 Logic)
            actions = []
            alert_tier = "HEALTHY"
            
            if stockout_occurred:
                alert_tier = "CRITICAL_STOCKOUT_INCIDENT"
                actions.append(f"CRITICAL: Stockout of {unmet_demand} units! Warehouse inventory clamped to 0.")
                
                # Check for Inter-Warehouse Transfer from companion states (Fast 1-day transit)
                transfer_qty = self._find_and_dispatch_transfer(
                    conn, sku_id, requesting_state=state_id, needed_qty=unmet_demand + safety_stock, current_day=day_index
                )
                if transfer_qty > 0:
                    actions.append(f"Dispatched expedited inter-warehouse transfer of {transfer_qty} units (Arrival: Day {day_index+1}).")
                
                # Net Requirements Planning (Accounting for Pipeline Stock & Inbound Transfer)
                gross_deficit = unmet_demand + safety_stock
                effective_incoming = pipeline_qty + transfer_qty
                net_factory_deficit = max(0, gross_deficit - effective_incoming)
                
                if net_factory_deficit > 0:
                    factory_order_qty = max(reorder_batch, net_factory_deficit)
                    self._place_factory_po(conn, sku_id, state_id, day_index, factory_order_qty, lead_time, is_emergency=True)
                    actions.append(f"Placed Emergency Factory PO for {factory_order_qty} units (Expedited Lead Time: {lead_time} days; Netted incoming {effective_incoming}).")
                else:
                    actions.append(f"Factory PO suppressed: {effective_incoming} units already in pipeline (Anti-bullwhip protection).")
                
            elif anomaly_status == "PROVISIONAL_ALERT":
                alert_tier = "PROVISIONAL_SURGE_RISK"
                if runway_sustained <= lead_time:
                    if pipeline_qty >= reorder_batch:
                        actions.append(
                            f"Sustained runway ({runway_sustained:.1f}d) <= lead time ({lead_time}d), but Standby PO suppressed: {pipeline_qty} units already in pipeline."
                        )
                    else:
                        actions.append(
                            f"WARNING: Sustained runway ({runway_sustained:.1f}d) <= lead time ({lead_time}d). "
                            f"Placed standby factory PO of {reorder_batch} units."
                        )
                        self._place_factory_po(conn, sku_id, state_id, day_index, reorder_batch, lead_time, is_emergency=False)
                else:
                    actions.append(
                        f"SAFE BUFFER: Sustained runway is {runway_sustained:.1f} days vs {lead_time}d lead time. Standing by."
                    )
            elif "PLATEAU" in anomaly_status or "SURGE" in anomaly_status:
                is_below_buffer = (closing_stock <= safety_stock)
                alert_tier = "RECOVERED_BELOW_SAFETY_BUFFER" if is_below_buffer else "ELEVATED_DEMAND_ESCALATION"
                target_cover = int(actual_demand * lead_time) + safety_stock
                effective_stock_pos = closing_stock + pipeline_qty
                net_deficit = max(0, target_cover - effective_stock_pos)
                if net_deficit > 0:
                    order_qty = max(reorder_batch, net_deficit)
                    self._place_factory_po(conn, sku_id, state_id, day_index, order_qty, lead_time, is_emergency=True)
                    if is_below_buffer:
                        actions.append(f"FRAGILE RECOVERY: Demand fulfilled (Unmet=0), but closing stock ({closing_stock}) <= Safety Stock ({safety_stock}). Committed plateau replenishment order of {order_qty} units.")
                    else:
                        actions.append(f"PLATEAU REPLENISHMENT: Committed factory replenishment order of {order_qty} units.")
                else:
                    actions.append(f"Demand elevation monitored. Stock position ({effective_stock_pos} units on-hand/pipeline) covers {target_cover} target.")
            elif closing_stock <= safety_stock:
                alert_tier = "ROUTINE_REORDER_TRIGGERED"
                if pipeline_qty < reorder_batch:
                    self._place_factory_po(conn, sku_id, state_id, day_index, reorder_batch, lead_time, is_emergency=False)
                    actions.append(f"Routine reorder triggered: Stock {closing_stock} <= Safety Stock {safety_stock}.")
                else:
                    actions.append(f"Stock {closing_stock} <= Safety Stock, but {pipeline_qty} units already in pipeline. Standing by.")
            else:
                actions.append("Inventory nominal. Demand fulfilled within standard safety buffer.")
                
            action_summary = " | ".join(actions)
            
            # 6. Update database records
            cursor.execute("""
                UPDATE warehouse_inventory
                SET current_stock = ?, last_updated_day = ?
                WHERE sku_id = ? AND state_id = ?
            """, (closing_stock, day_index, sku_id, state_id))
            
            cursor.execute("""
                INSERT OR REPLACE INTO inventory_daily_ledger
                (day_index, sku_id, state_id, opening_stock, inbound_received, actual_demand, 
                 fulfilled_demand, unmet_demand, closing_stock, stockout_occurred, 
                 runway_impulse_days, runway_sustained_days, action_taken)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                day_index, sku_id, state_id, opening_stock, inbound_received, actual_demand,
                fulfilled_demand, unmet_demand, closing_stock, 1 if stockout_occurred else 0,
                round(runway_impulse, 2), round(runway_sustained, 2), action_summary
            ))
            
            conn.commit()
            
            return {
                "day_index": day_index,
                "sku_id": sku_id,
                "state_id": state_id,
                "opening_stock": opening_stock,
                "inbound_received": inbound_received,
                "actual_demand": actual_demand,
                "fulfilled_demand": fulfilled_demand,
                "unmet_demand": unmet_demand,
                "closing_stock": closing_stock,
                "stockout_occurred": stockout_occurred,
                "alert_tier": alert_tier,
                "runway_impulse_days": round(runway_impulse, 2),
                "runway_sustained_days": round(runway_sustained, 2),
                "action_taken": action_summary
            }

    def _find_and_dispatch_transfer(
        self,
        conn: sqlite3.Connection,
        sku_id: str,
        requesting_state: str,
        needed_qty: int,
        current_day: int
    ) -> int:
        """
        Scans companion warehouses in other states for surplus stock above 2x safety buffer.
        If surplus is found, dispatches an expedited 1-day inter-warehouse transfer.
        """
        cursor = conn.cursor()
        
        # Look for sibling SKUs in different states
        # e.g. If sku_id is FOODS_2_285_TX_1_validation, find FOODS_2_285_CA_1_validation or WI
        item_prefix = sku_id.split("_")[0] + "_" + sku_id.split("_")[1] + "_" + sku_id.split("_")[2]
        
        cursor.execute("""
            SELECT sku_id, state_id, current_stock, safety_stock 
            FROM warehouse_inventory
            WHERE sku_id LIKE ? AND state_id != ?
            ORDER BY (current_stock - safety_stock) DESC
        """, (f"{item_prefix}%", requesting_state))
        
        companion_whs = cursor.fetchall()
        for wh in companion_whs:
            surplus = wh['current_stock'] - (wh['safety_stock'] * 2)
            if surplus > 10:
                transfer_qty = min(needed_qty, surplus)
                donor_sku = wh['sku_id']
                donor_state = wh['state_id']
                
                # Decrement donor warehouse
                new_donor_stock = wh['current_stock'] - transfer_qty
                cursor.execute("""
                    UPDATE warehouse_inventory 
                    SET current_stock = ?
                    WHERE sku_id = ? AND state_id = ?
                """, (new_donor_stock, donor_sku, donor_state))
                
                # Create inbound shipment for requesting warehouse (Arrives next day = current_day + 1)
                shipment_id = f"TRANSFER_{donor_state}_TO_{requesting_state}_D{current_day}_{sku_id[:12]}"
                cursor.execute("""
                    INSERT OR REPLACE INTO inbound_shipments
                    (shipment_id, sku_id, state_id, order_day, arrival_day, quantity, shipment_type, status, source_location)
                    VALUES (?, ?, ?, ?, ?, ?, 'INTER_WAREHOUSE_TRANSFER', 'IN_TRANSIT', ?)
                """, (shipment_id, sku_id, requesting_state, current_day, current_day + 1, transfer_qty, f"{donor_state}_WAREHOUSE"))
                
                return transfer_qty
                
        return 0

    def _place_factory_po(
        self,
        conn: sqlite3.Connection,
        sku_id: str,
        state_id: str,
        current_day: int,
        quantity: int,
        lead_time_days: int,
        is_emergency: bool = False
    ):
        """Places a factory purchase order, arriving on current_day + lead_time_days."""
        cursor = conn.cursor()
        order_type = "EMERGENCY_FACTORY_PO" if is_emergency else "STANDARD_PO"
        shipment_id = f"PO_{order_type[:4]}_D{current_day}_{sku_id[:12]}_{state_id}"
        
        # Check if an order was already placed today for this SKU to prevent duplicate spam
        cursor.execute("""
            SELECT shipment_id FROM inbound_shipments
            WHERE sku_id = ? AND state_id = ? AND order_day = ?
        """, (sku_id, state_id, current_day))
        existing = cursor.fetchone()
        if not existing:
            arrival_day = current_day + lead_time_days
            cursor.execute("""
                INSERT INTO inbound_shipments
                (shipment_id, sku_id, state_id, order_day, arrival_day, quantity, shipment_type, status, source_location)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'IN_TRANSIT', 'FACTORY')
            """, (shipment_id, sku_id, state_id, current_day, arrival_day, quantity, order_type))

    def get_ledger_history(self, sku_id: str, state_id: str) -> pd.DataFrame:
        """Fetches complete daily audit history for a given SKU and state."""
        with self.get_connection() as conn:
            query = """
                SELECT * FROM inventory_daily_ledger
                WHERE sku_id = ? AND state_id = ?
                ORDER BY day_index ASC
            """
            return pd.read_sql_query(query, conn, params=(sku_id, state_id))
