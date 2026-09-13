"""
IntelliMark AI — Autonomous Demand-Sensing & Inventory Rebalancing Engine
-------------------------------------------------------------------------
Interactive Enterprise Dashboard (Streamlit + Plotly)
Implements:
1. Autonomous Historical Replay Simulation over Walmart M5 Benchmark (365 Days)
2. Dual Navigation: Curated Hero Benchmarks (CEO Demo) + Full Catalog Explorer (9,147 SKUs)
3. Idempotent State Simulation: Zero data drift across reloads & scrubber scrubbing
4. Interactive Plotly Demand Curve with Rolling Baseline & Confidence Ribbon
5. Live Multi-Agent Conversational Dialogue Feed (Agent 1 <-> Agent 2 Handshake)
6. Compact Single-Screen Cockpit Layout with Collapsible Multi-Warehouse Drawer
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sqlite3
import os
from typing import Dict, Any, List

from anomaly_engine import AnomalyDetectionEngine
from reason_engine import RootCauseIntelligenceEngine
from inventory_engine import WarehouseInventoryEngine
from agent_orchestrator import MultiAgentSystemOrchestrator

# -----------------------------------------------------------------------------
# 1. Page Configuration & Professional Enterprise Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="IntelliMark AI • Autonomous Supply Chain Agent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise CSS for Single-Screen Cockpit
st.markdown("""
<style>
    /* Dark Enterprise Palette */
    .stApp {
        background-color: #0B0F19;
        color: #E2E8F0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Compact Metric Cards */
    .metric-card {
        background: #131B2E;
        border: 1px solid #1F2B48;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .metric-title {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        margin-bottom: 2px;
    }
    .metric-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .metric-subtext {
        font-size: 0.72rem;
        color: #64748B;
        margin-top: 2px;
    }

    /* Agent Communication Cards */
    .agent-card {
        background: #111827;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
        border-left: 4px solid #3B82F6;
    }
    .agent-1-header {
        color: #38BDF8;
        font-weight: 700;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        display: flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 6px;
    }
    .agent-2-header {
        color: #34D399;
        font-weight: 700;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        display: flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 6px;
    }
    .agent-body {
        font-size: 0.84rem;
        line-height: 1.45;
        color: #CBD5E1;
    }
    
    /* Badges */
    .badge-critical {
        background: rgba(239, 68, 68, 0.2);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-provisional {
        background: rgba(245, 158, 11, 0.2);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-healthy {
        background: rgba(16, 185, 129, 0.2);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    /* Header banner */
    .system-title {
        font-size: 1.3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #FFFFFF;
    }
    .system-subtitle {
        font-size: 0.8rem;
        color: #94A3B8;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Data Loading & Caching
# -----------------------------------------------------------------------------
@st.cache_data
def load_datasets():
    m5 = pd.read_csv("data_m5_daily_365.csv")
    cal = pd.read_csv("m5_repo/calendar.csv")
    hero = pd.read_csv("top_5_demo_skus.csv")
    return m5, cal, hero

m5_df, cal_df, hero_df = load_datasets()

# Initialize Warehouse Engine
def get_inventory_engine():
    db_path = "inventory.db"
    engine = WarehouseInventoryEngine(db_path=db_path)
    engine.init_db()
    return engine

inv_engine = get_inventory_engine()
orchestrator = MultiAgentSystemOrchestrator(inventory_engine=inv_engine, warmup_days=7)

@st.cache_data(show_spinner="Simulating 365-Day Continuous Digital Twin...")
def run_continuous_simulation(sku_id: str, state_id: str, cat_id: str):
    db_file = f"sim_{sku_id[:12]}_{state_id}.db"
    with sqlite3.connect(db_file) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        
    inv = WarehouseInventoryEngine(db_file)
    inv.init_db(reset=True)
    inv.seed_initial_inventory(m5_df, [sku_id])
    
    sku_row = m5_df[m5_df['id'] == sku_id].iloc[0]
    sales = [float(sku_row[f"d_{d}"]) for d in range(1, 366)]
    
    anom_engine = AnomalyDetectionEngine(warmup_days=7)
    anom_df = anom_engine.analyze_series(sales, cal_df, sku_id, state_id, cat_id)
    
    days_data = {}
    for d in range(1, 366):
        a_row = anom_df.iloc[d - 1]
        dem = int(a_row['actual_sales'])
        stat = a_row['status']
        base = float(a_row['baseline_mean']) if pd.notnull(a_row['baseline_mean']) else 0.0
        z = float(a_row['z_score']) if pd.notnull(a_row['z_score']) else 0.0
        
        inv_res = inv.process_daily_demand(d, sku_id, state_id, dem, stat, base, z)
        
        with inv.get_connection() as conn:
            c = conn.cursor()
            item_prefix = "_".join(sku_id.replace("_validation", "").split("_")[:3])
            c.execute("SELECT state_id, current_stock, safety_stock FROM warehouse_inventory WHERE sku_id LIKE ?", (f"{item_prefix}%",))
            comp_stocks = {r['state_id']: dict(r) for r in c.fetchall()}
            
            c.execute("SELECT shipment_id, quantity, shipment_type, order_day, arrival_day, source_location, status FROM inbound_shipments WHERE sku_id = ? AND arrival_day >= ? AND order_day <= ?", (sku_id, d, d))
            shipments = [dict(r) for r in c.fetchall()]
            
        days_data[d] = {
            'anom': a_row.to_dict(),
            'inv': inv_res,
            'comp': comp_stocks,
            'shipments': shipments
        }
        
    return days_data, anom_df

# -----------------------------------------------------------------------------
# 3. Sidebar Navigation: Strategic Archetypes vs Enterprise Catalog Explorer
# -----------------------------------------------------------------------------
st.sidebar.markdown("<div style='font-size:1.1rem; font-weight:800; color:#38BDF8;'>INTELLIMARK AI</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='font-size:0.75rem; color:#94A3B8; margin-bottom:12px;'>Autonomous Supply Chain Digital Twin</div>", unsafe_allow_html=True)

# Unified Session State Management
if "selected_sku_id" not in st.session_state:
    st.session_state.selected_sku_id = "FOODS_3_090_TX_1_validation"
if "current_day_num" not in st.session_state:
    st.session_state.current_day_num = 9
if "slider_sidebar" not in st.session_state:
    st.session_state.slider_sidebar = st.session_state.current_day_num
if "slider_main" not in st.session_state:
    st.session_state.slider_main = st.session_state.current_day_num

def on_sidebar_slider_change():
    st.session_state.current_day_num = st.session_state.slider_sidebar
    st.session_state.slider_main = st.session_state.slider_sidebar

def on_main_slider_change():
    st.session_state.current_day_num = st.session_state.slider_main
    st.session_state.slider_sidebar = st.session_state.slider_main

# 1. Pinned Strategic Archetypes (Quick Jump Shortcuts)
st.sidebar.markdown("<div style='font-size:0.82rem; font-weight:700; color:#CBD5E1; margin-bottom:6px;'>📌 Pinned Strategic Scenarios:</div>", unsafe_allow_html=True)

col_b1, col_b2 = st.sidebar.columns(2)
with col_b1:
    if st.button("🏈 Day 9 (SuperBowl)", use_container_width=True):
        st.session_state.selected_sku_id = "FOODS_3_090_TX_1_validation"
        st.session_state.current_day_num = 9
        st.session_state.slider_sidebar = 9
        st.session_state.slider_main = 9
        st.rerun()
    if st.button("🛡️ Day 126 (Guardrail)", use_container_width=True):
        st.session_state.selected_sku_id = "HOBBIES_1_209_TX_1_validation"
        st.session_state.current_day_num = 126
        st.session_state.slider_sidebar = 126
        st.session_state.slider_main = 126
        st.rerun()
    if st.button("📈 Day 343 (Plateau)", use_container_width=True):
        st.session_state.selected_sku_id = "HOUSEHOLD_2_440_TX_1_validation"
        st.session_state.current_day_num = 343
        st.session_state.slider_sidebar = 343
        st.session_state.slider_main = 343
        st.rerun()
with col_b2:
    if st.button("💥 Day 98 (Mega SNAP)", use_container_width=True):
        st.session_state.selected_sku_id = "FOODS_2_285_TX_1_validation"
        st.session_state.current_day_num = 98
        st.session_state.slider_sidebar = 98
        st.session_state.slider_main = 98
        st.rerun()
    if st.button("📦 Day 332 (Impulse)", use_container_width=True):
        st.session_state.selected_sku_id = "HOUSEHOLD_2_440_TX_1_validation"
        st.session_state.current_day_num = 332
        st.session_state.slider_sidebar = 332
        st.session_state.slider_main = 332
        st.rerun()

# 2. Always-Visible Live Catalog Search (9,147 Products)
st.sidebar.markdown("---")
st.sidebar.markdown("<div style='font-size:0.82rem; font-weight:700; color:#38BDF8; margin-bottom:6px;'>🔍 Live Catalog Search (9,147 SKUs):</div>", unsafe_allow_html=True)

col_f1, col_f2 = st.sidebar.columns(2)
with col_f1:
    f_cat = st.selectbox("Category:", ["All", "FOODS", "HOBBIES", "HOUSEHOLD"], index=0)
with col_f2:
    f_state = st.selectbox("Region:", ["All", "TX", "CA", "WI"], index=0)

filtered_df = m5_df.copy()
if f_cat != "All":
    filtered_df = filtered_df[filtered_df['cat_id'] == f_cat]
if f_state != "All":
    filtered_df = filtered_df[filtered_df['state_id'] == f_state]

sku_options = filtered_df['id'].tolist()
if not sku_options:
    sku_options = m5_df['id'].tolist()

if st.session_state.selected_sku_id in sku_options:
    sku_idx = sku_options.index(st.session_state.selected_sku_id)
else:
    sku_options.insert(0, st.session_state.selected_sku_id)
    sku_idx = 0

chosen_sku = st.sidebar.selectbox(
    f"Active Product ({len(sku_options):,} available):",
    options=sku_options,
    index=sku_idx,
    key=f"sku_picker_{st.session_state.selected_sku_id}"
)

if chosen_sku != st.session_state.selected_sku_id:
    st.session_state.selected_sku_id = chosen_sku
    st.rerun()

current_sku = st.session_state.selected_sku_id
sku_row_meta = m5_df[m5_df['id'] == current_sku].iloc[0]
current_state = sku_row_meta['state_id']
current_cat = sku_row_meta['cat_id']

# 3. Sidebar Timeline Slider
st.sidebar.markdown("---")
st.sidebar.slider(
    "Timeline Scrubber (Historical Replay):",
    min_value=1,
    max_value=365,
    key="slider_sidebar",
    on_change=on_sidebar_slider_change,
    step=1,
    format="Day %d"
)
current_day = st.session_state.current_day_num
day_col_tag = f"d_{current_day}"

st.sidebar.markdown("""
<div style='background:#1E293B; border-radius:6px; padding:8px 10px; font-size:0.72rem; color:#94A3B8; margin-top:12px;'>
    <b>Simulation Architecture:</b><br/>
    Autonomous Digital Twin historical replay over official Walmart M5 benchmark data. Every slider scrub evaluates real-time event sensing & physical rebalancing.
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. Data Extraction & Real-time Simulation
# -----------------------------------------------------------------------------
day_cols = [f"d_{i}" for i in range(1, 366)]
sku_row = m5_df[m5_df['id'] == current_sku].iloc[0]
full_sales = [float(sku_row[c]) for c in day_cols]
cal_row = cal_df[cal_df['d'] == day_col_tag].iloc[0]
date_str = str(cal_row.get('date', 'Unknown Date'))

# Execute 365-Day Continuous Digital Twin Simulation
sim_days, anom_df = run_continuous_simulation(current_sku, current_state, current_cat)
day_data = sim_days[current_day]
anom_info = day_data['anom']
inv_info = day_data['inv']
comp_stocks = day_data['comp']
shipments = day_data['shipments']

today_sales = int(anom_info['actual_sales'])
today_baseline = round(float(anom_info['baseline_mean']), 2) if pd.notnull(anom_info['baseline_mean']) else 0.0
today_z = round(float(anom_info['z_score']), 2) if pd.notnull(anom_info['z_score']) else 0.0
today_status = anom_info['status']

current_wh_stock = inv_info['closing_stock']
current_wh_safety = comp_stocks.get(current_state, {}).get('safety_stock', 10)

# Evaluate live multi-agent dialogue for current_day
is_surge = (today_z >= 2.0 and today_sales > 0)
is_plateau = ("PLATEAU" in today_status and today_sales > 0)
is_severe_drop = (today_z <= -2.5 and today_baseline >= 5.0 and today_sales == 0)
is_event = (is_surge or is_plateau or is_severe_drop or inv_info['stockout_occurred'] or "WARNING" in inv_info['action_taken'] or "CRITICAL" in inv_info['action_taken'] or "FRAGILE" in inv_info['action_taken'] or "PLATEAU" in inv_info['action_taken'])

if is_event:
    cal_r = cal_df[cal_df['d'] == day_col_tag].iloc[0]
    anom_dict = dict(anom_info)
    anom_dict['sku_id'] = current_sku
    anom_dict['cat_id'] = current_cat
    anom_dict['state_id'] = current_state
    
    diag = orchestrator.agent1.reason_engine.evaluate_root_cause(anom_dict, cal_r, cross_state_spiked=False)
    a1_payload = {
        "sender": "Agent 1 (Demand Detective)",
        "recipient": "Agent 2 (Inventory Operator)",
        "timestamp_day": day_col_tag,
        "date": str(cal_r.get('date', '')),
        "sku_id": current_sku,
        "category": current_cat,
        "state_id": current_state,
        "metrics": {
            "actual_demand": today_sales,
            "baseline_mean": today_baseline,
            "z_score": today_z,
            "status": today_status,
            "streak_day": int(anom_info.get('streak_day', 1))
        },
        "root_cause_diagnosis": diag,
        "executive_summary": diag.get('executive_summary', '')
    }
    
    actions = [act.strip() for act in inv_info['action_taken'].split(" | ")]
    a2_payload = {
        "sender": "Agent 2 (Inventory Operator)",
        "recipient": "Agent 1 (Demand Detective)",
        "timestamp_day": day_col_tag,
        "sku_id": current_sku,
        "state_id": current_state,
        "warehouse_accounting": {
            "opening_stock": inv_info['opening_stock'],
            "inbound_received": inv_info['inbound_received'],
            "actual_demand": today_sales,
            "fulfilled_demand": inv_info['fulfilled_demand'],
            "unmet_demand_backlog": inv_info['unmet_demand'],
            "closing_stock": inv_info['closing_stock'],
            "stockout_incident": inv_info['stockout_occurred']
        },
        "risk_and_runway": {
            "runway_impulse_days": inv_info['runway_impulse_days'],
            "runway_sustained_days": inv_info['runway_sustained_days'],
            "lead_time_days": 3
        },
        "decisions_and_actions": actions,
        "conversational_dialogue": f"Physical inventory state updated. Warehouse balance is {inv_info['closing_stock']} units with 0 backorders." if inv_info['unmet_demand'] == 0 else f"CRITICAL: Stockout incident active. Clamped to 0. Dispatched emergency transfers and expedited orders."
    }
    handshake = {
        "agent_1_outbound": a1_payload,
        "agent_2_inbound_response": a2_payload
    }
else:
    handshake = None

# -----------------------------------------------------------------------------
# 5. Top Header & 5 Compact KPI Status Cards
# -----------------------------------------------------------------------------
st.markdown("<div class='system-title'>⚡ Autonomous Demand-Sensing & Inventory Rebalancing Engine</div>", unsafe_allow_html=True)
st.markdown(f"<div class='system-subtitle'>Enterprise Digital Twin • SKU: <b>{current_sku}</b> | Region: <b>{current_state} Hub</b> | Timeline: <b>{date_str} (Day {current_day})</b></div>", unsafe_allow_html=True)

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-title'>Today's POS Sales</div>
        <div class='metric-value'>{today_sales:,} <span style='font-size:0.75rem; color:#94A3B8;'>units</span></div>
        <div class='metric-subtext'>Expected: {today_baseline:.1f} units</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    z_color = "#EF4444" if today_z >= 2.5 else ("#F59E0B" if today_z >= 1.0 else "#10B981")
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-title'>Statistical Deviation</div>
        <div class='metric-value' style='color:{z_color};'>{today_z:+.2f}σ</div>
        <div class='metric-subtext'>Threshold: Entry ≥ +2.50σ</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    badge_class = "badge-critical" if "PLATEAU" in today_status or today_z >= 2.5 else ("badge-provisional" if "PROVISIONAL" in today_status else "badge-healthy")
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-title'>Demand Agent Status</div>
        <div style='margin-top:4px; margin-bottom:4px;'><span class='{badge_class}'>{today_status}</span></div>
        <div class='metric-subtext'>Hysteresis State Machine</div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    stock_color = "#EF4444" if current_wh_stock == 0 else ("#F59E0B" if current_wh_stock <= current_wh_safety else "#10B981")
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-title'>Physical Warehouse Stock</div>
        <div class='metric-value' style='color:{stock_color};'>{current_wh_stock:,} <span style='font-size:0.75rem; color:#94A3B8;'>units</span></div>
        <div class='metric-subtext'>Safety Buffer: {current_wh_safety} units</div>
    </div>
    """, unsafe_allow_html=True)

with kpi5:
    unmet_count = inv_info['unmet_demand']
    unmet_color = "#EF4444" if unmet_count > 0 else "#10B981"
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-title'>Unmet Demand Backlog</div>
        <div class='metric-value' style='color:{unmet_color};'>{unmet_count:,} <span style='font-size:0.75rem; color:#94A3B8;'>units</span></div>
        <div class='metric-subtext'>Zero-Clamping Enforced</div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. Main Cockpit: Side-by-Side Chart & Live Multi-Agent Dialogue
# -----------------------------------------------------------------------------
col_chart, col_dialogue = st.columns([1.35, 1.0])

with col_chart:
    st.markdown("<div style='font-size:0.88rem; font-weight:700; color:#F1F5F9; margin-bottom:4px;'>📈 POS Demand & Physical Inventory Trajectory (365 Days)</div>", unsafe_allow_html=True)
    
    # Compute display slice (show from max(1, current_day - 50) to current_day)
    start_view = max(0, current_day - 50)
    view_days = list(range(start_view + 1, current_day + 1))
    view_dates = [cal_df[cal_df['d'] == f"d_{d}"]['date'].iloc[0] for d in view_days]
    view_sales = full_sales[start_view:current_day]
    
    # Calculate baseline curve for view window
    view_baselines = []
    view_sigmas = []
    for d in view_days:
        sub_row = anom_df[anom_df['day'] == f"d_{d}"]
        if not sub_row.empty:
            b_val = sub_row['baseline_mean'].iloc[0]
            s_val = sub_row['baseline_std'].iloc[0]
            view_baselines.append(float(b_val) if not pd.isna(b_val) else full_sales[d-1])
            view_sigmas.append(float(s_val) if (not pd.isna(s_val) and float(s_val) > 0) else 1.0)
        else:
            view_baselines.append(full_sales[d-1])
            view_sigmas.append(1.0)
            
    upper_tunnel = [b + (2.0 * s) for b, s in zip(view_baselines, view_sigmas)]
    lower_tunnel = [max(0.0, b - (2.0 * s)) for b, s in zip(view_baselines, view_sigmas)]

    fig = go.Figure()

    # 1. Shaded Confidence Tunnel (+/- 2 Sigma)
    fig.add_trace(go.Scatter(
        x=view_dates, y=upper_tunnel,
        mode='lines', line=dict(color='rgba(148, 163, 184, 0.0)'),
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=view_dates, y=lower_tunnel,
        mode='lines', line=dict(color='rgba(148, 163, 184, 0.0)'),
        fill='tonexty', fillcolor='rgba(148, 163, 184, 0.12)',
        name='Noise Tunnel (±2σ)', hoverinfo='skip'
    ))

    # 2. Rolling Baseline Mean
    fig.add_trace(go.Scatter(
        x=view_dates, y=view_baselines,
        mode='lines', line=dict(color='#94A3B8', width=1.5, dash='dash'),
        name='Baseline (μ)'
    ))

    # 3. Actual POS Demand Line
    fig.add_trace(go.Scatter(
        x=view_dates, y=view_sales,
        mode='lines+markers',
        line=dict(color='#38BDF8', width=2.5),
        marker=dict(size=5, color='#38BDF8'),
        name='Actual Sales (POS)'
    ))

    # 3b. Physical Warehouse Stock Level (Sawtooth curve)
    view_stocks = [sim_days[d]['inv']['closing_stock'] for d in view_days]
    fig.add_trace(go.Scatter(
        x=view_dates, y=view_stocks,
        mode='lines',
        line=dict(color='#10B981', width=2),
        name='Warehouse Stock'
    ))

    # 4. Highlight Active Day with Target Marker
    today_marker_color = '#EF4444' if today_z >= 2.5 else ('#F59E0B' if today_z >= 1.0 else '#10B981')
    fig.add_trace(go.Scatter(
        x=[date_str], y=[today_sales],
        mode='markers',
        marker=dict(size=13, color=today_marker_color, line=dict(color='#FFFFFF', width=2)),
        name=f"Day {current_day} ({today_z:+.2f}σ)"
    ))

    fig.update_layout(
        paper_bgcolor='#111827',
        plot_bgcolor='#111827',
        margin=dict(l=15, r=15, t=10, b=10),
        height=335,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1.0,
            font=dict(size=9, color='#94A3B8')
        ),
        xaxis=dict(
            gridcolor='#1F2B48',
            tickfont=dict(size=9, color='#94A3B8'),
            showgrid=True
        ),
        yaxis=dict(
            gridcolor='#1F2B48',
            tickfont=dict(size=9, color='#94A3B8'),
            showgrid=True,
            title=dict(text="Units", font=dict(size=10, color='#94A3B8'))
        )
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.slider(
        "⏩ Timeline Scrubber (Scrub 365 Days of Digital Twin History):",
        min_value=1,
        max_value=365,
        key="slider_main",
        on_change=on_main_slider_change,
        step=1,
        format="Day %d"
    )

with col_dialogue:
    st.markdown("<div style='font-size:0.88rem; font-weight:700; color:#F1F5F9; margin-bottom:4px;'>💬 Live Inter-Agent Operations Dialogue</div>", unsafe_allow_html=True)
    
    if handshake:
        a1 = handshake['agent_1_outbound']
        a2 = handshake['agent_2_inbound_response']
        
        # Agent 1 Card
        st.markdown(f"""
        <div class='agent-card' style='border-left-color: #38BDF8;'>
            <div class='agent-1-header'>
                <span>🔍</span> Agent 1: Demand Detective (Market Sensing)
            </div>
            <div class='agent-body'>
                <b>Observed:</b> {a1['metrics']['actual_demand']} units vs {a1['metrics']['baseline_mean']} baseline (<b>{a1['metrics']['z_score']:+.2f}σ</b>)<br/>
                <b>Causal Diagnosis:</b> <span style='color:#FBBF24; font-weight:600;'>{a1['root_cause_diagnosis']['classification']}</span><br/>
                <i>"{a1['executive_summary']}"</i>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Agent 2 Card
        wh_acc = a2['warehouse_accounting']
        rw_acc = a2['risk_and_runway']
        
        # Clear, unambiguous fulfillment phrasing
        if wh_acc['unmet_demand_backlog'] > 0:
            fulfillment_markup = f"<span style='color:#EF4444; font-weight:700;'>{wh_acc['fulfilled_demand']} / {wh_acc['actual_demand']} units fulfilled</span> — <span style='color:#F87171;'><b>{wh_acc['unmet_demand_backlog']} units unmet backlog</b></span>, <span style='color:#94A3B8;'>0 units left in warehouse (Zero-Clamped)</span>"
        else:
            fulfillment_markup = f"<span style='color:#34D399; font-weight:700;'>{wh_acc['fulfilled_demand']} / {wh_acc['actual_demand']} units fulfilled (100% complete)</span> — <span style='color:#E2E8F0;'><b>{wh_acc['closing_stock']} units remaining</b> in warehouse stock</span> <span style='color:#64748B;'>(0 unmet)</span>"
            
        imp_txt = f"{rw_acc['runway_impulse_days']:.1f}d" if rw_acc['runway_impulse_days'] is not None else "N/A"
        sus_txt = f"{rw_acc['runway_sustained_days']:.1f}d" if rw_acc['runway_sustained_days'] is not None else "N/A"

        st.markdown(f"""
        <div class='agent-card' style='border-left-color: #34D399;'>
            <div class='agent-2-header'>
                <span>📦</span> Agent 2: Inventory Operator (Execution & Physical Balancing)
            </div>
            <div class='agent-body'>
                <b>Fulfillment Status:</b> {fulfillment_markup}<br/>
                <b>Dual Runway:</b> Impulse: {imp_txt} | Sustained: <b>{sus_txt}</b> (Lead time: 3d)<br/>
                <b>Operational Actions:</b><br/>
        """, unsafe_allow_html=True)
        
        for act in a2['decisions_and_actions']:
            st.markdown(f"<div style='font-size:0.8rem; margin-left:10px; color:#E2E8F0;'>• {act}</div>", unsafe_allow_html=True)
            
        st.markdown(f"""
                <div style='margin-top:6px; font-style:italic; font-size:0.78rem; color:#94A3B8;'>"{a2['conversational_dialogue']}"</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif today_status == "CALIBRATION_PERIOD" or current_day <= 14:
        st.markdown(f"""
        <div class='agent-card' style='border-left-color: #38BDF8;'>
            <div style='color:#38BDF8; font-size:0.88rem; font-weight:700; margin-bottom:4px;'>
                ⚙️ Baseline Calibration Phase (Day {current_day} of 14)
            </div>
            <div style='color:#CBD5E1; font-size:0.82rem; line-height:1.4;'>
                System is actively profiling historical distribution and 6-week day-of-week seasonality (14-day warm-up window). 
                Automated statistical anomaly detection and autonomous multi-agent rebalancing will activate on Day 15.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='agent-card' style='border-left-color: #64748B;'>
            <div style='color:#94A3B8; font-size:0.85rem;'>
                <b>Nominal Operating Conditions:</b> Demand signals tracking within standard ±2σ confidence bounds.
                Agent 1 maintains automated passive surveillance. Agent 2 routine replenishments active.
            </div>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 7. Collapsible Multi-Warehouse Network Topology & Highway Pipeline
# -----------------------------------------------------------------------------
with st.expander("🌐 Multi-Warehouse Network Topology & Highway In-Transit Pipeline (Click to Expand)", expanded=False):
    col_tx, col_ca, col_wi, col_ship = st.columns([1, 1, 1, 1.5])

    tx_m = comp_stocks.get("TX", {})
    ca_m = comp_stocks.get("CA", {})
    wi_m = comp_stocks.get("WI", {})

    tx_stock = tx_m.get('current_stock', inv_info['closing_stock'])
    tx_ss = tx_m.get('safety_stock', 10)
    tx_surplus = max(0, tx_stock - (2 * tx_ss))

    ca_stock = ca_m.get('current_stock', 0)
    ca_ss = ca_m.get('safety_stock', 10)
    ca_surplus = max(0, ca_stock - (2 * ca_ss))

    wi_stock = wi_m.get('current_stock', 0)
    wi_ss = wi_m.get('safety_stock', 10)
    wi_surplus = max(0, wi_stock - (2 * wi_ss))

    # Detect transfers dispatched today from companion hubs
    ca_transferred_today = sum(s['quantity'] for s in shipments if s.get('order_day') == current_day and 'CA' in s.get('source_location', ''))
    wi_transferred_today = sum(s['quantity'] for s in shipments if s.get('order_day') == current_day and 'WI' in s.get('source_location', ''))

    ca_status_line = f"<span style='color:#38BDF8; font-weight:600;'>🚛 -{ca_transferred_today} units dispatched to TX</span>" if ca_transferred_today > 0 else (f"<span style='color:#10B981;'>Ready buffer</span>" if ca_surplus > 0 else "<span style='color:#94A3B8;'>At safety threshold</span>")
    wi_status_line = f"<span style='color:#38BDF8; font-weight:600;'>🚛 -{wi_transferred_today} units dispatched to TX</span>" if wi_transferred_today > 0 else (f"<span style='color:#10B981;'>Ready buffer</span>" if wi_surplus > 0 else "<span style='color:#94A3B8;'>At safety threshold</span>")

    with col_tx:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>📍 Texas (Primary Hub)</div>
            <div class='metric-value'>{tx_stock:,} <span style='font-size:0.75rem; color:#94A3B8;'>units</span></div>
            <div class='metric-subtext'>Safety Stock: {tx_ss} | Surplus: {tx_surplus}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_ca:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>📍 California (Companion Hub)</div>
            <div class='metric-value'>{ca_stock:,} <span style='font-size:0.75rem; color:#94A3B8;'>units</span></div>
            <div class='metric-subtext'>Safety: {ca_ss} | Surplus: <b>{ca_surplus:,} units</b><br/>{ca_status_line}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_wi:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>📍 Wisconsin (Companion Hub)</div>
            <div class='metric-value'>{wi_stock:,} <span style='font-size:0.75rem; color:#94A3B8;'>units</span></div>
            <div class='metric-subtext'>Safety: {wi_ss} | Surplus: <b>{wi_surplus:,} units</b><br/>{wi_status_line}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_ship:
        if shipments:
            display_tbl = pd.DataFrame(shipments)[['shipment_type', 'quantity', 'arrival_day', 'source_location', 'status']].copy()
            display_tbl.columns = ['Type', 'Units', 'Arrives Day', 'Source', 'Status']
            st.dataframe(display_tbl, hide_index=True, use_container_width=True)
        else:
            st.markdown("<div style='color:#64748B; font-size:0.8rem; padding:8px;'>No active shipments currently on the highway. All historical orders delivered.</div>", unsafe_allow_html=True)
