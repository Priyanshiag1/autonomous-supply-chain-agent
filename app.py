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

# Initialize Persistent Warehouse Engine
@st.cache_resource
def get_inventory_engine():
    db_path = "inventory.db"
    engine = WarehouseInventoryEngine(db_path=db_path)
    engine.init_db()
    engine.seed_initial_inventory(m5_df, hero_df['id'].tolist())
    return engine

inv_engine = get_inventory_engine()
orchestrator = MultiAgentSystemOrchestrator(inventory_engine=inv_engine, warmup_days=7)

# -----------------------------------------------------------------------------
# 3. Sidebar Navigation: Curated Hero Archetypes vs Full Catalog Explorer
# -----------------------------------------------------------------------------
st.sidebar.markdown("<div style='font-size:1.1rem; font-weight:800; color:#38BDF8;'>INTELLIMARK AI</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='font-size:0.75rem; color:#94A3B8; margin-bottom:12px;'>Autonomous Supply Chain Digital Twin</div>", unsafe_allow_html=True)

nav_mode = st.sidebar.radio(
    "Navigation Mode:",
    options=["🎯 Curated Hero Benchmarks (CEO Demo)", "🔍 Full Catalog Explorer (9,147 SKUs)"],
    index=0
)

# Curated Hero Archetypes
hero_scenarios = {
    "🏈 SuperBowl Party Snacks (FOODS_3_090_TX)": {
        "sku_id": "FOODS_3_090_TX_1_validation",
        "state_id": "TX",
        "cat_id": "FOODS",
        "default_day": 9,
        "description": "SuperBowl Sunday spike (Day 9). Single-day surge with comfortable stock & standby PO."
    },
    "💥 86x Mega SNAP Outlier & Stockout (FOODS_2_285_TX)": {
        "sku_id": "FOODS_2_285_TX_1_validation",
        "state_id": "TX",
        "cat_id": "FOODS",
        "default_day": 98,
        "description": "Massive 86.6x SNAP welfare shock. Physical zero-clamping, CA transfer & factory PO."
    },
    "🛡️ Toys & Crafts SNAP Defense (HOBBIES_1_209_TX)": {
        "sku_id": "HOBBIES_1_209_TX_1_validation",
        "state_id": "TX",
        "cat_id": "HOBBIES",
        "default_day": 126,
        "description": "Spurious SNAP welfare flag rejected by Category Guardrail. Diagnosed as B2B Wholesale."
    },
    "📦 Cleaners Single-Day Impulse (HOUSEHOLD_2_440_TX)": {
        "sku_id": "HOUSEHOLD_2_440_TX_1_validation",
        "state_id": "TX",
        "cat_id": "HOUSEHOLD",
        "default_day": 332,
        "description": "Post-Christmas single-day commercial impulse. Unmarked anomaly."
    },
    "📈 Multi-Day Cleaner Plateau & Pipeline (HOUSEHOLD_2_440_TX)": {
        "sku_id": "HOUSEHOLD_2_440_TX_1_validation",
        "state_id": "TX",
        "cat_id": "HOUSEHOLD",
        "default_day": 343,
        "description": "Consecutive multi-day surge. Tests pipeline tracking, anti-bullwhip & fragile recovery."
    }
}

scenario_keys = list(hero_scenarios.keys())

if "active_scenario_name" not in st.session_state:
    st.session_state.active_scenario_name = scenario_keys[0]
if "current_day_num" not in st.session_state:
    st.session_state.current_day_num = hero_scenarios[st.session_state.active_scenario_name]['default_day']

if nav_mode == "🎯 Curated Hero Benchmarks (CEO Demo)":
    st.sidebar.markdown("---")
    st.sidebar.markdown("<div style='font-size:0.8rem; font-weight:700; color:#CBD5E1;'>Quick-Jump to Milestone Days:</div>", unsafe_allow_html=True)

    col_b1, col_b2 = st.sidebar.columns(2)
    with col_b1:
        if st.button("🏈 Day 9 (SuperBowl)", use_container_width=True):
            st.session_state.active_scenario_name = scenario_keys[0]
            st.session_state.current_day_num = 9
            st.rerun()
        if st.button("🛡️ Day 126 (Guardrail)", use_container_width=True):
            st.session_state.active_scenario_name = scenario_keys[2]
            st.session_state.current_day_num = 126
            st.rerun()
        if st.button("📈 Day 343 (Plateau)", use_container_width=True):
            st.session_state.active_scenario_name = scenario_keys[4]
            st.session_state.current_day_num = 343
            st.rerun()
    with col_b2:
        if st.button("💥 Day 98 (Mega SNAP)", use_container_width=True):
            st.session_state.active_scenario_name = scenario_keys[1]
            st.session_state.current_day_num = 98
            st.rerun()
        if st.button("📦 Day 332 (Impulse)", use_container_width=True):
            st.session_state.active_scenario_name = scenario_keys[3]
            st.session_state.current_day_num = 332
            st.rerun()

    current_idx = scenario_keys.index(st.session_state.active_scenario_name) if st.session_state.active_scenario_name in scenario_keys else 0

    selected_scenario_name = st.sidebar.selectbox(
        "Curated Product & Scenario Archetype:",
        options=scenario_keys,
        index=current_idx
    )

    if selected_scenario_name != st.session_state.active_scenario_name:
        st.session_state.active_scenario_name = selected_scenario_name
        st.session_state.current_day_num = hero_scenarios[selected_scenario_name]['default_day']
        st.rerun()

    selected_scenario = hero_scenarios[st.session_state.active_scenario_name]
    current_sku = selected_scenario['sku_id']
    current_state = selected_scenario['state_id']
    current_cat = selected_scenario['cat_id']

else:
    # Full Catalog Explorer (All 9,147 Series)
    st.sidebar.markdown("---")
    st.sidebar.markdown("<div style='font-size:0.8rem; font-weight:700; color:#38BDF8;'>🔍 Catalog Explorer Filters:</div>", unsafe_allow_html=True)
    
    f_cat = st.sidebar.selectbox("Filter Category:", ["All", "FOODS", "HOBBIES", "HOUSEHOLD"])
    f_state = st.sidebar.selectbox("Filter State:", ["All", "TX", "CA", "WI"])
    
    filtered_df = m5_df.copy()
    if f_cat != "All":
        filtered_df = filtered_df[filtered_df['cat_id'] == f_cat]
    if f_state != "All":
        filtered_df = filtered_df[filtered_df['state_id'] == f_state]
        
    sku_options = filtered_df['id'].tolist()
    
    if not sku_options:
        sku_options = m5_df['id'].tolist()
        
    current_sku = st.sidebar.selectbox(
        f"Select SKU ({len(sku_options):,} series available):",
        options=sku_options,
        index=0
    )
    
    sku_row_meta = m5_df[m5_df['id'] == current_sku].iloc[0]
    current_state = sku_row_meta['state_id']
    current_cat = sku_row_meta['cat_id']

# Timeline Slider
st.sidebar.markdown("---")
day_slider = st.sidebar.slider(
    "Timeline Scrubber (Historical Replay):",
    min_value=1,
    max_value=365,
    value=st.session_state.current_day_num,
    step=1,
    format="Day %d"
)
st.session_state.current_day_num = day_slider
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

# Run Anomaly Engine over the historical sequence up to current_day
anomaly_engine = AnomalyDetectionEngine(warmup_days=7)
anomaly_df = anomaly_engine.analyze_series(
    full_sales[:current_day],
    cal_df,
    current_sku,
    current_state,
    current_cat
)

current_anom_row = anomaly_df[anomaly_df['day'] == day_col_tag].iloc[0]
today_sales = int(current_anom_row['actual_sales'])
today_baseline = round(float(current_anom_row['baseline_mean']), 2)
today_z = round(float(current_anom_row['z_score']), 2)
today_status = current_anom_row['status']

# CRITICAL IDEMPOTENCY: Reset scenario baseline before processing day
# Guarantees zero cumulative drift across repeated evaluations and refreshes
inv_engine.reset_scenario_baseline(
    sku_id=current_sku,
    state_id=current_state,
    day_index=current_day,
    m5_df=m5_df
)

# Execute Multi-Agent Handshake
handshake = orchestrator.process_day(
    day_index=current_day,
    sku_id=current_sku,
    state_id=current_state,
    cat_id=current_cat,
    historical_sales=full_sales,
    calendar_df=cal_df
)

# Fetch Current Warehouse State from SQLite
stock_record = inv_engine.get_stock_record(current_sku, current_state)
current_wh_stock = stock_record['current_stock'] if stock_record else 0
current_wh_safety = stock_record['safety_stock'] if stock_record else 0

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
    unmet_count = handshake['agent_2_inbound_response']['warehouse_accounting']['unmet_demand_backlog'] if handshake else 0
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
    st.markdown("<div style='font-size:0.88rem; font-weight:700; color:#F1F5F9; margin-bottom:4px;'>📈 POS Demand vs Rolling Confidence Tunnel (365 Days)</div>", unsafe_allow_html=True)
    
    # Compute display slice (show from max(1, current_day - 50) to current_day)
    start_view = max(0, current_day - 50)
    view_days = list(range(start_view + 1, current_day + 1))
    view_dates = [cal_df[cal_df['d'] == f"d_{d}"]['date'].iloc[0] for d in view_days]
    view_sales = full_sales[start_view:current_day]
    
    # Calculate baseline curve for view window
    view_baselines = []
    view_sigmas = []
    for d in view_days:
        sub_row = anomaly_df[anomaly_df['day'] == f"d_{d}"]
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
            
        st.markdown(f"""
        <div class='agent-card' style='border-left-color: #34D399;'>
            <div class='agent-2-header'>
                <span>📦</span> Agent 2: Inventory Operator (Execution & Physical Balancing)
            </div>
            <div class='agent-body'>
                <b>Fulfillment Status:</b> {fulfillment_markup}<br/>
                <b>Dual Runway:</b> Impulse: {rw_acc['runway_impulse_days']:.1f}d | Sustained: <b>{rw_acc['runway_sustained_days']:.1f}d</b> (Lead time: 3d)<br/>
                <b>Operational Actions:</b><br/>
        """, unsafe_allow_html=True)
        
        for act in a2['decisions_and_actions']:
            st.markdown(f"<div style='font-size:0.8rem; margin-left:10px; color:#E2E8F0;'>• {act}</div>", unsafe_allow_html=True)
            
        st.markdown(f"""
                <div style='margin-top:6px; font-style:italic; font-size:0.78rem; color:#94A3B8;'>"{a2['conversational_dialogue']}"</div>
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

    # Fetch all 3 state balances for current product
    with inv_engine.get_connection() as conn:
        item_prefix = "_".join(current_sku.replace("_validation", "").split("_")[:3])
        df_net = pd.read_sql_query(
            "SELECT state_id, current_stock, safety_stock, lead_time_days FROM warehouse_inventory WHERE sku_id LIKE ?",
            conn, params=(f"{item_prefix}%",)
        )
        
        # Active shipments
        df_ship = pd.read_sql_query(
            "SELECT shipment_id, quantity, shipment_type, order_day, arrival_day, source_location, status FROM inbound_shipments WHERE sku_id LIKE ? ORDER BY arrival_day ASC",
            conn, params=(f"{item_prefix}%",)
        )

    def get_state_metrics(st_code: str):
        m = df_net[df_net['state_id'] == st_code]
        if not m.empty:
            stock = m['current_stock'].iloc[0]
            ss = m['safety_stock'].iloc[0]
            surplus = max(0, stock - (2 * ss))
            return stock, ss, surplus
        return 0, 0, 0

    tx_stock, tx_ss, tx_surplus = get_state_metrics("TX")
    ca_stock, ca_ss, ca_surplus = get_state_metrics("CA")
    wi_stock, wi_ss, wi_surplus = get_state_metrics("WI")

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
            <div class='metric-title'>📍 California (Donor Hub)</div>
            <div class='metric-value'>{ca_stock:,} <span style='font-size:0.75rem; color:#94A3B8;'>units</span></div>
            <div class='metric-subtext'>Safety Stock: {ca_ss} | Available Surplus: <b>{ca_surplus:,} units</b></div>
        </div>
        """, unsafe_allow_html=True)

    with col_wi:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>📍 Wisconsin (Companion Hub)</div>
            <div class='metric-value'>{wi_stock:,} <span style='font-size:0.75rem; color:#94A3B8;'>units</span></div>
            <div class='metric-subtext'>Safety Stock: {wi_ss} | Available Surplus: <b>{wi_surplus:,} units</b></div>
        </div>
        """, unsafe_allow_html=True)

    with col_ship:
        if not df_ship.empty:
            active_p = df_ship[df_ship['arrival_day'] >= current_day]
            if not active_p.empty:
                display_tbl = active_p[['shipment_type', 'quantity', 'arrival_day', 'source_location', 'status']].copy()
                display_tbl.columns = ['Type', 'Units', 'Arrives Day', 'Source', 'Status']
                st.dataframe(display_tbl, hide_index=True, use_container_width=True)
            else:
                st.markdown("<div style='color:#64748B; font-size:0.8rem; padding:8px;'>No active shipments currently on the highway. All historical orders delivered.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='color:#64748B; font-size:0.8rem; padding:8px;'>No active shipments recorded in database for this product.</div>", unsafe_allow_html=True)
