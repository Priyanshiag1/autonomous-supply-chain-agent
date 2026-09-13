# ⚡ StockSentinel — Autonomous Demand-Sensing & Inventory Rebalancing Digital Twin

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/UI-Streamlit%20%7C%20Plotly-FF4B4B.svg)](https://streamlit.io/)
[![Database](https://img.shields.io/badge/Ledger-SQLite3-003B57.svg)](https://www.sqlite.org/)
[![Benchmark](https://img.shields.io/badge/Data-Walmart%20M5%20Forecasting-0071DC.svg)](https://mofc.unic.ac.cy/m5-competition/)
[![Architecture](https://img.shields.io/badge/Multi--Agent-Communicating%20State%20Machine-success.svg)](#1-system-architecture)

**StockSentinel** is an enterprise-grade **Autonomous Multi-Agent Supply Chain Defense Network**. Built on 365 days of real Walmart M5 retail data across 9,147 SKUs, it pairs statistical time-series anomaly detection with causal semantic guardrails and a zero-clamped physical inventory state machine to autonomously sense demand shocks, eliminate the bullwhip effect, and execute multi-echelon warehouse rebalancing.

---

## 1. System Architecture

```mermaid
flowchart TD
    classDef sensing fill:#1E293B,stroke:#38BDF8,stroke-width:2px,color:#F8FAFC;
    classDef reasoning fill:#1E293B,stroke:#FBBF24,stroke-width:2px,color:#F8FAFC;
    classDef protocol fill:#0F172A,stroke:#818CF8,stroke-width:2px,color:#F8FAFC;
    classDef execution fill:#1E293B,stroke:#34D399,stroke-width:2px,color:#F8FAFC;
    classDef ui fill:#1E293B,stroke:#EC4899,stroke-width:2px,color:#F8FAFC;

    subgraph L1 ["📡 Layer 1: Real-Time Demand Sensing"]
        M5["📊 Real Walmart M5 POS Stream (9,147 SKUs • 365 Days)"]:::sensing
        A1["🔍 Agent 1: Demand Detective<br/>• Rolling Mean μ ± 2σ Noise Tunnel<br/>• Hysteresis Trigger (Entry ≥ +2.5σ, Exit ≤ 1.0σ)<br/>• 3-Zone Rolling State Machine"]:::sensing
    end

    subgraph L2 ["🧠 Layer 2: Causal Intelligence & Guardrails"]
        G1{"Category Affinity<br/>Guardrail"}:::reasoning
        CAL["📅 Tier 1: Calendar Intelligence<br/>(SuperBowl, SNAP, Cultural Events)"]:::reasoning
        SHP["📈 Tier 2/3: Statistical Shape Heuristics<br/>(B2B Impulse, Multi-Day Surge, Plateau)"]:::reasoning
        REJECT["🛡️ Spurious Correlation Block<br/>(e.g., Bar SNAP on HOBBIES)"]:::reasoning
    end

    subgraph L3 ["⚡ Layer 3: Inter-Agent Protocol Handshake"]
        BUS["🤝 Structured JSON Protocol Payload<br/>{SKU, Spike Units, Z-Score, Causal Diagnosis, Risk Brackets}"]:::protocol
    end

    subgraph L4 ["📦 Layer 4: Physical Warehouse Balancing & Ledger"]
        DB[("💾 SQLite ERP Ledger (inventory.db)<br/>Safety Stock • Zero-Clamping • Lead Time=3d")]:::execution
        A2["⚙️ Agent 2: Inventory Operator<br/>• Zero Stock Clamping & Backlog Accounting<br/>• Dual Runway Evaluation (Impulse vs Sustained)"]:::execution
        DEC{"Runway ≤ 3-Day<br/>Lead Time?"}:::execution
        TRANSFER["🔄 Emergency Inter-Warehouse Transfer<br/>(Nearest Regional Surplus Hub)"]:::execution
        PO["📦 Expedited Standard Supplier PO"]:::execution
        NORMAL["✅ Nominal Routine Replenishment"]:::execution
    end

    subgraph L5 ["💻 Layer 5: Operator Digital Twin Cockpit"]
        DASH["⚡ StockSentinel Interactive Dashboard<br/>• Plotly Trajectory with Click-to-Jump Spike Diamonds<br/>• Dual Synchronized 365-Day Timeline Scrubbers<br/>• Live Multi-Agent Operations Dialogue Feed"]:::ui
    end

    M5 --> A1
    A1 --> G1
    G1 -- "Eligible Category" --> CAL
    G1 -- "Ineligible (Spurious)" --> REJECT --> SHP
    CAL --> BUS
    SHP --> BUS
    BUS --> A2
    A2 <--> DB
    A2 --> DEC
    DEC -- "Critical Deficit" --> TRANSFER & PO
    DEC -- "Safe Runway" --> NORMAL
    TRANSFER & PO & NORMAL --> DASH
    A1 -. "Live Telemetry" .-> DASH
```

### Component Breakdown

| Layer | Component | Core Responsibility | Output / Artifact |
| :--- | :--- | :--- | :--- |
| **1. Sensing** | `AnomalyDetectionEngine` | Day-of-week rolling baseline, $\pm2\sigma$ confidence tunnel, hysteresis state machine | Anomaly trigger ($Z \ge +2.5\sigma$) |
| **2. Reasoning** | `RootCauseIntelligenceEngine` | Cross-category guardrail audit, holiday calendar matching, streak heuristics | Causal diagnosis (`SNAP`, `SuperBowl`, `B2B Impulse`) |
| **3. Protocol** | `MultiAgentSystemOrchestrator` | Standardized agent-to-agent JSON contract | Inter-agent structured handshake payload |
| **4. Execution** | `WarehouseInventoryEngine` | Zero-clamping, safety stock buffers, multi-echelon regional transfers | Idempotent SQLite state, emergency PO dispatch |
| **5. Interface** | `app.py` (Streamlit + Plotly) | Single-screen operator cockpit with click-to-jump interactive trajectory | Real-time visual digital twin |

---

## 2. Key Technical Innovations

### I. Category-Event Semantic Relevance Guardrail (Spurious Correlation Defense)
Conventional retail AI blindly pairs any holiday on the calendar with a sales spike. Our system enforces an explicit **Category-Event Semantic Affinity Matrix**:
* **SNAP Welfare (Food Stamps):** Legally restricted to `FOODS` ($\text{Affinity} = 1.0$). Spikes in `HOUSEHOLD` or `HOBBIES` on SNAP days **reject** the event as spurious correlation ($\text{Affinity} = 0.0$) and route to statistical shape heuristics.
* **SuperBowl Sunday:** High affinity for `FOODS` party snacks ($0.95$); zero/low affinity for `HOUSEHOLD` ($0.10$).

### II. Physical Warehouse Clamping & Unmet Demand Accounting
Physical warehouses cannot hold negative stock. Our state machine strictly computes:
$$\text{Fulfilled\_Demand}_t = \min(\text{Opening\_Stock}_t, D_t)$$
$$\text{Current\_Stock}_t = \max(0, \text{Opening\_Stock}_t - D_t)$$
$$\text{Unmet\_Demand}_t = \max(0, D_t - \text{Opening\_Stock}_t)$$
When $\text{Unmet\_Demand} > 0$, the system fires an **`ACTIVE_STOCKOUT_INCIDENT`** rather than a predictive warning.

### III. Rolling Multi-Day State Machine (Chicken-and-Egg Resolution)
When a spike occurs on Day $t$, its longevity is unknown. The system does not lock in a premature one-shot verdict:
* **Day $t$:** Issues a `🟡 PROVISIONAL_ALERT` with dual-runway brackets (Best-Case Impulse vs. Worst-Case Sustained).
* **Day $t+1$ (3-Zone Reconciliation):**
  * $Z_{t+1} \le 1.0 \to$ `🟢 DOWNGRADED (RESOLVED_ONE_OFF)`
  * $1.0 < Z_{t+1} < 2.0 \to$ `🟠 PARTIALLY_ELEVATED_WATCH` (Amber cooling state)
  * $Z_{t+1} \ge 2.0 \to$ `🔴 ELEVATED_SURGE (DAY 2)`
* **Day $t+2$:** If surge sustains for 3 consecutive days, executes a **Changepoint Re-Anchoring**, updating the baseline upward to $\mu_{\text{new}}$ while archiving the pre-shift baseline for audit compliance.

### IV. Hysteresis Dual-Threshold Design
* **Entry Barrier ($Z_t \ge 2.5$):** Prevents false alarms on routine seasonal noise (~99th empirical percentile).
* **Exit Barrier ($Z_{t+1} \le 1.0$):** Emergency logistics do not stand down until demand fully returns to the standard noise band ($\le 1.0\sigma$).

---

## 3. Data Provenance & Transparency

| Layer | Source | Details |
| :--- | :--- | :--- |
| **Point-of-Sale Demand** | **100% Real Walmart M5 Benchmark** | 9,147 daily time-series across California (`CA`), Texas (`TX`), and Wisconsin (`WI`) for 365 continuous days. Verified 0 null values. |
| **Warehouse Inventory Ledger** | **Active SQLite State Machine (`inventory.db`)** | Production warehouses keep inventory proprietary. We model an active SQLite ledger enforcing safety stocks, replenishment batch sizes, lead times, and zero-clamping. |

---

## 4. Curated Demo Hero SKUs

Mathematically ranked by Coefficient of Variation ($CV = \frac{\sigma}{\mu}$) and peak surge ratios:

1. **`FOODS_3_090_TX_1` (SuperBowl Hero):** Normal 18.0 units/day surges to **177 units** on SuperBowl Sunday (Feb 6, 2011; 9.3x spike).
2. **`FOODS_2_285_TX_1` (Mega SNAP Spike):** Normal 7.3 units/day explodes to **634 units** (86.6x spike) on Texas SNAP welfare disbursement day.
3. **`FOODS_3_030_CA_1` (California Regional Surge):** Normal 7.5 units/day jumps to **195 units** (25.9x spike).
4. **`HOBBIES_1_209_TX_1` (Semantic Guardrail Proof):** Normal 2.8 units/day jumps to **96 units** on a SNAP day; AI rejects SNAP as irrelevant to hobbies and detects B2B wholesale impulse.
5. **`HOUSEHOLD_2_440_TX_1` (Boxing Day Clean-up):** Normal 2.8 units/day jumps to **65 units** (23.4x spike) on the day after Christmas.

---

## 5. Repository Structure

```
├── app.py                     # StockSentinel Interactive Digital Twin Cockpit (Streamlit + Plotly)
├── anomaly_engine.py          # Time-Series Anomaly Detection (Rolling Baselines & Hysteresis Machine)
├── reason_engine.py           # Root-Cause Intelligence (Semantic Guardrails & Streak Heuristics)
├── inventory_engine.py        # SQLite Physical Warehouse Engine (Lead Time, Clamping, Transfers)
├── agent_orchestrator.py      # Multi-Agent Coordination Protocol (Agent 1 <-> Agent 2 Handshake)
├── data_m5_daily_365.csv      # Real Walmart M5 benchmark dataset (9,147 series, 365 days)
├── top_5_demo_skus.csv        # Curated top-variance strategic archetypes with metadata
├── requirements.txt           # Production dependencies
└── README.md                  # Comprehensive System Architecture & Engineering Documentation
```

---

## 6. Quickstart & Local Setup

```bash
# 1. Clone repository
git clone https://github.com/Priyanshiag1/autonomous-supply-chain-agent.git
cd autonomous-supply-chain-agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch interactive dashboard
streamlit run app.py
```

