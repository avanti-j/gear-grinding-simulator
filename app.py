import math
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(
    page_title="2D Creep-Feed Gear Grinding Simulator",
    page_icon="⚙️",
    layout="wide"
)

MATERIALS_DB = {
    "C45 Steel": {
        "specific_energy": 24.0,
        "wheel": "Vitrified Al2O3 (9A 60 H 12 V)",
        "coolant": "Water-soluble Emulsion (Flood, 5 bar)",
        "heat_partition": 0.65,
        "burn_limit": 32.0,
        "g_ratio": 60.0,
        "rate_per_liter": 0.08,
    },
    "Inconel 718": {
        "specific_energy": 72.0,
        "wheel": "Vitrified CBN (B126 V240 P)",
        "coolant": "High-Pressure Jet Oil (16 bar)",
        "heat_partition": 0.22,
        "burn_limit": 18.0,
        "g_ratio": 350.0,
        "rate_per_liter": 0.28,
    },
}

MACHINE_SPECS = {
    "model": "Kapp Niles KNG 350 / Klingelnberg VIPER 500",
    "spindle_max_kw": 24.0,
    "idle_power_kw": 1.8,
    "efficiency": 0.85,
    "hourly_rate": 125.0
}

def calculate_gear_geometry(module, teeth, pressure_angle_deg, face_width):
    alpha_rad = math.radians(pressure_angle_deg)
    d = module * teeth
    da = d + 2.0 * module
    df = d - 2.5 * module
    db = d * math.cos(alpha_rad)
    return {
        "m": module,
        "z": teeth,
        "b": face_width,
        "d": d,
        "da": da,
        "df": df,
        "db": db,
        "h": 2.25 * module,
        "alpha_rad": alpha_rad
    }

def calculate_grinding(geom, mat_name, doc, vf_mmpm):
    mat = MATERIALS_DB[mat_name]
    vf_mmps = vf_mmpm / 60.0
    total_stock = 0.35
    num_passes = max(1, int(math.ceil(total_stock / doc)))
    actual_doc = total_stock / num_passes

    qw = vf_mmps * geom["b"] * actual_doc
    pc = (mat["specific_energy"] * qw) / 1000.0
    p_in = (pc / MACHINE_SPECS["efficiency"]) + MACHINE_SPECS["idle_power_kw"]

    lc = math.sqrt(actual_doc * 220.0)
    contact_area = max(1.0, geom["b"] * lc)
    flux_total = (pc * 1000.0) / contact_area
    flux_workpiece = flux_total * mat["heat_partition"]
    burn_ratio = flux_workpiece / mat["burn_limit"]
    burn_status = "Low (Safe Zone)" if burn_ratio < 0.70 else "High (Thermal Burn Risk)"

    stroke_time = (geom["b"] + 15.0) / vf_mmps
    sparkout_passes = 1 if mat_name == "C45 Steel" else 2
    sparkout_time = sparkout_passes * (stroke_time / 1.5)
    grind_per_slot = (num_passes * (stroke_time + 1.8)) + sparkout_time
    total_cycle_min = ((grind_per_slot + 1.0) * geom["z"] + 45.0) / 60.0

    machine_cost = (MACHINE_SPECS["hourly_rate"] / 60.0) * total_cycle_min
    energy_cost = (p_in * total_cycle_min / 60.0) * 0.14
    total_stock_vol = geom["z"] * (2.0 * geom["h"] * total_stock * geom["b"])
    wheel_cost = (total_stock_vol / mat["g_ratio"]) * (0.0003 if mat_name == "C45 Steel" else 0.0022)
    total_cost = machine_cost + energy_cost + wheel_cost

    return {
        "passes": num_passes,
        "actual_doc": actual_doc,
        "qw": qw,
        "pc": pc,
        "p_in": p_in,
        "burn_status": burn_status,
        "cycle_min": total_cycle_min,
        "total_cost": total_cost,
        "wheel": mat["wheel"],
        "coolant": mat["coolant"]
    }

st.sidebar.title("⚙️ Gear & Tool Settings")
in_m = st.sidebar.number_input("Gear Module (m) [mm]", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
in_z = st.sidebar.number_input("Number of Teeth (z)", min_value=10, max_value=120, value=20, step=1)
in_b = st.sidebar.number_input("Face Width (b) [mm]", min_value=5.0, max_value=100.0, value=24.0, step=1.0)
in_doc = st.sidebar.slider("Depth of Cut per Pass (mm)", min_value=0.02, max_value=0.40, value=0.15, step=0.01)
in_feed = st.sidebar.slider("Feed Rate (mm/min)", min_value=10.0, max_value=500.0, value=120.0, step=10.0)

geom = calculate_gear_geometry(in_m, in_z, 20.0, in_b)

st.title("Industrial Creep-Feed Gear Grinding Simulator")
tab_view, tab_data = st.tabs(["🎯 2D Simulation", "📊 C45 vs Inconel 718 Benchmark"])

with tab_view:
    selected_mat = st.selectbox("Material View:", ["C45 Steel", "Inconel 718"])
    res = calculate_grinding(geom, selected_mat, in_doc, in_feed)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Total Flank Stock", "0.35 mm")
        st.metric("Number of Passes", res["passes"])
        st.metric("Cutting Power (Pc)", f"{res['pc']:.2f} kW")
        st.metric("Thermal Risk", res["burn_status"])
        sim_step = st.slider("Step Through Pass:", 1, res["passes"], 1)

    with c2:
        x_pts = np.linspace(-geom["m"] * 1.6, geom["m"] * 1.6, 90)
        y_stock = (geom["df"] / 2.0) + (x_pts**2) / (geom["m"] * 3.0)
        fraction_done = sim_step / res["passes"]
        y_cut = y_stock - (0.35 * (1.0 - fraction_done))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_pts, y=y_stock, mode="lines", name="Original Tooth Slot", line=dict(color="royalblue", width=2)))
        fig.add_trace(go.Scatter(x=x_pts, y=y_cut, mode="lines", name=f"Pass {sim_step} Boundary", line=dict(color="darkorange", dash="dash")))
        fig.update_layout(title="2D Profile Stock Removal", height=420, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

with tab_data:
    res_c45 = calculate_grinding(geom, "C45 Steel", in_doc, in_feed)
    res_inc = calculate_grinding(geom, "Inconel 718", in_doc, in_feed)

    df = pd.DataFrame({
        "Metric": [
            "Cutting Power (Pc)",
            "Spindle Input Power (Pin)",
            "Estimated Cycle Time / Gear",
            "Thermal Damage Risk",
            "Recommended Grinding Wheel",
            "Coolant Strategy",
            "Total Cost / Gear"
        ],
        "C45 Steel": [
            f"{res_c45['pc']:.2f} kW",
            f"{res_c45['p_in']:.2f} kW",
            f"{res_c45['cycle_min']:.2f} min",
            res_c45['burn_status'],
            res_c45['wheel'],
            res_c45['coolant'],
            f"${res_c45['total_cost']:.2f}"
        ],
        "Inconel 718": [
            f"{res_inc['pc']:.2f} kW",
            f"{res_inc['p_in']:.2f} kW",
            f"{res_inc['cycle_min']:.2f} min",
            res_inc['burn_status'],
            res_inc['wheel'],
            res_inc['coolant'],
            f"${res_inc['total_cost']:.2f}"
        ]
    })
    st.dataframe(df, use_container_width=True, hide_index=True)
