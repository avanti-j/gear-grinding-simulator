import math
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="3D Industrial Gear Creep-Feed Grinding Simulator",
    page_icon="⚙️",
    layout="wide"
)

# -----------------------------------------------------------------------------
# INDUSTRIAL BENCHMARKS & MATERIAL SPECIFICATIONS (₹ INR BASIS)
# -----------------------------------------------------------------------------
MATERIALS_DB = {
    "C45 Steel": {
        "condition": "Normalized / Q&T (220 HBW / ~28 HRC, IS Class 8)",
        "density_kg_m3": 7850,
        "thermal_conductivity_w_mk": 45.0,
        "specific_heat_j_kgk": 460.0,
        "u_specific_energy": 24.0,  # J/mm^3 (Empirical baseline for medium carbon steel)
        "recommended_wheel": "Vitrified Microcrystalline Al2O3 (9A 60 H 12 V)",
        "grit_size": "60 Mesh",
        "recommended_coolant": "Water-soluble Semi-Synthetic Emulsion (Flood, 5-6 bar)",
        "flow_rate_lpm": 75.0,
        "coolant_working_cost_per_liter": 18.0,  # ₹18/L diluted emulsion
        "workpiece_heat_partition_rw": 0.65,
        "burn_flux_limit_w_mm2": 32.0,
        "g_ratio": 60.0,  # Volume stock removed / volume wheel consumed
        "sparkout_strokes": 1,
        "dressing_interval_teeth": 20,
        "wheel_abrasive_cost_per_mm3": 0.025,  # ₹ per mm^3 of wheel wear
    },
    "Inconel 718": {
        "condition": "Precipitation Hardened / Solution Annealed (42-44 HRC)",
        "density_kg_m3": 8190,
        "thermal_conductivity_w_mk": 11.4,
        "specific_heat_j_kgk": 435.0,
        "u_specific_energy": 72.0,  # J/mm^3 (Severe dynamic shear strength)
        "recommended_wheel": "High-Porosity Vitrified CBN (B126 V240 P)",
        "grit_size": "B126 (120/140 Mesh)",
        "recommended_coolant": "High-Pressure Coherent Jet Synthetic Grinding Oil (16-18 bar)",
        "flow_rate_lpm": 120.0,
        "coolant_working_cost_per_liter": 380.0,  # ₹380/L neat synthetic grinding oil
        "workpiece_heat_partition_rw": 0.22,  # Hydrodynamic oil jet carries majority of heat
        "burn_flux_limit_w_mm2": 18.0,  # Low threshold to prevent untempered white layer
        "g_ratio": 380.0,
        "sparkout_strokes": 2,
        "dressing_interval_teeth": 45,
        "wheel_abrasive_cost_per_mm3": 0.180,  # ₹ per mm^3 of CBN rim consumed
    }
}

MACHINE_DATA = {
    "category": "CNC Gear Profile Grinding Machine",
    "verified_model": "Kapp Niles KNG 350 / Klingelnberg VIPER 500",
    "spindle_power_kw": 24.0,
    "idle_power_kw": 1.8,
    "spindle_efficiency": 0.85,
    "machine_hourly_rate_inr": 3200.0,  # ₹3,200/hr CNC shop rate (India standard)
    "power_tariff_per_kwh_inr": 11.50,  # ₹11.50/kWh Industrial HT tariff
}

# -----------------------------------------------------------------------------
# GEOMETRY FUNCTIONS
# -----------------------------------------------------------------------------
def get_gear_geometry(module, teeth, alpha_deg, face_width):
    alpha_rad = math.radians(alpha_deg)
    d = module * teeth
    da = d + 2.0 * module
    df = d - 2.5 * module
    db = d * math.cos(alpha_rad)
    tooth_height = 2.25 * module
    circular_pitch = math.pi * module
    tooth_thickness = circular_pitch / 2.0
    return {
        "m": module, "z": teeth, "alpha_deg": alpha_deg, "alpha_rad": alpha_rad,
        "b": face_width, "d": d, "da": da, "df": df, "db": db,
        "h": tooth_height, "p": circular_pitch, "s": tooth_thickness
    }

# -----------------------------------------------------------------------------
# PROCESS, THERMAL & FINANCIAL CALCULATIONS
# -----------------------------------------------------------------------------
def run_process_analysis(geom, mat_key, doc_mm, feed_mmpm):
    mat = MATERIALS_DB[mat_key]
    vf_mmps = feed_mmpm / 60.0
    b = geom["b"]
    z = geom["z"]

    total_radial_allowance = 0.35  # mm standard grinding stock on milled tooth flanks
    passes = max(1, int(math.ceil(total_radial_allowance / doc_mm)))
    actual_doc = total_radial_allowance / passes

    # 1. Removal rate & power
    qw_mm3_s = vf_mmps * b * actual_doc
    pc_kw = (mat["u_specific_energy"] * qw_mm3_s) / 1000.0
    p_in_kw = (pc_kw / MACHINE_DATA["spindle_efficiency"]) + MACHINE_DATA["idle_power_kw"]
    power_feasible = p_in_kw <= MACHINE_DATA["spindle_power_kw"]

    # 2. Contact length & thermal burn evaluation
    wheel_diameter_mm = 220.0
    lc_mm = math.sqrt(actual_doc * wheel_diameter_mm)
    contact_area = max(1.0, b * lc_mm)
    heat_flux_total = (pc_kw * 1000.0) / contact_area
    heat_flux_workpiece = heat_flux_total * mat["workpiece_heat_partition_rw"]
    burn_ratio = heat_flux_workpiece / mat["burn_flux_limit_w_mm2"]

    if burn_ratio < 0.65:
        burn_status = "Low (Safe Operating Window)"
    elif burn_ratio <= 1.0:
        burn_status = "Moderate (Process Monitoring Advised)"
    else:
        burn_status = "Critical (High Risk of Thermal Microstructural Burn)"

    # 3. Cycle time calculation
    stroke_traverse_mm = b + 15.0
    time_per_stroke_s = stroke_traverse_mm / vf_mmps
    rapid_return_s = 1.6
    index_time_s = 1.0
    sparkout_time_s = mat["sparkout_strokes"] * (stroke_traverse_mm / (vf_mmps * 1.5))

    grind_time_per_tooth_s = (passes * (time_per_stroke_s + rapid_return_s)) + sparkout_time_s
    total_cycle_time_min = (((grind_time_per_tooth_s + index_time_s) * z) + 45.0) / 60.0

    # 4. Consumables & Fluid Loss
    fluid_loss_rate_lpm = mat["flow_rate_lpm"] * 0.0016
    fluid_consumed_liters = fluid_loss_rate_lpm * total_cycle_time_min

    # 5. Economic costing in INR (₹)
    machine_cost = (MACHINE_DATA["machine_hourly_rate_inr"] / 60.0) * total_cycle_time_min
    energy_kwh = p_in_kw * (total_cycle_time_min / 60.0)
    energy_cost = energy_kwh * MACHINE_DATA["power_tariff_per_kwh_inr"]
    coolant_cost = fluid_consumed_liters * mat["coolant_working_cost_per_liter"]

    total_stock_vol_mm3 = z * (2.0 * geom["h"] * total_radial_allowance * b)
    wheel_wear_mm3 = total_stock_vol_mm3 / mat["g_ratio"]
    wheel_cost = wheel_wear_mm3 * mat["wheel_abrasive_cost_per_mm3"]

    total_cost_inr = machine_cost + energy_cost + coolant_cost + wheel_cost

    return {
        "material": mat_key,
        "passes": passes,
        "actual_doc": actual_doc,
        "qw_mm3_s": qw_mm3_s,
        "pc_kw": pc_kw,
        "p_in_kw": p_in_kw,
        "power_feasible": power_feasible,
        "heat_flux_workpiece": heat_flux_workpiece,
        "burn_status": burn_status,
        "burn_ratio": burn_ratio,
        "cycle_time_min": total_cycle_time_min,
        "production_rate_per_hr": 60.0 / total_cycle_time_min,
        "fluid_consumed_liters": fluid_consumed_liters,
        "cost_machine": machine_cost,
        "cost_energy": energy_cost,
        "cost_coolant": coolant_cost,
        "cost_wheel": wheel_cost,
        "cost_total": total_cost_inr,
        "wheel": mat["recommended_wheel"],
        "coolant": mat["recommended_coolant"]
    }

# -----------------------------------------------------------------------------
# 3D MESH GENERATION (TOOTH SLOT + DRESSED WHEEL)
# -----------------------------------------------------------------------------
def generate_3d_grinding_scene(geom, pass_idx, total_passes):
    b = geom["b"]
    m = geom["m"]
    df = geom["df"]

    # 1. 2D profile coordinates across tooth gap
    u = np.linspace(-m * 1.5, m * 1.5, 30)
    z_ax = np.linspace(0, b, 20)
    U, Z = np.meshgrid(u, z_ax)

    # Base unground slot profile
    Y_base = (df / 2.0) + (U**2) / (m * 2.8)

    # Cut boundary after pass progression
    completed_frac = pass_idx / total_passes
    cut_offset = 0.35 * (1.0 - completed_frac)
    Y_cut = Y_base - cut_offset

    # 2. 3D Dressed Grinding Wheel Surface Section
    wheel_center_y = (df / 2.0) + cut_offset + 14.0
    wheel_radius = 28.0  # Locally scaled for visual contact clarity
    theta = np.linspace(-math.pi / 4, math.pi / 4, 30)
    w_width = np.linspace(b * 0.15, b * 0.85, 20)
    THETA, W_Z = np.meshgrid(theta, w_width)

    W_X = wheel_radius * np.sin(THETA) * 0.25
    W_Y = wheel_center_y - wheel_radius * np.cos(THETA)

    fig = go.Figure()

    # Pre-formed gear surface
    fig.add_trace(go.Surface(
        x=U, y=Y_base, z=Z,
        colorscale=[[0, 'rgb(70,130,180)'], [1, 'rgb(30,90,140)']],
        opacity=0.35,
        name='Unground Tooth Stock',
        showscale=False
    ))

    # Ground tooth slot surface after current pass
    fig.add_trace(go.Surface(
        x=U, y=Y_cut, z=Z,
        colorscale=[[0, 'rgb(240,160,40)'], [1, 'rgb(200,90,10)']],
        opacity=0.9,
        name=f'Ground Flank (Pass {pass_idx})',
        showscale=False
    ))

    # Dressed profile grinding wheel
    fig.add_trace(go.Surface(
        x=W_X, y=W_Y, z=W_Z,
        colorscale=[[0, 'rgb(180,40,40)'], [1, 'rgb(120,20,20)']],
        opacity=0.85,
        name='Form Grinding Wheel',
        showscale=False
    ))

    fig.update_layout(
        title=f"3D Creep-Feed Profile Engagement (Pass {pass_idx} of {total_passes})",
        scene=dict(
            xaxis_title="Transverse X (mm)",
            yaxis_title="Radial Y (mm)",
            zaxis_title="Axial Face Width Z (mm)",
            aspectmode='data',
            camera=dict(eye=dict(x=-1.5, y=-1.5, z=1.2))
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=580
    )
    return fig

# -----------------------------------------------------------------------------
# USER INTERFACE & CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ Gear Geometry & Process")

preset = st.sidebar.radio("Load Gear Reference:", ["Design Pinion (z=20)", "Mating Gear (z=63)", "Custom Specification"])
if preset == "Design Pinion (z=20)":
    init_m, init_z, init_b = 2.0, 20, 24.0
elif preset == "Mating Gear (z=63)":
    init_m, init_z, init_b = 2.0, 63, 24.0
else:
    init_m, init_z, init_b = 2.0, 25, 30.0

in_m = st.sidebar.number_input("Module (m) [mm]", 0.5, 10.0, float(init_m), 0.5)
in_z = st.sidebar.number_input("Number of Teeth (z)", 10, 150, int(init_z), 1)
in_b = st.sidebar.number_input("Face Width (b) [mm]", 5.0, 100.0, float(init_b), 1.0)
in_alpha = st.sidebar.number_input("Pressure Angle (α) [°]", 14.5, 25.0, 20.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.subheader("Grinding Parameters")
in_doc = st.sidebar.slider("Depth of Cut per Pass (a) [mm]", 0.02, 0.35, 0.12, 0.01)
in_feed = st.sidebar.slider("Table Feed Rate (vf) [mm/min]", 10.0, 450.0, 100.0, 10.0)

gear_geom = get_gear_geometry(in_m, in_z, in_alpha, in_b)

st.title("⚙️ 3D Industrial Gear Creep-Feed Grinding Simulator")
st.caption("Precision Gear Manufacturing • Thermal Integrity • Indian Industrial Economics (₹ INR)")

tab_3d, tab_bench, tab_machine = st.tabs([
    "🎯 3D Volumetric Grinding Simulation",
    "📊 C45 vs Inconel 718 Benchmark (₹ INR)",
    "🏭 Machine Specifications & Strategy"
])

# -----------------------------------------------------------------------------
# TAB 1: 3D SIMULATION
# -----------------------------------------------------------------------------
with tab_3d:
    col_ctrl, col_view = st.columns([1, 2])
    with col_ctrl:
        sim_mat = st.selectbox("Select Workpiece Material:", ["C45 Steel", "Inconel 718"])
        res_active = run_process_analysis(gear_geom, sim_mat, in_doc, in_feed)

        st.metric("Total Flank Stock", "0.35 mm")
        st.metric("Automated Passes", f"{res_active['passes']} pass(es)")
        st.metric("Actual Cut Depth", f"{res_active['actual_doc']:.3f} mm")
        st.metric("Cutting Power (Pc)", f"{res_active['pc_kw']:.2f} kW")
        st.metric("Thermal Risk Status", res_active["burn_status"])

        pass_choice = st.slider("Step Through Pass:", 1, res_active["passes"], 1)
        st.caption("Rotate, zoom, and pan the 3D model using your mouse.")

    with col_view:
        fig_3d = generate_3d_grinding_scene(gear_geom, pass_choice, res_active["passes"])
        st.plotly_chart(fig_3d, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 2: INDUSTRIAL COMPARISON (₹ INR)
# -----------------------------------------------------------------------------
with tab_bench:
    st.subheader("Quantitative Production & Economic Benchmark (₹ INR)")
    r_c45 = run_process_analysis(gear_geom, "C45 Steel", in_doc, in_feed)
    r_inc = run_process_analysis(gear_geom, "Inconel 718", in_doc, in_feed)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Spindle Input Power", f"{r_c45['p_in_kw']:.2f} kW", f"{r_inc['p_in_kw'] - r_c45['p_in_kw']:+.2f} kW (Inconel)", delta_color="inverse")
    with c2:
        st.metric("Cycle Time / Gear", f"{r_c45['cycle_time_min']:.1f} min", f"{r_inc['cycle_time_min'] - r_c45['cycle_time_min']:+.1f} min (Inconel)", delta_color="inverse")
    with c3:
        st.metric("Throughput (8-hr Shift)", f"{r_c45['production_rate_per_hr']*8:.1f} gears", f"{(r_inc['production_rate_per_hr'] - r_c45['production_rate_per_hr'])*8:.1f} gears", delta_color="normal")
    with c4:
        st.metric("Total Cost / Gear", f"₹{r_c45['cost_total']:.2f}", f"+₹{r_inc['cost_total'] - r_c45['cost_total']:.2f} (Inconel)", delta_color="inverse")

    st.markdown("---")

    col_table, col_chart = st.columns([3, 2])
    with col_table:
        df_comp = pd.DataFrame({
            "Industrial Parameter": [
                "Specific Grinding Energy (u)",
                "Material Removal Rate (Qw)",
                "Grinding Cutting Power (Pc)",
                "Total Spindle Input Power (Pin)",
                "Spindle Feasibility (<= 24 kW)",
                "Heat Flux into Workpiece",
                "Thermal Damage / Burn Risk",
                "Recommended Grinding Wheel",
                "Coolant Strategy",
                "Cycle Time per Gear",
                "Machine Time Cost / Gear",
                "Electrical Energy Cost / Gear",
                "Coolant Consumable Cost / Gear",
                "Wheel Abrasive Wear Cost / Gear",
                "Total Manufacturing Cost / Gear"
            ],
            "C45 Structural Steel": [
                "24.0 J/mm³",
                f"{r_c45['qw_mm3_s']:.2f} mm³/s",
                f"{r_c45['pc_kw']:.2f} kW",
                f"{r_c45['p_in_kw']:.2f} kW",
                "✅ Feasible" if r_c45['power_feasible'] else "❌ Overload",
                f"{r_c45['heat_flux_workpiece']:.2f} W/mm²",
                r_c45['burn_status'],
                r_c45['wheel'],
                r_c45['coolant'],
                f"{r_c45['cycle_time_min']:.2f} min",
                f"₹{r_c45['cost_machine']:.2f}",
                f"₹{r_c45['cost_energy']:.2f}",
                f"₹{r_c45['cost_coolant']:.2f}",
                f"₹{r_c45['cost_wheel']:.2f}",
                f"₹{r_c45['cost_total']:.2f}"
            ],
            "Inconel 718 Superalloy": [
                "72.0 J/mm³",
                f"{r_inc['qw_mm3_s']:.2f} mm³/s",
                f"{r_inc['pc_kw']:.2f} kW",
                f"{r_inc['p_in_kw']:.2f} kW",
                "✅ Feasible" if r_inc['power_feasible'] else "❌ Overload",
                f"{r_inc['heat_flux_workpiece']:.2f} W/mm²",
                r_inc['burn_status'],
                r_inc['wheel'],
                r_inc['coolant'],
                f"{r_inc['cycle_time_min']:.2f} min",
                f"₹{r_inc['cost_machine']:.2f}",
                f"₹{r_inc['cost_energy']:.2f}",
                f"₹{r_inc['cost_coolant']:.2f}",
                f"₹{r_inc['cost_wheel']:.2f}",
                f"₹{r_inc['cost_total']:.2f}"
            ]
        })
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

    with col_chart:
        fig_cost = go.Figure(data=[
            go.Bar(name='C45 Steel', x=['Machine Time', 'Energy', 'Coolant', 'Wheel Wear'],
                   y=[r_c45['cost_machine'], r_c45['cost_energy'], r_c45['cost_coolant'], r_c45['cost_wheel']],
                   marker_color='rgb(55, 126, 184)'),
            go.Bar(name='Inconel 718', x=['Machine Time', 'Energy', 'Coolant', 'Wheel Wear'],
                   y=[r_inc['cost_machine'], r_inc['cost_energy'], r_inc['cost_coolant'], r_inc['cost_wheel']],
                   marker_color='rgb(228, 26, 28)')
        ])
        fig_cost.update_layout(
            title="Manufacturing Cost Breakdown (₹ INR)",
            yaxis_title="Cost per Unit (₹)",
            barmode='group',
            template="plotly_white",
            height=460
        )
        st.plotly_chart(fig_cost, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: MACHINE & PROCESS STANDARDS
# -----------------------------------------------------------------------------
with tab_machine:
    st.subheader("Industrial Machine Tool & Cooling System Specifications")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown(f"""
        #### Verified CNC Gear Profile Grinder
        * **Category:** {MACHINE_DATA['category']}
        * **Reference Benchmark:** [{MACHINE_DATA['verified_model']}](https://www.kapp-niles.com/en/machines/gear-profile-grinding/)
        * **Continuous Spindle Power:** {MACHINE_DATA['spindle_power_kw']} kW (Direct Drive)
        * **Machine Hour Rate:** ₹{MACHINE_DATA['machine_hourly_rate_inr']:.2f} / hr
        * **Power Tariff:** ₹{MACHINE_DATA['power_tariff_per_kwh_inr']:.2f} / kWh
        * **Gear Envelope Compatibility:** Modules 0.5 to 10.0 mm; Tip Diameter $d_a$ up to 350 mm.
        """)
    with col_m2:
        st.markdown("""
        #### Coolant Selection Rationale
        * **C45 Carbon Steel:** Water-soluble semi-synthetic flood cooling at 5–6 bar provides sufficient bulk heat capacity to dissipate thermal expansion in high-volume production.
        * **Inconel 718 Superalloy:** Demands high-pressure coherent jet neat grinding oil at 16–18 bar. Low thermal conductivity ($k = 11.4\\text{ W/m}\\cdot\\text{K}$) traps heat in the shearing zone; high-pressure jets penetrate the hydrodynamic air boundary layer of the rotating wheel to prevent tensile surface residual stresses.
        """)
