import math
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="3D Industrial Creep-Feed Gear Grinding Simulator",
    page_icon="⚙️",
    layout="wide"
)

# -----------------------------------------------------------------------------
# INDIAN INDUSTRIAL BENCHMARKS & MATERIAL METRICS (₹ INR)
# -----------------------------------------------------------------------------
MATERIALS_DB = {
    "C45 Steel": {
        "condition": "Normalized / Q&T (220 HBW / ~28 HRC, IS Class 8)",
        "density_kg_m3": 7850,
        "thermal_conductivity_w_mk": 45.0,
        "specific_heat_j_kgk": 460.0,
        "u_specific_energy": 24.0,  # J/mm^3
        "recommended_wheel": "Vitrified Microcrystalline Al2O3 (9A 60 H 12 V)",
        "grit_size": "60 Mesh",
        "recommended_coolant": "Semi-Synthetic Soluble Oil (Flood, 5-6 bar)",
        "flow_rate_lpm": 75.0,
        "coolant_cost_per_liter": 18.0,  # ₹18/L working emulsion
        "workpiece_heat_partition_rw": 0.65,
        "burn_flux_limit_w_mm2": 32.0,
        "g_ratio": 60.0,
        "sparkout_strokes": 1,
        "wheel_cost_per_mm3": 0.025,  # ₹/mm^3
    },
    "Inconel 718": {
        "condition": "Precipitation Aged (42-44 HRC)",
        "density_kg_m3": 8190,
        "thermal_conductivity_w_mk": 11.4,
        "specific_heat_j_kgk": 435.0,
        "u_specific_energy": 72.0,  # J/mm^3
        "recommended_wheel": "High-Porosity Vitrified CBN (B126 V240 P)",
        "grit_size": "B126 (120/140 Mesh)",
        "recommended_coolant": "High-Pressure Coherent Jet Synthetic Oil (16-18 bar)",
        "flow_rate_lpm": 120.0,
        "coolant_cost_per_liter": 380.0,  # ₹380/L neat grinding oil
        "workpiece_heat_partition_rw": 0.22,
        "burn_flux_limit_w_mm2": 18.0,
        "g_ratio": 380.0,
        "sparkout_strokes": 2,
        "wheel_cost_per_mm3": 0.180,  # ₹/mm^3
    }
}

MACHINE_DATA = {
    "category": "CNC Gear Profile Grinding Machine",
    "verified_model": "Kapp Niles KNG 350 / Klingelnberg VIPER 500",
    "spindle_power_kw": 24.0,
    "idle_power_kw": 1.8,
    "spindle_efficiency": 0.85,
    "machine_hourly_rate_inr": 3200.0,  # ₹3,200/hr shop rate
    "power_tariff_per_kwh_inr": 11.50,   # ₹11.50/kWh Industrial HT tariff
}

# -----------------------------------------------------------------------------
# FULL 3D INVOLUTE GEAR MESH GENERATOR (All Teeth + Bore + Keyway)
# -----------------------------------------------------------------------------
def build_full_3d_gear(module, teeth, face_width, pressure_angle_deg=20.0):
    """Generates 3D polygonal mesh coordinates for the entire 360 spur gear."""
    alpha_rad = math.radians(pressure_angle_deg)
    d = module * teeth
    da = d + 2.0 * module
    df = d - 2.5 * module
    db = d * math.cos(alpha_rad)
    bore_r = max(5.0, (df / 2.0) * 0.38)
    
    ra = da / 2.0
    rf = df / 2.0
    rb = db / 2.0

    # 1. Build 2D boundary of complete gear profile (all z teeth)
    pts_per_flank = 8
    poly_x, poly_y = [], []
    pitch_angle = 2.0 * math.pi / teeth

    for i in range(teeth):
        center_angle = i * pitch_angle
        # Involute left flank
        r_vals = np.linspace(rf, ra, pts_per_flank)
        left_x, left_y = [], []
        right_x, right_y = [], []

        for r in r_vals:
            cos_a = min(1.0, max(-1.0, rb / max(rb, r)))
            a = math.acos(cos_a)
            inv = math.tan(a) - a
            th_tooth = (math.pi / (2 * teeth)) - inv
            
            # Left side
            th_l = center_angle - th_tooth
            left_x.append(r * math.cos(th_l))
            left_y.append(r * math.sin(th_l))

            # Right side
            th_r = center_angle + th_tooth
            right_x.append(r * math.cos(th_r))
            right_y.append(r * math.sin(th_r))

        # Tip arc
        tip_angles = np.linspace(center_angle - (math.pi / (2 * teeth)) + 0.05, center_angle + (math.pi / (2 * teeth)) - 0.05, 3)
        tip_x = [ra * math.cos(t) for t in tip_angles]
        tip_y = [ra * math.sin(t) for t in tip_angles]

        # Root space between teeth
        next_center = (i + 1) * pitch_angle
        root_angles = np.linspace(center_angle + (pitch_angle / 2.0) - 0.08, center_angle + (pitch_angle / 2.0) + 0.08, 3)
        root_x = [rf * math.cos(t) for t in root_angles]
        root_y = [rf * math.sin(t) for t in root_angles]

        poly_x.extend(left_x + tip_x + list(reversed(right_x)) + root_x)
        poly_y.extend(left_y + tip_y + list(reversed(right_y)) + root_y)

    poly_x = np.array(poly_x)
    poly_y = np.array(poly_y)

    # 2. Extrude along face width Z to form 3D surfaces
    n_pts = len(poly_x)
    z_front = np.zeros(n_pts)
    z_back = np.full(n_pts, face_width)

    # Rim triangles
    x_mesh = list(poly_x) + list(poly_x)
    y_mesh = list(poly_y) + list(poly_y)
    z_mesh = list(z_front) + list(z_back)

    i_tri, j_tri, k_tri = [], [], []
    for idx in range(n_pts - 1):
        # Two triangles per quad segment on outer rim
        i_tri.extend([idx, idx + n_pts])
        j_tri.extend([idx + 1, idx + 1])
        k_tri.extend([idx + n_pts, idx + n_pts + 1])

    # Connect last point to first
    i_tri.extend([n_pts - 1, (2 * n_pts) - 1])
    j_tri.extend([0, 0])
    k_tri.extend([(2 * n_pts) - 1, n_pts])

    # 3. Bore hole with keyway
    bore_pts = 32
    th_bore = np.linspace(0, 2 * math.pi, bore_pts)
    bx = bore_r * np.cos(th_bore)
    by = bore_r * np.sin(th_bore)

    # Add keyway notch at top (0 deg)
    kw_w = bore_r * 0.4
    kw_h = bore_r * 0.3
    bx = np.where((bx > -kw_w/2) & (bx < kw_w/2) & (by > 0), bx, bx)
    by = np.where((bx > -kw_w/2) & (bx < kw_w/2) & (by > 0), by + kw_h, by)

    return {
        "x": x_mesh, "y": y_mesh, "z": z_mesh,
        "i": i_tri, "j": j_tri, "k": k_tri,
        "poly_x": poly_x, "poly_y": poly_y,
        "da": da, "df": df, "d": d, "b": face_width,
        "bx": bx, "by": by, "bore_r": bore_r
    }

# -----------------------------------------------------------------------------
# 3D DRESSED PROFILE WHEEL GENERATOR
# -----------------------------------------------------------------------------
def build_3d_grinding_wheel(gear_data, wheel_dia=160.0, current_z=12.0, plunge_depth=0.25):
    """Creates 3D geometry of the conjugate form grinding wheel positioned in the top tooth slot."""
    rw = wheel_dia / 2.0
    rf = gear_data["df"] / 2.0
    
    # Wheel center position: directly above top tooth gap
    y_center = rf + rw - plunge_depth
    x_center = 0.0

    # Wheel cross-section (double conical profile)
    wheel_rim_thick = gear_data["b"] * 0.55
    v_angles = np.linspace(0, 2 * math.pi, 36)
    
    # Outer contact rim profile
    wx_rim = []
    wy_rim = []
    wz_rim = []

    # 3D disk cylinder representing wheel
    u_rad = np.linspace(rw * 0.55, rw, 15)
    TH, R = np.meshgrid(v_angles, u_rad)

    # Wheel is positioned with its spindle axis parallel to X
    W_X = R * np.cos(TH) * 0.18  # Wheel profile width
    W_Y = y_center + R * np.cos(TH)
    W_Z = current_z + R * np.sin(TH) * 0.5  # Angled perspective

    return W_X, W_Y, W_Z

# -----------------------------------------------------------------------------
# PROCESS CALCULATIONS
# -----------------------------------------------------------------------------
def calculate_process_performance(gear_geom, mat_name, doc_mm, feed_mmpm):
    mat = MATERIALS_DB[mat_name]
    vf_mmps = feed_mmpm / 60.0
    b = gear_geom["b"]
    z = gear_geom["z"]

    total_radial_allowance = 0.35  # mm
    passes = max(1, int(math.ceil(total_radial_allowance / doc_mm)))
    actual_doc = total_radial_allowance / passes

    qw_mm3_s = vf_mmps * b * actual_doc
    pc_kw = (mat["u_specific_energy"] * qw_mm3_s) / 1000.0
    p_in_kw = (pc_kw / MACHINE_DATA["spindle_efficiency"]) + MACHINE_DATA["idle_power_kw"]

    lc_mm = math.sqrt(actual_doc * 200.0)
    contact_area = max(1.0, b * lc_mm)
    flux_work = ((pc_kw * 1000.0) / contact_area) * mat["workpiece_heat_partition_rw"]
    burn_ratio = flux_work / mat["burn_flux_limit_w_mm2"]

    burn_status = "Low (Safe Zone)" if burn_ratio < 0.65 else ("Moderate (Monitor)" if burn_ratio <= 1.0 else "Critical (Burn Risk)")

    stroke_len = b + 15.0
    stroke_t = stroke_len / vf_mmps
    sparkout_t = mat["sparkout_strokes"] * (stroke_len / (vf_mmps * 1.5))
    slot_t = (passes * (stroke_t + 1.6)) + sparkout_t
    cycle_min = (((slot_t + 1.0) * z) + 45.0) / 60.0

    fluid_consumed_l = mat["flow_rate_lpm"] * 0.0016 * cycle_min
    cost_mach = (MACHINE_DATA["machine_hourly_rate_inr"] / 60.0) * cycle_min
    cost_energy = (p_in_kw * (cycle_min / 60.0)) * MACHINE_DATA["power_tariff_per_kwh_inr"]
    cost_cool = fluid_consumed_l * mat["coolant_cost_per_liter"]
    
    vol_stock = z * (2.0 * (2.25 * gear_geom["m"]) * total_radial_allowance * b)
    cost_wheel = (vol_stock / mat["g_ratio"]) * mat["wheel_cost_per_mm3"]
    total_cost = cost_mach + cost_energy + cost_cool + cost_wheel

    return {
        "passes": passes, "actual_doc": actual_doc, "qw_mm3_s": qw_mm3_s,
        "pc_kw": pc_kw, "p_in_kw": p_in_kw, "burn_status": burn_status,
        "cycle_min": cycle_min, "throughput_shift": (60.0 / cycle_min) * 8.0,
        "cost_mach": cost_mach, "cost_energy": cost_energy, "cost_cool": cost_cool,
        "cost_wheel": cost_wheel, "cost_total": total_cost,
        "wheel": mat["recommended_wheel"], "coolant": mat["recommended_coolant"]
    }

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ Gear & Machine Controls")
gear_choice = st.sidebar.radio("Select Component:", ["Pinion (z=20, d=40mm)", "Mating Gear (z=63, d=126mm)", "Custom Gear"])

if gear_choice == "Pinion (z=20, d=40mm)":
    in_m, in_z, in_b = 2.0, 20, 24.0
elif gear_choice == "Mating Gear (z=63, d=126mm)":
    in_m, in_z, in_b = 2.0, 63, 24.0
else:
    in_m = st.sidebar.number_input("Module (m) [mm]", 0.5, 10.0, 2.0, 0.5)
    in_z = st.sidebar.number_input("Teeth (z)", 10, 100, 20, 1)
    in_b = st.sidebar.number_input("Face Width (b) [mm]", 5.0, 100.0, 24.0, 1.0)

st.sidebar.markdown("---")
st.sidebar.subheader("Cutting Parameters")
in_doc = st.sidebar.slider("Depth of Cut per Pass (a) [mm]", 0.02, 0.35, 0.12, 0.01)
in_feed = st.sidebar.slider("Table Feed Rate (vf) [mm/min]", 10.0, 450.0, 110.0, 10.0)

# Build gear geometry
gear_geom_dict = {"m": in_m, "z": in_z, "b": in_b}
gear_3d = build_full_3d_gear(in_m, in_z, in_b)

# -----------------------------------------------------------------------------
# MAIN APP TABS
# -----------------------------------------------------------------------------
st.title("⚙️ 3D Industrial Spur Gear Creep-Feed Grinding Simulator")
st.caption("Realistic 360° Gear Tooth Geometry • Dressed Profile Grinding Wheel Kinematics • Indian Industrial Standards (₹ INR)")

tab_sim, tab_compare, tab_machine = st.tabs([
    "🎯 3D Gear Grinding Simulation",
    "📊 C45 vs Inconel 718 Benchmark (₹ INR)",
    "🏭 Machine Tool & Fluid Specifications"
])

# -----------------------------------------------------------------------------
# TAB 1: 3D SIMULATION (Realistic Gear + Form Wheel)
# -----------------------------------------------------------------------------
with tab_sim:
    col_c, col_v = st.columns([1, 2])
    with col_c:
        st.subheader("Grinding Wheel Traverse")
        sim_mat = st.selectbox("Workpiece Material:", ["C45 Steel", "Inconel 718"])
        res_active = calculate_process_performance(gear_geom_dict, sim_mat, in_doc, in_feed)

        st.metric("Total Stock Allowance", "0.35 mm")
        st.metric("Required Passes", f"{res_active['passes']} pass(es)")
        st.metric("Grinding Cutting Power", f"{res_active['pc_kw']:.2f} kW")
        st.metric("Thermal Integrity Risk", res_active["burn_status"])

        wheel_pos_z = st.slider("Wheel Axial Feed Traverse (Z-axis) [mm]:", 0.0, float(in_b), float(in_b / 2.0), 1.0)
        curr_pass = st.slider("Active Grinding Pass:", 1, res_active["passes"], 1)

        st.info(f"Wheel currently positioned at **Z = {wheel_pos_z:.1f} mm** engaging tooth slot at **{in_feed} mm/min**.")

    with col_v:
        # Generate 3D visualization
        fig = go.Figure()

        # 1. Full 3D Gear Body (Mesh3d)
        gear_color = "rgb(175, 185, 195)" if sim_mat == "C45 Steel" else "rgb(215, 195, 155)"
        fig.add_trace(go.Mesh3d(
            x=gear_3d["x"], y=gear_3d["y"], z=gear_3d["z"],
            i=gear_3d["i"], j=gear_3d["j"], k=gear_3d["k"],
            color=gear_color,
            opacity=0.95,
            name=f"Spur Gear ({sim_mat})",
            flatshading=True,
            lighting=dict(ambient=0.4, diffuse=0.7, specular=0.5, roughness=0.3)
        ))

        # 2. Add Center Bore & Keyway (Front & Back Faces)
        for z_plane in [0.0, in_b]:
            fig.add_trace(go.Scatter3d(
                x=gear_3d["bx"], y=gear_3d["by"], z=np.full_like(gear_3d["bx"], z_plane),
                mode='lines',
                line=dict(color='black', width=4),
                name='Bore & Keyway',
                showlegend=False
            ))

        # 3. 3D Profile Grinding Wheel
        WX, WY, WZ = build_3d_grinding_wheel(gear_3d, wheel_dia=140.0, current_z=wheel_pos_z, plunge_depth=in_doc * curr_pass)
        wheel_color = "rgb(190, 70, 70)" if sim_mat == "C45 Steel" else "rgb(70, 70, 70)"  # Red for Al2O3, Black/Charcoal for CBN
        
        fig.add_trace(go.Surface(
            x=WX, y=WY, z=WZ,
            colorscale=[[0, wheel_color], [1, wheel_color]],
            opacity=0.9,
            showscale=False,
            name="Profile Grinding Wheel"
        ))

        # 4. Coolant Jet Indicator
        fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[gear_3d["df"]/2.0 + 8.0, gear_3d["df"]/2.0], z=[wheel_pos_z - 5.0, wheel_pos_z],
            mode='lines+markers',
            line=dict(color='rgb(30, 144, 255)', width=8),
            marker=dict(size=4, color='rgb(0, 191, 255)'),
            name='Targeted High-Pressure Coolant Jet'
        ))

        fig.update_layout(
            title=f"3D Creep-Feed Profile Grinding Engagement (Module {in_m}, z={in_z})",
            scene=dict(
                xaxis_title="X (Transverse)",
                yaxis_title="Y (Radial)",
                zaxis_title="Z (Face Width [mm])",
                aspectmode='data',
                camera=dict(eye=dict(x=-1.6, y=-1.6, z=1.3))
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 2: INDUSTRIAL BENCHMARK (₹ INR)
# -----------------------------------------------------------------------------
with tab_compare:
    st.subheader("Process Planning & Manufacturing Cost Comparison (₹ INR)")
    r_c45 = calculate_process_performance(gear_geom_dict, "C45 Steel", in_doc, in_feed)
    r_inc = calculate_process_performance(gear_geom_dict, "Inconel 718", in_doc, in_feed)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Spindle Power", f"{r_c45['p_in_kw']:.2f} kW", f"{r_inc['p_in_kw'] - r_c45['p_in_kw']:+.2f} kW (Inconel)", delta_color="inverse")
    with m2:
        st.metric("Cycle Time", f"{r_c45['cycle_min']:.1f} min", f"{r_inc['cycle_min'] - r_c45['cycle_min']:+.1f} min (Inconel)", delta_color="inverse")
    with m3:
        st.metric("Throughput (8-hr Shift)", f"{r_c45['throughput_shift']:.1f} gears", f"{r_inc['throughput_shift'] - r_c45['throughput_shift']:+.1f} gears", delta_color="normal")
    with m4:
        st.metric("Cost / Gear", f"₹{r_c45['cost_total']:.2f}", f"+₹{r_inc['cost_total'] - r_c45['cost_total']:.2f} (Inconel)", delta_color="inverse")

    st.markdown("---")

    col_tb, col_gr = st.columns([3, 2])
    with col_tb:
        df_bench = pd.DataFrame({
            "Specification / Metric": [
                "Specific Grinding Energy (u)",
                "Cutting Power (Pc)",
                "Spindle Input Power (Pin)",
                "Thermal Damage Risk",
                "Recommended Wheel Spec",
                "Coolant Delivery Strategy",
                "Cycle Time per Gear",
                "Machine Time Cost / Gear",
                "Electrical Energy Cost / Gear",
                "Coolant Fluid Cost / Gear",
                "Wheel Abrasive Wear Cost / Gear",
                "Total Manufacturing Cost / Gear"
            ],
            "C45 Carbon Steel (Reference)": [
                "24.0 J/mm³",
                f"{r_c45['pc_kw']:.2f} kW",
                f"{r_c45['p_in_kw']:.2f} kW",
                r_c45["burn_status"],
                r_c45["wheel"],
                r_c45["coolant"],
                f"{r_c45['cycle_min']:.2f} min",
                f"₹{r_c45['cost_mach']:.2f}",
                f"₹{r_c45['cost_energy']:.2f}",
                f"₹{r_c45['cost_cool']:.2f}",
                f"₹{r_c45['cost_wheel']:.2f}",
                f"₹{r_c45['cost_total']:.2f}"
            ],
            "Inconel 718 (Superalloy)": [
                "72.0 J/mm³",
                f"{r_inc['pc_kw']:.2f} kW",
                f"{r_inc['p_in_kw']:.2f} kW",
                r_inc["burn_status"],
                r_inc["wheel"],
                r_inc["coolant"],
                f"{r_inc['cycle_min']:.2f} min",
                f"₹{r_inc['cost_mach']:.2f}",
                f"₹{r_inc['cost_energy']:.2f}",
                f"₹{r_inc['cost_cool']:.2f}",
                f"₹{r_inc['cost_wheel']:.2f}",
                f"₹{r_inc['cost_total']:.2f}"
            ]
        })
        st.dataframe(df_bench, use_container_width=True, hide_index=True)

    with col_gr:
        fig_bar = go.Figure(data=[
            go.Bar(name='C45 Steel', x=['Machine Time', 'Energy', 'Coolant', 'Wheel Wear'],
                   y=[r_c45['cost_mach'], r_c45['cost_energy'], r_c45['cost_cool'], r_c45['cost_wheel']],
                   marker_color='rgb(55, 126, 184)'),
            go.Bar(name='Inconel 718', x=['Machine Time', 'Energy', 'Coolant', 'Wheel Wear'],
                   y=[r_inc['cost_mach'], r_inc['cost_energy'], r_inc['cost_cool'], r_inc['cost_wheel']],
                   marker_color='rgb(228, 26, 28)')
        ])
        fig_bar.update_layout(title="Cost Breakdown per Unit (₹ INR)", yaxis_title="Cost (₹)", barmode='group', template='plotly_white')
        st.plotly_chart(fig_bar, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: MACHINE TOOL & COOLANT
# -----------------------------------------------------------------------------
with tab_machine:
    st.subheader("Machine Tool & Process Strategy")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        #### CNC Gear Profile Grinder Specifications
        * **Machine Benchmark:** [{MACHINE_DATA['verified_model']}](https://www.kapp-niles.com/en/machines/gear-profile-grinding/)
        * **Spindle Power Rating:** {MACHINE_DATA['spindle_power_kw']} kW Continuous Direct Drive
        * **Machine Hour Rate:** ₹{MACHINE_DATA['machine_hourly_rate_inr']:.2f} / hr (Industrial standard for high-precision CNC gear grinding)
        * **Electricity HT Tariff:** ₹{MACHINE_DATA['power_tariff_per_kwh_inr']:.2f} / kWh
        * **Workpiece Envelope:** Modules 0.5–10 mm, Tip Diameter up to 350 mm.
        """)
    with col2:
        st.markdown("""
        #### Fluid & Thermal Management Rationale
        * **C45 Steel:** Flood emulsion at 5–6 bar effectively quenches bulk temperature and clears chips.
        * **Inconel 718:** Low thermal conductivity ($k = 11.4\\text{ W/m}\\cdot\\text{K}$) causes thermal concentration. Coherent jet synthetic oil at 16–18 bar breaks the boundary air layer of the spinning wheel to prevent white layer microcracking and tensile residual stress.
        """)
