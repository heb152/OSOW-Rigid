import streamlit as st
import numpy as np
import os
import datetime
from pathlib import Path

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="OSOW Damage Calculator",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────
MNDOT_CLASS11_SPLIT = {11: 0.3, 12: 0.4, 13: 0.3}

AXLE_COEFFICIENTS = {
    4:  {'single': 1.62, 'tandem': 0.39, 'tridem': 0.00},
    5:  {'single': 2.00, 'tandem': 0.00, 'tridem': 0.00},
    6:  {'single': 1.02, 'tandem': 0.99, 'tridem': 0.00},
    7:  {'single': 1.00, 'tandem': 0.26, 'tridem': 0.83},
    8:  {'single': 2.38, 'tandem': 0.67, 'tridem': 0.00},
    9:  {'single': 1.13, 'tandem': 1.93, 'tridem': 0.00},
    10: {'single': 1.19, 'tandem': 1.09, 'tridem': 0.89},
    11: {'single': 4.29, 'tandem': 0.26, 'tridem': 0.06},
    12: {'single': 3.52, 'tandem': 1.14, 'tridem': 0.06},
    13: {'single': 2.15, 'tandem': 2.13, 'tridem': 0.35},
}

TTC_DEFINITIONS = {
    1:  "TTC 1: Major single-trailer truck route (Type I)",
    2:  "TTC 2: Major single-trailer truck route (Type II)",
    3:  "TTC 3: Major single- and multi-trailer truck route (Type I)",
    4:  "TTC 4: Major single-trailer truck route (Type III)",
    5:  "TTC 5: Major single- and multi-trailer truck route (Type II)",
    6:  "TTC 6: Intermediate light and single-trailer truck route (I)",
    7:  "TTC 7: Major mixed truck route (Type I)",
    8:  "TTC 8: Major multi-trailer truck route (Type I)",
    9:  "TTC 9: Intermediate light and single-trailer truck route (II)",
    10: "TTC 10: Major mixed truck route (Type II)",
    11: "TTC 11: Major multi-trailer truck route (Type II)",
    12: "TTC 12: Intermediate light and single-trailer truck route (III)",
    13: "TTC 13: Major mixed truck route (Type III)",
    14: "TTC 14: Major light truck route (Type I)",
    15: "TTC 15: Major light truck route (Type II)",
    16: "TTC 16: Major light and multi-trailer truck route",
    17: "TTC 17: Major bus route",
    18: "TTC 18: ME User Define",
    0:  "User Defined",
}

BASE_MAP     = {1: "AGG (Aggregate Base)", 2: "ATPB (Asphalt-Treated Permeable Base)", 3: "CTPB (Cement-Treated Permeable Base)"}
SLAB_MAP     = {1: "12-ft (Conventional Slab)", 2: "15-ft (Widened Lane)"}
SHOULDER_MAP = {1: "Tied Shoulder (LTE=50%)", 2: "Untied Shoulder (LTE=1%)"}

# ── Load tables ───────────────────────────────────────────────
@st.cache_resource
def load_tables():
    try:
        from tables import TTC, SA, TA
        return TTC, SA, TA, True
    except ImportError:
        return None, None, None, False

TTC_DATA, SA_DATA, TA_DATA, TABLES_OK = load_tables()

# ── Model cache ───────────────────────────────────────────────
@st.cache_resource
def load_model_and_scalers(model_type, climate, base, slab, shoulder, base_path):
    from sklearn.preprocessing import MinMaxScaler
    from pickle import load as pkl_load
    from tensorflow.keras.models import load_model

    climate_folder = os.path.join(base_path, f"C{climate}")
    sx_path    = os.path.join(climate_folder, "scalers-mn",
                    f"Min_Max_scaler_X_Climate{climate}Base{base}Slab{slab}Shoulder{shoulder}.pkl")
    sfd_path   = os.path.join(climate_folder, "scalers-mn",
                    f"Min_Max_scaler_FD_Climate{climate}Base{base}Slab{slab}Shoulder{shoulder}.pkl")
    model_path = os.path.join(climate_folder, "weights-mn",
                    f"ANN_model_{model_type}_Climate{climate}Base{base}Slab{slab}Shoulder{shoulder}.h5")

    scaler_x  = pkl_load(open(sx_path,  'rb'))
    scaler_fd = pkl_load(open(sfd_path, 'rb'))
    tf_model  = load_model(model_path)

    model_index = {"SABU": 0, "TABU": 1, "TATD": 2}[model_type]
    min_val = scaler_fd.data_min_[model_index]
    max_val = scaler_fd.data_max_[model_index]
    scaler_out = MinMaxScaler((-0.75, 0.75))
    scaler_out.fit(np.array([min_val, max_val]).reshape(-1, 1))

    return tf_model, scaler_x, scaler_out

# ── Prediction helpers ────────────────────────────────────────
def predict_fd_matrix(model_type, climate, base, slab, shoulder,
                      hpcc, jt_sp, cote, mr, epcc,
                      weights_kips, ages, base_path):
    tf_model, scaler_x, scaler_out = load_model_and_scalers(
        model_type, climate, base, slab, shoulder, base_path)

    rows = []
    for age in ages:
        for w in weights_kips:
            if model_type == "SABU":
                rows.append([hpcc, jt_sp, cote, mr, epcc, w, 0, age])
            else:
                rows.append([hpcc, jt_sp, cote, mr, epcc, 0, w, age])

    batch        = np.array(rows)
    batch_scaled = scaler_x.transform(batch)
    batch_scaled = np.delete(batch_scaled, 6 if model_type == "SABU" else 5, axis=1)
    preds        = tf_model.predict(batch_scaled, verbose=0, batch_size=512).astype('float64')
    out          = scaler_out.inverse_transform(preds)
    cf           = 1e-11 * (hpcc ** 12)
    fd_trans     = np.power(((1 / out) - 1), (1 / (-0.1)))
    fd_vals      = (fd_trans / cf) * 1000
    return fd_vals.reshape(len(ages), len(weights_kips), 1)[:, :, 0]

# ── TTC percentage lookup ─────────────────────────────────────
def get_ttc_percentages(ttc_num):
    if TTC_DATA is None:
        return {tc: 0.0 for tc in range(4, 14)}
    for row in TTC_DATA[1:]:
        if row[0] == ttc_num:
            return {tc: row[i + 2] for i, tc in enumerate(range(4, 14))}
    return {tc: 0.0 for tc in range(4, 14)}

def split_mndot(pct, mode):
    if mode == "fhwa":
        return pct
    converted     = dict(pct)
    class11_total = pct.get(11, 0.0)
    converted[11] = class11_total * MNDOT_CLASS11_SPLIT[11]
    converted[12] = class11_total * MNDOT_CLASS11_SPLIT[12]
    converted[13] = class11_total * MNDOT_CLASS11_SPLIT[13]
    return converted

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
.block-container { padding-top: 1rem; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: #f0f4f8; border-radius: 6px 6px 0 0;
    padding: 8px 16px; font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: #1a5276; color: white;
}
.metric-card {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 8px; padding: 12px 16px; margin: 4px 0;
}
.status-low    { background:#c8f7c5; border-left: 4px solid #2d8a2d; padding:8px; border-radius:4px; }
.status-mod    { background:#fef9c3; border-left: 4px solid #d97706; padding:8px; border-radius:4px; }
.status-high   { background:#fed7aa; border-left: 4px solid #ea580c; padding:8px; border-radius:4px; }
.status-crit   { background:#fecaca; border-left: 4px solid #dc2626; padding:8px; border-radius:4px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar navigation ────────────────────────────────────────
st.sidebar.image("https://img.shields.io/badge/OSOW-Damage%20Calculator-1a5276?style=for-the-badge", use_container_width=True)
st.sidebar.markdown("## 🛣️ OSOW Damage Calculator")
st.sidebar.markdown("*Rigid Pavement — ANN Models*")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "📋 Project Info",
    "🏗️ Structure",
    "🚦 Traffic & TTC",
    "📊 Cumulative Fatigue",
    "🚛 Heavy Vehicle",
    "📅 Year-by-Year",
])

BASE_PATH = st.sidebar.text_input("Model files base path", value=os.path.dirname(os.path.abspath(__file__)))
st.sidebar.markdown("---")
st.sidebar.markdown("**Climate:** Minneapolis, MN (Zone 33)")

# ═══════════════════════════════════════════════════════════════
# PAGE 1 — Project Info
# ═══════════════════════════════════════════════════════════════
if page == "📋 Project Info":
    st.title("📋 Project Information")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state["project_name"] = st.text_input("Project Name", st.session_state.get("project_name", "My OSOW Project"))
        st.session_state["project_id"]   = st.text_input("Project ID",   st.session_state.get("project_id",   "PRJ-001"))
        st.session_state["analyst"]      = st.text_input("Analyst Name", st.session_state.get("analyst", ""))
    with col2:
        st.session_state["location"]     = st.text_input("Location",     st.session_state.get("location", ""))
        st.session_state["proj_date"]    = st.date_input("Analysis Date", datetime.date.today())
        st.session_state["description"]  = st.text_area("Description",   st.session_state.get("description", ""), height=100)

    st.success("✅ Project info saved. Proceed to Structure tab.")

# ═══════════════════════════════════════════════════════════════
# PAGE 2 — Structure
# ═══════════════════════════════════════════════════════════════
elif page == "🏗️ Structure":
    st.title("🏗️ Pavement Structural Parameters")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state["hpcc"]     = st.slider("PCC Thickness (inches)", 6.0, 16.0, float(st.session_state.get("hpcc", 6.0)), 0.5)
        st.session_state["jt_sp"]    = st.slider("Joint Spacing (feet)",   12.0, 20.0, float(st.session_state.get("jt_sp", 12.0)), 1.0)
        st.session_state["cote"]     = st.slider("COTE (µε/°C)",           4.0,  6.0,  float(st.session_state.get("cote", 5.0)),  0.1)
        st.session_state["mr"]       = st.slider("Modulus of Rupture (psi)", 500, 900, int(st.session_state.get("mr", 500)), 10)
        st.session_state["epcc"]     = st.slider("Elastic Modulus (M psi)", 3.0, 6.0, float(st.session_state.get("epcc", 4.2)), 0.1)
    with col2:
        base_options = list(BASE_MAP.values())
        base_idx = int(st.session_state.get("base", 1)) - 1
        base_sel = st.selectbox("Base Type", base_options, index=base_idx)
        st.session_state["base"] = base_options.index(base_sel) + 1

        slab_options = list(SLAB_MAP.values())
        slab_idx = int(st.session_state.get("slab", 1)) - 1
        slab_sel = st.selectbox("Slab Width", slab_options, index=slab_idx)
        st.session_state["slab"] = slab_options.index(slab_sel) + 1

        sh_options = list(SHOULDER_MAP.values())
        sh_idx = int(st.session_state.get("shoulder", 1)) - 1
        sh_sel = st.selectbox("Shoulder Type", sh_options, index=sh_idx)
        st.session_state["shoulder"] = sh_options.index(sh_sel) + 1

        st.markdown("### Current Values")
        st.markdown(f"""
        | Parameter | Value |
        |-----------|-------|
        | PCC Thickness | **{st.session_state['hpcc']} in** |
        | Joint Spacing | **{st.session_state['jt_sp']} ft** |
        | COTE | **{st.session_state['cote']} µε/°C** |
        | Modulus of Rupture | **{st.session_state['mr']} psi** |
        | Elastic Modulus | **{st.session_state['epcc']} M psi** |
        | Base | **{base_sel}** |
        | Slab | **{slab_sel}** |
        | Shoulder | **{sh_sel}** |
        """)

# ═══════════════════════════════════════════════════════════════
# PAGE 3 — Traffic & TTC
# ═══════════════════════════════════════════════════════════════
elif page == "🚦 Traffic & TTC":
    st.title("🚦 Traffic & TTC Configuration")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Basic Parameters")
        st.session_state["design_period"] = st.number_input("Design Period (years)", 1, 50, int(st.session_state.get("design_period", 20)))
        st.session_state["growth_rate"]   = st.number_input("Annual Growth Rate (%)", 0.0, 20.0, float(st.session_state.get("growth_rate", 0.0)), 0.5)

        pav_cond = st.radio("Pavement Condition", ["New Pavement", "Existing Pavement"],
                            index=0 if st.session_state.get("is_new", True) else 1)
        st.session_state["is_new"] = (pav_cond == "New Pavement")
        if not st.session_state["is_new"]:
            st.session_state["pavement_age"] = st.number_input("Age at Start (years)", 1, 50, int(st.session_state.get("pavement_age", 5)))
        else:
            st.session_state["pavement_age"] = 0

        pa = st.session_state["pavement_age"]
        dp = st.session_state["design_period"]
        st.info(f"Age during analysis: **{pa+1}** → **{pa+dp}** years")

        st.subheader("Traffic Volume")
        input_mode = st.radio("Input Mode", ["Direct (trucks/year)", "AADTT-Based"])
        if input_mode == "Direct (trucks/year)":
            st.session_state["total_trucks"] = st.number_input("Total Trucks/Year", 1, 9999999, int(st.session_state.get("total_trucks", 360000)), 30000)
        else:
            aadtt  = st.number_input("Two-way AADTT Year 1", 1, 99999, 1000, 500)
            lanes  = st.selectbox("Number of Lanes", ["2 lanes → LDF=1.00", "4 lanes → LDF=0.90", "6 lanes → LDF=0.60", "8+ lanes → LDF=0.45"])
            ldf    = {"2 lanes → LDF=1.00": 1.00, "4 lanes → LDF=0.90": 0.90, "6 lanes → LDF=0.60": 0.60, "8+ lanes → LDF=0.45": 0.45}[lanes]
            computed = int(round(aadtt * ldf * 0.5 * 360))
            st.session_state["total_trucks"] = computed
            st.info(f"Computed: **{computed:,} trucks/year**")

        st.subheader("Classification Mode")
        cls_mode = st.radio("Mode", ["MnDOT (11+12+13 → 11)", "FHWA (Class 4–13 separate)"],
                            index=0 if st.session_state.get("cls_mode", "mndot") == "mndot" else 1)
        st.session_state["cls_mode"] = "mndot" if "MnDOT" in cls_mode else "fhwa"

    with col2:
        st.subheader("TTC Selection")
        ttc_options = {0: "User Defined"} | {i: TTC_DEFINITIONS[i] for i in range(1, 19)}
        ttc_labels  = [f"{k}: {v}" for k, v in ttc_options.items()]
        sel_idx     = list(ttc_options.keys()).index(st.session_state.get("selected_ttc", 0))
        ttc_sel     = st.selectbox("TTC Group", ttc_labels, index=sel_idx)
        ttc_num     = int(ttc_sel.split(":")[0])
        st.session_state["selected_ttc"] = ttc_num

        st.subheader("Class Distribution (%)")
        mode = st.session_state["cls_mode"]

        if ttc_num == 0:
            st.info("Enter custom percentages. Must sum to 100%.")
            pcts = {}
            if mode == "mndot":
                classes_to_show = list(range(4, 12))
            else:
                classes_to_show = list(range(4, 14))

            cols = st.columns(2)
            for i, tc in enumerate(classes_to_show):
                default = float(st.session_state.get(f"pct_{tc}", 0.0))
                with cols[i % 2]:
                    pcts[tc] = st.number_input(f"Class {tc} %", 0.0, 100.0, default, 0.1, key=f"pct_input_{tc}")
                st.session_state[f"pct_{tc}"] = pcts[tc]

            if mode == "mndot":
                pcts[12] = 0.0
                pcts[13] = 0.0

            total_pct = sum(pcts.values())
            color = "green" if abs(total_pct - 100.0) < 0.01 else "red"
            st.markdown(f"**Total: :{color}[{total_pct:.2f}%]**")
            st.session_state["truck_class_pct_raw"] = pcts
        else:
            raw_pcts = get_ttc_percentages(ttc_num)

            if mode == "mndot":
                merged = raw_pcts[11] + raw_pcts[12] + raw_pcts[13]
                display_pcts = {tc: raw_pcts[tc] for tc in range(4, 11)}
                display_pcts[11] = merged
            else:
                display_pcts = raw_pcts

            import pandas as pd
            df = pd.DataFrame([
                {"Class": f"Class {tc}", "Percentage (%)": f"{v:.2f}"}
                for tc, v in display_pcts.items()
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)
            total = sum(display_pcts.values())
            st.success(f"✅ Total: {total:.2f}%")
            st.session_state["truck_class_pct_raw"] = raw_pcts

    # ── Calculate Load Distribution ───────────────────────────
    st.markdown("---")
    if st.button("📊 Calculate Load Distribution & Traffic Results", type="primary", use_container_width=True):
        if not TABLES_OK:
            st.error("tables.py not found! Cannot calculate.")
        else:
            with st.spinner("Calculating load distribution..."):
                raw_pcts = st.session_state.get("truck_class_pct_raw", get_ttc_percentages(ttc_num))
                truck_class_pct = split_mndot(raw_pcts, mode)
                total_trucks    = st.session_state["total_trucks"]

                results      = []
                total_single = total_tandem = total_tridem = 0
                for tc in range(4, 14):
                    ttc_pct    = truck_class_pct[tc]
                    num_trucks = total_trucks * (ttc_pct / 100.0)
                    coeffs     = AXLE_COEFFICIENTS[tc]
                    sa = int(round(num_trucks * coeffs['single']))
                    ta = int(round(num_trucks * coeffs['tandem']))
                    tr = int(round(num_trucks * coeffs['tridem']))
                    results.append({'class': tc, 'ttc_pct': ttc_pct, 'num_trucks': num_trucks,
                                    'single_axles': sa, 'tandem_axles': ta, 'tridem_axles': tr,
                                    'single_coeff': coeffs['single'], 'tandem_coeff': coeffs['tandem'],
                                    'tridem_coeff': coeffs['tridem']})
                    total_single += sa; total_tandem += ta; total_tridem += tr

                st.session_state["distribution_results"] = results
                st.session_state["distribution_totals"]  = {
                    'total_trucks': total_trucks,
                    'total_single': total_single,
                    'total_tandem': total_tandem,
                    'total_tridem': total_tridem,
                }

                # Traffic results (SA & TA)
                def compute_traffic(table_data, axle_col):
                    rows_data = table_data[1:]
                    load_pct  = {}
                    for row in rows_data:
                        ml = row[0]
                        load_pct[ml] = {tc: row[i+1] for i, tc in enumerate(range(4,14))}
                    mean_loads   = sorted(load_pct.keys())
                    axle_counts  = {r['class']: r[axle_col] for r in results}
                    out_results  = []
                    class_totals = {tc: 0.0 for tc in range(4, 14)}
                    for ml in mean_loads:
                        row_data  = {'mean_load': ml}
                        row_total = 0.0
                        for tc in range(4, 14):
                            passes = axle_counts.get(tc, 0.0) * (load_pct[ml].get(tc, 0.0) / 100.0)
                            row_data[f'class_{tc}'] = passes
                            row_total += passes
                            class_totals[tc] += passes
                        row_data['total'] = row_total
                        out_results.append(row_data)
                    grand = sum(class_totals.values())
                    return out_results, class_totals, grand

                sa_res, sa_ct, sa_grand = compute_traffic(SA_DATA, 'single_axles')
                ta_res, ta_ct, ta_grand = compute_traffic(TA_DATA, 'tandem_axles')
                st.session_state["sa_traffic_results"] = {'data': sa_res, 'class_totals': sa_ct, 'grand_total': sa_grand}
                st.session_state["ta_traffic_results"] = {'data': ta_res, 'class_totals': ta_ct, 'grand_total': ta_grand}

            st.success(f"✅ Done!  Single Axles: **{total_single:,}**  |  Tandem Axles: **{total_tandem:,}**  |  SA passes: **{sa_grand:,.0f}**  |  TA passes: **{ta_grand:,.0f}**")

    # Show distribution table if available
    if "distribution_results" in st.session_state:
        import pandas as pd
        st.markdown("### Load Distribution")
        df = pd.DataFrame(st.session_state["distribution_results"])
        df = df[['class','ttc_pct','num_trucks','single_axles','tandem_axles','tridem_axles']]
        df.columns = ['Class','TTC%','Num Trucks','Single Axles','Tandem Axles','Tridem Axles']
        df['Num Trucks']    = df['Num Trucks'].apply(lambda x: f"{int(x):,}")
        df['Single Axles']  = df['Single Axles'].apply(lambda x: f"{x:,}")
        df['Tandem Axles']  = df['Tandem Axles'].apply(lambda x: f"{x:,}")
        df['Tridem Axles']  = df['Tridem Axles'].apply(lambda x: f"{x:,}")
        df['TTC%']          = df['TTC%'].apply(lambda x: f"{x:.2f}")
        st.dataframe(df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 4 — Cumulative Fatigue
# ═══════════════════════════════════════════════════════════════
elif page == "📊 Cumulative Fatigue":
    st.title("📊 Cumulative Fatigue Analysis")

    if "sa_traffic_results" not in st.session_state:
        st.warning("⚠️ Please go to **Traffic & TTC** tab and calculate load distribution first.")
        st.stop()

    # Parameters summary
    with st.expander("📋 Current Parameters", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**PCC Thickness:** {st.session_state.get('hpcc',6)} in")
            st.write(f"**Joint Spacing:** {st.session_state.get('jt_sp',12)} ft")
            st.write(f"**COTE:** {st.session_state.get('cote',5)} µε/°C")
        with col2:
            st.write(f"**MR:** {st.session_state.get('mr',500)} psi")
            st.write(f"**Epcc:** {st.session_state.get('epcc',4.2)} M psi")
            st.write(f"**Base:** {BASE_MAP.get(st.session_state.get('base',1))}")
        with col3:
            st.write(f"**Design Period:** {st.session_state.get('design_period',20)} yr")
            st.write(f"**Growth Rate:** {st.session_state.get('growth_rate',0.0):.1f}%")
            st.write(f"**Total Trucks:** {st.session_state.get('total_trucks',360000):,}")

    if st.button("🚀 Calculate All Models (SABU + TABU + TATD)", type="primary", use_container_width=True):
        try:
            climate  = 33
            hpcc     = float(st.session_state.get("hpcc", 6))
            jt_sp    = float(st.session_state.get("jt_sp", 12))
            cote     = float(st.session_state.get("cote", 5))
            mr       = float(st.session_state.get("mr", 500))
            epcc     = float(st.session_state.get("epcc", 4.2))
            base     = int(st.session_state.get("base", 1))
            slab     = int(st.session_state.get("slab", 1))
            shoulder = int(st.session_state.get("shoulder", 1))
            n_years  = int(st.session_state.get("design_period", 20))
            r_pct    = float(st.session_state.get("growth_rate", 0.0))
            r        = r_pct / 100.0
            initial_age = int(st.session_state.get("pavement_age", 0))
            n_base   = 360_000

            gf = float(n_years) if r == 0.0 else ((1 + r) ** n_years - 1) / r

            sa_store = st.session_state["sa_traffic_results"]
            ta_store = st.session_state["ta_traffic_results"]

            WB_PCT = {'SWB': 0.17, 'MWB': 0.22, 'LWB': 0.61}

            all_loads = sorted(set(
                r_['mean_load'] for store in [sa_store['data'], ta_store['data']] for r_ in store
            ))

            passes_by_model = {
                'SABU': {r_['mean_load']: r_['total'] for r_ in sa_store['data']},
                'TABU': {r_['mean_load']: r_['total'] for r_ in ta_store['data']},
                'TATD': {r_['mean_load']: r_['total'] for r_ in ta_store['data']},
                'SWB':  {r_['mean_load']: r_['total'] * WB_PCT['SWB'] for r_ in ta_store['data']},
                'MWB':  {r_['mean_load']: r_['total'] * WB_PCT['MWB'] for r_ in ta_store['data']},
                'LWB':  {r_['mean_load']: r_['total'] * WB_PCT['LWB'] for r_ in ta_store['data']},
            }

            all_ages        = [float(initial_age + y) for y in range(0, n_years + 1)]
            sa_weights_kips = [ml / 1000.0 for ml in all_loads if passes_by_model['SABU'].get(ml, 0.0) >= 0.01]
            ta_weights_kips = [ml / 1000.0 for ml in all_loads if passes_by_model['TABU'].get(ml, 0.0) >= 0.01]

            progress = st.progress(0, text="Loading models...")

            mat = {}
            if sa_weights_kips:
                progress.progress(10, text="Predicting SABU...")
                mat['SABU'] = predict_fd_matrix("SABU", climate, base, slab, shoulder,
                                                hpcc, jt_sp, cote, mr, epcc,
                                                sa_weights_kips, all_ages, BASE_PATH)
            if ta_weights_kips:
                progress.progress(40, text="Predicting TABU...")
                mat['TABU'] = predict_fd_matrix("TABU", climate, base, slab, shoulder,
                                                hpcc, jt_sp, cote, mr, epcc,
                                                ta_weights_kips, all_ages, BASE_PATH)
                progress.progress(70, text="Predicting TATD...")
                mat['TATD'] = predict_fd_matrix("TATD", climate, base, slab, shoulder,
                                                hpcc, jt_sp, cote, mr, epcc,
                                                ta_weights_kips, all_ages, BASE_PATH)

            progress.progress(85, text="Computing fatigue damage...")

            def fd_from_mat(model_key, w_kips, age_idx):
                weights = sa_weights_kips if model_key == 'SABU' else ta_weights_kips
                try:
                    wi = weights.index(w_kips)
                except ValueError:
                    return 0.0
                return float(mat[model_key][age_idx, wi])

            def gf_at(period):
                return float(period) if r == 0.0 else ((1 + r) ** period - 1) / r

            results  = []
            total_fd = {'SABU': 0.0, 'TABU': 0.0, 'TATD': 0.0, 'SWB': 0.0, 'MWB': 0.0, 'LWB': 0.0}

            for idx, ml_lbs in enumerate(all_loads):
                ml_kips = ml_lbs / 1000.0
                n_sa    = passes_by_model['SABU'].get(ml_lbs, 0.0)
                n_ta    = passes_by_model['TABU'].get(ml_lbs, 0.0)
                n_swb   = passes_by_model['SWB'].get(ml_lbs, 0.0)
                n_mwb   = passes_by_model['MWB'].get(ml_lbs, 0.0)
                n_lwb   = passes_by_model['LWB'].get(ml_lbs, 0.0)

                model_fd  = {k: 0.0 for k in ['SABU','TABU','TATD','SWB','MWB','LWB']}
                row_valid = False

                for model_key, ann_model, n_annual in [
                    ("SABU","SABU",n_sa), ("TABU","TABU",n_ta), ("TATD","TATD",n_ta),
                    ("SWB","TATD",n_swb), ("MWB","TATD",n_mwb), ("LWB","TATD",n_lwb),
                ]:
                    if n_annual < 0.01 or ann_model not in mat:
                        continue
                    try:
                        fd_cum = 0.0
                        if initial_age > 0:
                            fd_cum += fd_from_mat(ann_model, ml_kips, 0) * (n_annual * gf_at(initial_age) / n_base)
                        for year in range(1, n_years + 1):
                            fd_n    = fd_from_mat(ann_model, ml_kips, year)
                            fd_prev = fd_from_mat(ann_model, ml_kips, year - 1)
                            gf_n    = gf_at(initial_age + year)
                            gf_prev = gf_at(initial_age + year - 1)
                            fd_cum += max((fd_n * gf_n - fd_prev * gf_prev) * n_annual / n_base, 0.0)
                        model_fd[model_key]  = fd_cum
                        total_fd[model_key] += fd_cum
                        row_valid = True
                    except Exception:
                        pass

                if row_valid:
                    row = {'load_lbs': ml_lbs, 'load_kips': ml_kips,
                           'n_sa': n_sa, 'n_ta': n_ta, 'scale': (n_sa * gf) / n_base}
                    row.update(model_fd)
                    results.append(row)

            progress.progress(100, text="Done!")
            st.session_state["cum_results"]  = results
            st.session_state["cum_total_fd"] = total_fd
            st.session_state["cum_gf"]       = gf
            st.session_state["cum_n_years"]  = n_years
            st.session_state["cum_r_pct"]    = r_pct
            st.session_state["cum_initial_age"] = initial_age
            progress.empty()

        except Exception as e:
            st.error(f"❌ Error: {e}")
            import traceback
            st.code(traceback.format_exc())

    # ── Results display ───────────────────────────────────────
    if "cum_total_fd" in st.session_state:
        total_fd    = st.session_state["cum_total_fd"]
        gf          = st.session_state["cum_gf"]
        n_years     = st.session_state["cum_n_years"]
        r_pct       = st.session_state["cum_r_pct"]
        initial_age = st.session_state["cum_initial_age"]

        st.markdown("---")
        st.subheader("Results")

        # GF info bar
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Pavement Age", f"{initial_age} yr")
        col2.metric("Design Period", f"{n_years} yr")
        col3.metric("Growth Rate",   f"{r_pct:.1f}%")
        col4.metric("Growth Factor", f"{gf:.4f}")

        # Model cards
        def status_class(fd):
            if fd < 0.01:  return "status-low",  "✅ LOW"
            elif fd < 0.1: return "status-mod",  "⚠️ MODERATE"
            elif fd < 1.0: return "status-high", "⚠️ HIGH"
            else:          return "status-crit", "🚨 CRITICAL"

        col1, col2, col3 = st.columns(3)
        for col, model in zip([col1, col2, col3], ["SABU", "TABU", "TATD"]):
            fd   = total_fd[model]
            sf   = 1.0 / fd if fd > 0 else 999
            css, label = status_class(fd)
            with col:
                st.markdown(f"""
                <div class="{css}">
                <b>{model}</b><br>
                FD = {fd:.4e}<br>
                SF = {sf:.3f}<br>
                {label}
                </div>
                """, unsafe_allow_html=True)

        # Table
        st.markdown("### Load-by-Load Fatigue Damage")
        import pandas as pd
        results = st.session_state["cum_results"]
        df = pd.DataFrame([{
            'Load (lbs)': f"{r['load_lbs']:,.0f}",
            'SA Passes':  f"{r['n_sa']:,.1f}",
            'TA Passes':  f"{r['n_ta']:,.1f}",
            'Scale×GF':   f"{r['scale']:.4f}",
            'FD SABU':    f"{r['SABU']:.4e}",
            'FD TABU':    f"{r['TABU']:.4e}",
            'FD TATD':    f"{r['TATD']:.4e}",
        } for r in results])
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Chart
        st.markdown("### Fatigue Damage Chart")
        import plotly.graph_objects as go
        loads = [r['load_lbs'] for r in results]
        fig = go.Figure()
        colors = {'SABU': 'steelblue', 'TABU': '#2d8a2d', 'TATD': '#cc8800'}
        for model in ["SABU", "TABU", "TATD"]:
            fds = [r[model] for r in results]
            fig.add_trace(go.Bar(
                name=model, x=[f"{int(l):,}" for l in loads], y=fds,
                marker_color=colors[model], opacity=0.85
            ))
        fig.update_layout(
            barmode='group', height=400,
            xaxis_title="Mean Load (lbs)", yaxis_title="Fatigue Damage",
            legend_title="Model", xaxis_tickangle=-45,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Export
        if st.button("💾 Export Results to CSV"):
            import io
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            st.download_button("⬇️ Download CSV", buf.getvalue(),
                               "cumulative_fatigue.csv", "text/csv")

# ═══════════════════════════════════════════════════════════════
# PAGE 5 — Heavy Vehicle
# ═══════════════════════════════════════════════════════════════
elif page == "🚛 Heavy Vehicle":
    st.title("🚛 Heavy Vehicle / OSOW Analysis")

    if "sa_traffic_results" not in st.session_state:
        st.warning("⚠️ Please go to **Traffic & TTC** tab and calculate load distribution first.")
        st.stop()

    st.subheader("Axle Configuration")
    col1, col2 = st.columns(2)
    with col1:
        hv_trips   = st.number_input("Annual Trips", 1, 999999, int(st.session_state.get("hv_trips", 1)), 100)
        n_axles    = st.number_input("Total Axles (2–13)", 2, 13, int(st.session_state.get("hv_n_axles", 5)))
        st.session_state["hv_trips"]   = hv_trips
        st.session_state["hv_n_axles"] = n_axles

    st.markdown("**Enter axle weights and types:**")
    axle_weights = []
    axle_types   = []

    cols_per_row = 5
    axles_per_row = [list(range(n_axles))[i:i+cols_per_row] for i in range(0, n_axles, cols_per_row)]

    for row_axles in axles_per_row:
        cols = st.columns(len(row_axles))
        for col, i in zip(cols, row_axles):
            with col:
                label = "FRONT" if i == 0 else f"Axle {i+1}"
                default_w = float(st.session_state.get(f"hv_w_{i}", 12.0 if i == 0 else 18.0))
                default_t = st.session_state.get(f"hv_t_{i}", "Single")
                w = st.number_input(f"{label} (kips)", 0.1, 100.0, default_w, 0.5, key=f"hv_w_input_{i}")
                t = st.selectbox("Type", ["Single", "Tandem"], index=0 if default_t == "Single" else 1, key=f"hv_t_input_{i}")
                st.session_state[f"hv_w_{i}"] = w
                st.session_state[f"hv_t_{i}"] = t
                axle_weights.append(w)
                axle_types.append(t)

    total_w = sum(axle_weights)
    st.info(f"Total Weight: **{total_w:.2f} kips**  |  Front: **{axle_weights[0]:.1f}**  |  Rear: **{total_w - axle_weights[0]:.1f}**")

    if st.button("🚀 Calculate Heavy Vehicle Damage", type="primary", use_container_width=True):
        try:
            climate  = 33
            hpcc     = float(st.session_state.get("hpcc", 6))
            jt_sp    = float(st.session_state.get("jt_sp", 12))
            cote     = float(st.session_state.get("cote", 5))
            mr       = float(st.session_state.get("mr", 500))
            epcc     = float(st.session_state.get("epcc", 4.2))
            base     = int(st.session_state.get("base", 1))
            slab     = int(st.session_state.get("slab", 1))
            shoulder = int(st.session_state.get("shoulder", 1))
            n_years  = int(st.session_state.get("design_period", 20))
            r_pct    = float(st.session_state.get("growth_rate", 0.0))
            r        = r_pct / 100.0
            n_base   = 360_000

            gf           = float(n_years) if r == 0.0 else ((1 + r) ** n_years - 1) / r
            n_lifetime   = hv_trips * gf
            scale_factor = n_lifetime / n_base

            front_weight  = axle_weights[0]
            rear_weights  = axle_weights[1:]
            rear_types    = axle_types[1:]

            with st.spinner("Predicting FD..."):
                results  = []
                total_fd = {'SABU': 0.0, 'TABU': 0.0, 'TATD': 0.0}

                def pred_single(w, age):
                    mat = predict_fd_matrix("SABU", climate, base, slab, shoulder,
                                            hpcc, jt_sp, cote, mr, epcc,
                                            [w], [float(age)], BASE_PATH)
                    return float(mat[0, 0])

                def pred_tandem(model, w, age):
                    mat = predict_fd_matrix(model, climate, base, slab, shoulder,
                                            hpcc, jt_sp, cote, mr, epcc,
                                            [w], [float(age)], BASE_PATH)
                    return float(mat[0, 0])

                # Front axle
                fd_front = pred_single(front_weight, n_years) * scale_factor
                total_fd['SABU'] += fd_front
                results.append({'label': 'Front', 'type': 'Single', 'weight': front_weight,
                                 'fd_sabu': fd_front, 'fd_tabu': 0.0, 'fd_tatd': 0.0})

                # Rear axles
                for i, (w, t) in enumerate(zip(rear_weights, rear_types)):
                    if t == "Single":
                        fd_s = pred_single(w, n_years) * scale_factor
                        total_fd['SABU'] += fd_s
                        results.append({'label': f'Rear {i+1}', 'type': 'Single', 'weight': w,
                                         'fd_sabu': fd_s, 'fd_tabu': 0.0, 'fd_tatd': 0.0})
                    else:
                        fd_ta = pred_tandem("TABU", w, n_years) * scale_factor
                        fd_td = pred_tandem("TATD", w, n_years) * scale_factor
                        total_fd['TABU'] += fd_ta
                        total_fd['TATD'] += fd_td
                        results.append({'label': f'Rear {i+1}', 'type': 'Tandem', 'weight': w,
                                         'fd_sabu': 0.0, 'fd_tabu': fd_ta, 'fd_tatd': fd_td})

            st.session_state["hv_results"]  = results
            st.session_state["hv_total_fd"] = total_fd
            st.session_state["hv_gf"]       = gf
            st.session_state["hv_trips_u"]  = hv_trips
            st.session_state["hv_scale"]    = scale_factor

        except Exception as e:
            st.error(f"❌ Error: {e}")
            import traceback
            st.code(traceback.format_exc())

    if "hv_total_fd" in st.session_state:
        total_fd    = st.session_state["hv_total_fd"]
        results     = st.session_state["hv_results"]
        gf          = st.session_state["hv_gf"]
        scale       = st.session_state["hv_scale"]
        trips_used  = st.session_state["hv_trips_u"]

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.metric("SABU FD", f"{total_fd['SABU']:.4e}", f"SF={1/total_fd['SABU']:.2f}" if total_fd['SABU']>0 else "SF=∞")
        col2.metric("TABU FD", f"{total_fd['TABU']:.4e}", f"SF={1/total_fd['TABU']:.2f}" if total_fd['TABU']>0 else "SF=∞")
        col3.metric("TATD FD", f"{total_fd['TATD']:.4e}", f"SF={1/total_fd['TATD']:.2f}" if total_fd['TATD']>0 else "SF=∞")

        st.markdown(f"**Scale Factor:** {scale:.4f}  |  **Lifetime Trips:** {trips_used * gf:,.0f}  |  **GF:** {gf:.4f}")

        import pandas as pd
        df = pd.DataFrame([{
            'Axle':       r['label'],
            'Type':       r['type'],
            'Weight (kips)': f"{r['weight']:.2f}",
            'FD SABU':    f"{r['fd_sabu']:.4e}" if r['fd_sabu'] > 0 else "—",
            'FD TABU':    f"{r['fd_tabu']:.4e}" if r['fd_tabu'] > 0 else "—",
            'FD TATD':    f"{r['fd_tatd']:.4e}" if r['fd_tatd'] > 0 else "—",
        } for r in results])
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Chart
        import plotly.graph_objects as go
        labels = [r['label'] for r in results]
        fig = go.Figure()
        fig.add_trace(go.Bar(name='SABU', x=labels, y=[r['fd_sabu'] for r in results], marker_color='steelblue'))
        fig.add_trace(go.Bar(name='TABU', x=labels, y=[r['fd_tabu'] for r in results], marker_color='#2d8a2d'))
        fig.add_trace(go.Bar(name='TATD', x=labels, y=[r['fd_tatd'] for r in results], marker_color='#cc8800'))
        fig.update_layout(barmode='group', height=350, xaxis_title="Axle", yaxis_title="Fatigue Damage")
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 6 — Year-by-Year
# ═══════════════════════════════════════════════════════════════
elif page == "📅 Year-by-Year":
    st.title("📅 Year-by-Year Fatigue Damage")

    if "sa_traffic_results" not in st.session_state:
        st.warning("⚠️ Please go to **Traffic & TTC** tab and calculate load distribution first.")
        st.stop()

    mode = st.radio("Calculation Mode", ["Total (TTC)", "Grand Total (TTC + HV)"])

    if st.button("🚀 Calculate Year-by-Year", type="primary", use_container_width=True):
        try:
            climate  = 33
            hpcc     = float(st.session_state.get("hpcc", 6))
            jt_sp    = float(st.session_state.get("jt_sp", 12))
            cote     = float(st.session_state.get("cote", 5))
            mr       = float(st.session_state.get("mr", 500))
            epcc     = float(st.session_state.get("epcc", 4.2))
            base     = int(st.session_state.get("base", 1))
            slab     = int(st.session_state.get("slab", 1))
            shoulder = int(st.session_state.get("shoulder", 1))
            n_years  = int(st.session_state.get("design_period", 20))
            r_pct    = float(st.session_state.get("growth_rate", 0.0))
            r        = r_pct / 100.0
            initial_age = int(st.session_state.get("pavement_age", 0))
            n_base   = 360_000

            sa_data = st.session_state["sa_traffic_results"]['data']
            ta_data = st.session_state["ta_traffic_results"]['data']

            sa_weights_kips = [row['mean_load'] / 1000.0 for row in sa_data if row['total'] >= 0.01]
            ta_weights_kips = [row['mean_load'] / 1000.0 for row in ta_data if row['total'] >= 0.01]
            all_ages        = [float(initial_age + y) for y in range(0, n_years + 1)]

            progress = st.progress(0, text="Loading models...")

            mat = {}
            if sa_weights_kips:
                progress.progress(15, text="Predicting SABU matrix...")
                mat['SABU'] = predict_fd_matrix("SABU", climate, base, slab, shoulder,
                                                hpcc, jt_sp, cote, mr, epcc,
                                                sa_weights_kips, all_ages, BASE_PATH)
            if ta_weights_kips:
                progress.progress(45, text="Predicting TABU matrix...")
                mat['TABU'] = predict_fd_matrix("TABU", climate, base, slab, shoulder,
                                                hpcc, jt_sp, cote, mr, epcc,
                                                ta_weights_kips, all_ages, BASE_PATH)
                progress.progress(75, text="Predicting TATD matrix...")
                mat['TATD'] = predict_fd_matrix("TATD", climate, base, slab, shoulder,
                                                hpcc, jt_sp, cote, mr, epcc,
                                                ta_weights_kips, all_ages, BASE_PATH)

            progress.progress(85, text="Computing year-by-year...")

            WB_PCT = {'SWB': 0.17, 'MWB': 0.22, 'LWB': 0.61}

            def gf_at(period):
                return float(period) if r == 0.0 else ((1 + r) ** period - 1) / r

            def _ttc_fd(period_years, age_val):
                gf = gf_at(period_years)
                fd = {m: 0.0 for m in ['SABU','TABU','TATD','SWB','MWB','LWB']}
                for row in sa_data:
                    if row['total'] < 0.01 or 'SABU' not in mat: continue
                    scale = (row['total'] * gf) / n_base
                    w     = row['mean_load'] / 1000.0
                    try:
                        wi = sa_weights_kips.index(w)
                        ai = all_ages.index(float(age_val))
                        fd['SABU'] += float(mat['SABU'][ai, wi]) * scale
                    except (ValueError, IndexError):
                        pass
                for row in ta_data:
                    if row['total'] < 0.01: continue
                    scale = (row['total'] * gf) / n_base
                    w     = row['mean_load'] / 1000.0
                    try:
                        wi = ta_weights_kips.index(w)
                        ai = all_ages.index(float(age_val))
                        for m in ['TABU','TATD']:
                            if m in mat:
                                fd[m] += float(mat[m][ai, wi]) * scale
                        for wb, pct in WB_PCT.items():
                            scale_wb = (row['total'] * pct * gf) / n_base
                            if 'TATD' in mat:
                                fd[wb] += float(mat['TATD'][ai, wi]) * scale_wb
                    except (ValueError, IndexError):
                        pass
                return fd

            # HV component
            hv_on = (mode == "Grand Total (TTC + HV)" and "hv_results" in st.session_state)
            hv_trips_u = int(st.session_state.get("hv_trips", 1))

            def _hv_fd(period_years, age_val):
                if not hv_on:
                    return {m: 0.0 for m in ['SABU','TABU','TATD']}
                gf_val = gf_at(period_years)
                scale  = (hv_trips_u * gf_val) / n_base
                fd     = {m: 0.0 for m in ['SABU','TABU','TATD']}
                n_axles = int(st.session_state.get("hv_n_axles", 5))
                for i in range(n_axles):
                    w = float(st.session_state.get(f"hv_w_{i}", 12.0))
                    t = st.session_state.get(f"hv_t_{i}", "Single")
                    try:
                        if t == "Single":
                            m_mat = predict_fd_matrix("SABU", climate, base, slab, shoulder,
                                                      hpcc, jt_sp, cote, mr, epcc,
                                                      [w], [float(age_val)], BASE_PATH)
                            fd['SABU'] += float(m_mat[0,0]) * scale
                        else:
                            for model in ['TABU','TATD']:
                                m_mat = predict_fd_matrix(model, climate, base, slab, shoulder,
                                                          hpcc, jt_sp, cote, mr, epcc,
                                                          [w], [float(age_val)], BASE_PATH)
                                fd[model] += float(m_mat[0,0]) * scale
                    except Exception:
                        pass
                return fd

            def _crk(fd):
                if fd <= 0: return 0.0
                return 100.0 / (1.0 + 0.52 * (fd ** (-2.17)))

            if initial_age > 0:
                ttc_pre = _ttc_fd(initial_age, float(initial_age))
                hv_pre  = _hv_fd(initial_age, float(initial_age))
                fd_pre  = {m: ttc_pre[m] + hv_pre.get(m, 0.0) for m in ['SABU','TABU','TATD','SWB','MWB','LWB']}
            else:
                fd_pre = {m: 0.0 for m in ['SABU','TABU','TATD','SWB','MWB','LWB']}

            cum_fd    = dict(fd_pre)
            yby_rows  = []

            for year in range(1, n_years + 1):
                ttc_n   = _ttc_fd(initial_age + year, float(initial_age + year))
                hv_n    = _hv_fd(initial_age + year, float(initial_age + year))
                fd_n    = {m: ttc_n[m] + hv_n.get(m,0.0) for m in ['SABU','TABU','TATD','SWB','MWB','LWB']}

                ttc_p   = _ttc_fd(initial_age + year - 1, float(initial_age + year - 1))
                hv_p    = _hv_fd(initial_age + year - 1, float(initial_age + year - 1))
                fd_prev = {m: ttc_p[m] + hv_p.get(m,0.0) for m in ['SABU','TABU','TATD','SWB','MWB','LWB']}

                fd_yr   = {m: max(round(fd_n[m] - fd_prev[m], 12), 0.0) for m in ['SABU','TABU','TATD','SWB','MWB','LWB']}
                for m in fd_yr:
                    cum_fd[m] += fd_yr[m]

                fd_bu = cum_fd['SABU'] + cum_fd['TABU']
                fd_td = cum_fd['TATD']
                crk_bu = _crk(fd_bu)
                crk_td = _crk(fd_td)
                total_crack = min(100.0 * (crk_bu/100 + crk_td/100 - (crk_bu/100)*(crk_td/100)), 100.0)

                yby_rows.append({
                    'Year':       year,
                    'SABU (yr)':  f"{fd_yr['SABU']:.4e}",
                    'TABU (yr)':  f"{fd_yr['TABU']:.4e}",
                    'TATD (yr)':  f"{fd_yr['TATD']:.4e}",
                    'SABU Cum.':  f"{cum_fd['SABU']:.4e}",
                    'TABU Cum.':  f"{cum_fd['TABU']:.4e}",
                    'BU Cum.':    f"{cum_fd['SABU']+cum_fd['TABU']:.4e}",
                    'TATD Cum.':  f"{cum_fd['TATD']:.4e}",
                    'CRK_BU (%)': f"{crk_bu:.4f}",
                    'CRK_TD (%)': f"{crk_td:.4f}",
                    'Total Crack (%)': f"{total_crack:.4f}",
                    '_crk_bu': crk_bu,
                    '_crk_td': crk_td,
                    '_total_crack': total_crack,
                    '_sabu_cum': cum_fd['SABU'],
                    '_tabu_cum': cum_fd['TABU'],
                    '_tatd_cum': cum_fd['TATD'],
                })

            progress.progress(100, "Done!")
            progress.empty()

            st.session_state["yby_rows"]   = yby_rows
            st.session_state["yby_cum_fd"] = dict(cum_fd)

        except Exception as e:
            st.error(f"❌ Error: {e}")
            import traceback
            st.code(traceback.format_exc())

    if "yby_rows" in st.session_state:
        yby_rows = st.session_state["yby_rows"]
        cum_fd   = st.session_state["yby_cum_fd"]

        st.markdown("---")
        final = yby_rows[-1]
        col1, col2, col3 = st.columns(3)
        col1.metric("Final CRK_BU",     f"{final['_crk_bu']:.4f} %")
        col2.metric("Final CRK_TD",     f"{final['_crk_td']:.4f} %")
        col3.metric("Final Total Crack", f"{final['_total_crack']:.4f} %")

        import pandas as pd
        display_cols = ['Year','SABU (yr)','TABU (yr)','TATD (yr)',
                        'SABU Cum.','BU Cum.','TATD Cum.',
                        'CRK_BU (%)','CRK_TD (%)','Total Crack (%)']
        df = pd.DataFrame(yby_rows)[display_cols]
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Chart — cracking progression
        import plotly.graph_objects as go
        years = [r['Year'] for r in yby_rows]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years, y=[r['_crk_bu'] for r in yby_rows],
                                 name='CRK_BU', mode='lines+markers', line=dict(color='steelblue')))
        fig.add_trace(go.Scatter(x=years, y=[r['_crk_td'] for r in yby_rows],
                                 name='CRK_TD', mode='lines+markers', line=dict(color='#2d8a2d')))
        fig.add_trace(go.Scatter(x=years, y=[r['_total_crack'] for r in yby_rows],
                                 name='Total Crack', mode='lines+markers', line=dict(color='red', width=2)))
        fig.update_layout(height=400, xaxis_title="Year", yaxis_title="Cracking (%)",
                          title="Cracking Progression Over Design Period")
        st.plotly_chart(fig, use_container_width=True)

        # FD progression chart
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=years, y=[r['_sabu_cum'] for r in yby_rows],
                                  name='SABU Cum.', mode='lines+markers', line=dict(color='steelblue')))
        fig2.add_trace(go.Scatter(x=years, y=[r['_tabu_cum'] for r in yby_rows],
                                  name='TABU Cum.', mode='lines+markers', line=dict(color='#2d8a2d')))
        fig2.add_trace(go.Scatter(x=years, y=[r['_tatd_cum'] for r in yby_rows],
                                  name='TATD Cum.', mode='lines+markers', line=dict(color='#cc8800')))
        fig2.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="FD = 1.0 (Failure)")
        fig2.update_layout(height=400, xaxis_title="Year", yaxis_title="Cumulative FD",
                           title="Cumulative Fatigue Damage Over Design Period")
        st.plotly_chart(fig2, use_container_width=True)

        if st.button("💾 Export Year-by-Year CSV"):
            import io
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            st.download_button("⬇️ Download CSV", buf.getvalue(), "year_by_year.csv", "text/csv")
