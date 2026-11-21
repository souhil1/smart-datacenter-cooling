"""
app.py - Projet 19 : Smart datacenter cooling  (Streamlit interface, dark "control room" design)

Run locally :   streamlit run app.py
Deploy      :   GitHub repo + https://share.streamlit.io

Reads the files produced by the project (put them in ./outputs or set HVAC_DATA):
    hvac_simulation.csv   rf_metrics.json   rf_hvac.joblib      <- rf_hvac.py
    lstm_predictions.csv                                         <- LSTM notebook   (optional)
Theme colors live in .streamlit/config.toml
"""
import os, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st

st.set_page_config(page_title="Smart Datacenter Cooling", page_icon="❄️", layout="wide")
DATA = Path(os.environ.get("HVAC_DATA", "outputs"))

# ================================================================= DESIGN
CYAN, VIOLET, GREEN, AMBER, RED = "#22d3ee", "#a78bfa", "#34d399", "#fbbf24", "#f87171"
LEVEL_COLORS = ["#38bdf8", AMBER, RED]                   # ECO / NORMAL / BOOST
FG = "#9fb0c8"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, button, input { font-family: 'Inter', sans-serif !important; }
#MainMenu, footer { visibility: hidden; }
.stApp { background: radial-gradient(1200px 600px at 12% -10%, #12223f 0%, transparent 60%),
                     radial-gradient(900px 500px at 95% 0%, #1a1440 0%, transparent 55%), #0a0f1c; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1280px; }
section[data-testid="stSidebar"] { background: #0d1424; border-right: 1px solid rgba(255,255,255,.06); }

.hero h1 { font-size: 2.5rem; font-weight: 800; margin: 0; letter-spacing: -.02em;
           background: linear-gradient(90deg, #22d3ee, #a78bfa 60%, #f472b6); -webkit-background-clip: text;
           -webkit-text-fill-color: transparent; }
.hero p { color: #8ea0bb; margin: .3rem 0 .9rem 0; font-size: 1.02rem; }
.chip { display:inline-block; padding: .28rem .8rem; margin: 0 .35rem .3rem 0; border-radius: 999px; font-size: .8rem;
        font-weight: 600; color: #cfe3ff; background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.10); }
.arrow { color:#5b6b86; margin-right:.35rem; }

.kpi { position: relative; padding: 18px 20px 16px 20px; border-radius: 18px; overflow: hidden;
       background: linear-gradient(150deg, rgba(255,255,255,.07), rgba(255,255,255,.02));
       border: 1px solid rgba(255,255,255,.08); box-shadow: 0 10px 30px rgba(0,0,0,.35); transition: transform .2s; }
.kpi:hover { transform: translateY(-3px); }
.kpi::before { content:""; position:absolute; left:0; top:0; right:0; height:3px; background: var(--accent); }
.kpi-label { color:#8ea0bb; font-size:.78rem; font-weight:600; text-transform:uppercase; letter-spacing:.07em; }
.kpi-value { font-size: 2rem; font-weight: 800; color:#f1f6ff; margin: .25rem 0 .1rem 0; line-height:1.1; }
.kpi-value .unit { font-size: .95rem; font-weight:600; color:#8ea0bb; margin-left:.25rem; }
.kpi-sub { font-size:.82rem; color: var(--accent); font-weight:600; min-height: 1.1rem; }

.card { padding: 18px 20px; border-radius: 18px; background: linear-gradient(150deg, rgba(255,255,255,.06), rgba(255,255,255,.02));
        border: 1px solid rgba(255,255,255,.08); box-shadow: 0 10px 30px rgba(0,0,0,.30); height: 100%; }
.card h4 { margin: 0 0 .7rem 0; font-size: .8rem; text-transform: uppercase; letter-spacing:.08em; color:#8ea0bb; font-weight:700; }
.big { font-size: 2.2rem; font-weight: 800; color:#f1f6ff; line-height:1.1; }
.muted { color:#8ea0bb; font-size:.85rem; }

.pbar { margin: .55rem 0; }
.pbar-top { display:flex; justify-content:space-between; font-size:.82rem; color:#b7c6dc; margin-bottom:.25rem; }
.pbar-track { height:8px; border-radius:99px; background: rgba(255,255,255,.07); overflow:hidden; }
.pbar-fill { height:100%; border-radius:99px; }

.decision { display:inline-block; padding:.45rem 1.4rem; border-radius: 14px; font-size: 1.9rem; font-weight: 800;
            letter-spacing:.04em; color: var(--c); background: color-mix(in srgb, var(--c) 14%, transparent);
            border: 1px solid color-mix(in srgb, var(--c) 55%, transparent); box-shadow: 0 0 28px color-mix(in srgb, var(--c) 35%, transparent); }
.ok  { color:#34d399; font-weight:600; font-size:.9rem; }
.bad { color:#f87171; font-weight:600; font-size:.9rem; }
.stamp { display:inline-block; padding:.3rem .9rem; border-radius:10px; background:rgba(34,211,238,.10);
         border:1px solid rgba(34,211,238,.35); color:#7ee7f7; font-weight:700; font-size:.9rem; margin-bottom:.8rem; }

.stTabs [data-baseweb="tab-list"] { gap: 6px; background: rgba(255,255,255,.04); padding: 6px; border-radius: 14px; width: fit-content; }
.stTabs [data-baseweb="tab"] { height: 40px; border-radius: 10px; padding: 0 18px; color:#8ea0bb; font-weight:600; }
.stTabs [aria-selected="true"] { background: linear-gradient(90deg, rgba(34,211,238,.20), rgba(167,139,250,.20)); color:#fff; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }
[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
    "text.color": FG, "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
    "axes.edgecolor": "#26324a", "axes.grid": True, "grid.color": FG, "grid.alpha": .12,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "font.size": 10, "axes.titleweight": "bold", "axes.titlecolor": "#cfe3ff"})


def kpi(label, value, unit="", sub="", accent=CYAN):
    u = f'<span class="unit">{unit}</span>' if unit else ""
    return (f'<div class="kpi" style="--accent:{accent}"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}{u}</div><div class="kpi-sub">{sub}</div></div>')

def card(title, body):
    return f'<div class="card"><h4>{title}</h4>{body}</div>'

def bar(label, p, color):
    p = float(np.clip(p, 0, 1))
    return (f'<div class="pbar"><div class="pbar-top"><span>{label}</span><b>{p:.0%}</b></div>'
            f'<div class="pbar-track"><div class="pbar-fill" style="width:{p*100:.0f}%;background:{color}"></div></div></div>')

def show(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def time_axis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))


# ================================================================= helpers / data
@st.cache_data
def load_data(folder: str):
    d = Path(folder)
    sim = pd.read_csv(d / "hvac_simulation.csv", index_col=0, parse_dates=True)
    meta = json.load(open(d / "rf_metrics.json"))
    lstm = None
    for cand in [d / "lstm_predictions.csv", Path("lstm_predictions.csv")]:
        if cand.exists():
            lstm = pd.read_csv(cand, index_col=0, parse_dates=True)
            break
    return sim, meta, lstm

@st.cache_resource
def load_model(folder: str):
    try:
        import joblib
        return joblib.load(Path(folder) / "rf_hvac.joblib")
    except Exception as e:
        return str(e)

def decide(P, eps):
    """Cheapest cooling level whose probability of being insufficient is <= eps."""
    tail = 1.0 - np.cumsum(P, axis=1)
    return (tail <= eps + 1e-9).argmax(axis=1)

def energy(act, frac, p_rated):
    return float((p_rated * np.asarray(frac)[np.asarray(act)]).sum() / 60)   # 1 sample = 1 minute -> kWh


try:
    sim, meta, lstm = load_data(str(DATA))
except Exception as e:
    st.error(f"Could not load the result files from '{DATA}': {e}\n\n"
             "Run rf_hvac.py first and copy its outputs into the 'outputs' folder.")
    st.stop()

LEVELS = meta["levels"]
T_WARM, T_HOT = meta["thresholds_C"]["warm"], meta["thresholds_C"]["hotspot"]
H = meta["horizon_min"]

# ================================================================= sidebar
st.sidebar.markdown("## ❄️ Smart Cooling")
st.sidebar.caption("Projet 19 · Datacenter energy efficiency")
period = st.sidebar.radio("Period", ["Test (unseen data)", "Whole recording"], index=0)
eps = st.sidebar.select_slider("Safety level  ε", options=meta["eps_grid"], value=meta["eps"],
                               help="ε = accepted risk of too little cooling. Small ε = safer but more energy.")
with st.sidebar.expander("⚙️ Energy model (assumptions)"):
    p_rated = st.number_input("Rated cooling power (kW)", 1.0, 500.0, float(meta["p_rated_kw"]))
    ef = st.number_input("Emission factor (kgCO₂/kWh)", 0.0, 2.0, float(meta["ef_kg_per_kwh"]), 0.01)
    price = st.number_input("Electricity price (per kWh)", 0.0, 10.0, float(meta["price_per_kwh"]), 0.01)
    frac = [st.slider(f"{LEVELS[i]} power (% of rated)", 10, 100, int(100 * meta["power_frac"][i]), 5) / 100
            for i in range(3)]

view = sim[sim.split == "test"] if period.startswith("Test") else sim
view = view.copy()
P = view[["p_eco", "p_normal", "p_boost"]].to_numpy()
view["action"] = decide(P, eps)
req = view["required"].to_numpy()
lab = ~np.isnan(req)

n = len(view)
e_rf, e_boost = energy(view.action, frac, p_rated), energy(np.full(n, 2), frac, p_rated)
e_react = energy(np.digitize(view.hot_max, [T_WARM, T_HOT]), frac, p_rated)
short = (req[lab] - view.action.to_numpy()[lab]) > 0
under = 100 * float(short.mean()) if lab.any() else float("nan")
saved = e_boost - e_rf
save_pct = 100 * saved / e_boost if e_boost else 0.0

# ================================================================= hero + KPIs
st.markdown(f"""
<div class="hero">
  <h1>Smart Datacenter Cooling</h1>
  <p>Predict the heat, cool only what is needed — less energy, less CO₂, same safety.</p>
  <span class="chip">🧠 CNN · thermal state</span><span class="arrow">→</span>
  <span class="chip">📈 LSTM · CPU load</span><span class="arrow">→</span>
  <span class="chip">🌲 Random Forest · HVAC decision</span>
</div>""", unsafe_allow_html=True)
st.write("")

u_col = GREEN if under <= 2 else (AMBER if under <= 5 else RED)
k = st.columns(5)
k[0].markdown(kpi("Energy used", f"{e_rf:,.1f}", "kWh", f"▼ {save_pct:.1f}% vs always BOOST", GREEN), unsafe_allow_html=True)
k[1].markdown(kpi("Energy saved", f"{saved:,.1f}", "kWh", f"reactive rule: {e_react:,.1f} kWh", CYAN), unsafe_allow_html=True)
k[2].markdown(kpi("CO₂ avoided", f"{saved * ef:,.1f}", "kg", f"at {ef:.2f} kg/kWh", VIOLET), unsafe_allow_html=True)
k[3].markdown(kpi("Cost saved", f"{saved * price:,.2f}", "", f"price {price:.2f} / kWh", AMBER), unsafe_allow_html=True)
k[4].markdown(kpi("Too little cooling", f"{under:.1f}", "%", "of minutes · lower is safer", u_col), unsafe_allow_html=True)
st.caption(f"{n} minutes replayed · energy figures use the assumptions in the sidebar")
st.write("")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "▶ Replay", "🧪 What-if", "📈 Models", "ℹ️ About"])

# ================================================================= TAB 1 : dashboard
with tab1:
    fig, axs = plt.subplots(2, 1, figsize=(12, 5.2), sharex=True, gridspec_kw={"height_ratios": [7, 0.8], "hspace": .08})
    ax = axs[0]
    lo, hi = float(view.cold_mean.min()) - 1.5, float(view.hot_max.max()) + 1
    ax.axhspan(T_WARM, T_HOT, color=AMBER, alpha=.07, lw=0)
    ax.axhspan(T_HOT, hi, color=RED, alpha=.08, lw=0)
    ax.plot(view.index, view.hot_max, color=CYAN, lw=1.6, label="Hottest hot-side sensor")
    ax.plot(view.index, view.cold_mean, color=VIOLET, lw=1.4, label="Cold-side mean")
    ax.axhline(T_WARM, color=AMBER, ls=(0, (4, 4)), lw=.9, label=f"Warm {T_WARM:.1f}°C")
    ax.axhline(T_HOT, color=RED, ls=(0, (4, 4)), lw=.9, label=f"Hotspot {T_HOT:.1f}°C")
    ax.set_ylim(lo, hi); ax.set_ylabel("Temperature (°C)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(.5, 1.14), fontsize=9)
    s = axs[1]
    s.vlines(view.index, 0, 1, colors=[LEVEL_COLORS[a] for a in view.action], lw=1.6)
    s.set_yticks([]); s.set_ylabel("HVAC", rotation=0, labelpad=26, va="center"); s.grid(False); time_axis(s)
    show(fig)
    st.markdown(f'<span class="muted">Strip under the chart: cooling level chosen every minute — '
                f'<b style="color:{LEVEL_COLORS[0]}">ECO</b> · <b style="color:{LEVEL_COLORS[1]}">NORMAL</b> · '
                f'<b style="color:{LEVEL_COLORS[2]}">BOOST</b></span>', unsafe_allow_html=True)
    st.write("")
    c1, c2 = st.columns([3, 2])
    with c1:
        kw_rf = p_rated * np.asarray(frac)[view.action.to_numpy()]
        cum_b = np.cumsum(np.full(n, p_rated * frac[2]) / 60); cum_r = np.cumsum(kw_rf / 60)
        fig, ax = plt.subplots(figsize=(7.5, 3.4))
        ax.plot(view.index, cum_b, color="#8ea0bb", lw=1.6, label="Always BOOST")
        ax.plot(view.index, cum_r, color=GREEN, lw=2, label="RF controller")
        ax.fill_between(view.index, cum_r, cum_b, color=GREEN, alpha=.15, label="Energy saved")
        ax.set_title("Cumulative energy (kWh)", loc="left"); ax.legend(loc="upper left"); time_axis(ax)
        show(fig)
    with c2:
        share = view.action.value_counts(normalize=True).reindex([0, 1, 2]).fillna(0).to_numpy()
        fig, ax = plt.subplots(figsize=(4.2, 3.4)); ax.grid(False)
        w, _ = ax.pie(share + 1e-9, colors=LEVEL_COLORS, startangle=90, counterclock=False,
                      wedgeprops=dict(width=.38, edgecolor="#0a0f1c", linewidth=2))
        ax.text(0, 0, f"{100 * share[0]:.0f}%\nECO", ha="center", va="center", fontsize=15, fontweight="bold", color="#f1f6ff")
        ax.set_title("Time in each level", loc="center")
        ax.legend(w, [f"{l}  {100 * v:.0f}%" for l, v in zip(LEVELS, share)], loc="lower center",
                  bbox_to_anchor=(.5, -.18), ncol=3, fontsize=8, handlelength=1)
        show(fig)

# ================================================================= TAB 2 : replay
with tab2:
    st.write("Move through time and see what each part of the project contributes at that moment.")
    if "i" not in st.session_state:
        st.session_state.i = 0
    st.session_state.i = min(st.session_state.i, n - 1)
    b = st.columns([1, 1, 1, 1, 6])
    if b[0].button("⏮ −10"): st.session_state.i = max(0, st.session_state.i - 10)
    if b[1].button("◀ −1"): st.session_state.i = max(0, st.session_state.i - 1)
    if b[2].button("+1 ▶"): st.session_state.i = min(n - 1, st.session_state.i + 1)
    if b[3].button("+10 ⏭"): st.session_state.i = min(n - 1, st.session_state.i + 10)
    i = st.slider("Time step (minutes)", 0, n - 1, st.session_state.i, key="slider_i")
    st.session_state.i = i
    r = view.iloc[i]
    lvl = int(r.action)
    st.markdown(f'<div class="stamp">🕒 {view.index[i]:%d %b %Y · %H:%M}</div>', unsafe_allow_html=True)
    c = st.columns(4)
    c[0].markdown(card("🧠 CNN · thermal state",
        f'<div class="big">{r.hot_max:.1f}<span class="muted"> °C hot side</span></div>'
        + bar("P(hotspot)", r.p_hotspot, RED) + bar("P(warm)", r.p_warm, AMBER)), unsafe_allow_html=True)
    c[1].markdown(card("📈 LSTM · CPU load",
        f'<div class="big">{r.load:.2f}<span class="muted"> %</span></div>'
        f'<div class="muted">forecast for this time of day</div>'
        f'<div style="margin-top:.6rem;font-weight:700;color:{AMBER if r.load_trend > 0 else GREEN}">'
        f'{r.load_trend:+.2f} pts in {H} min</div>'), unsafe_allow_html=True)
    actual = "" if np.isnan(r.hot_max_future) else f'<div class="muted">actual later: {r.hot_max_future:.1f} °C</div>'
    c[2].markdown(card(f"🌲 Random Forest · in {H} min",
        f'<div class="big">{r.pred_hot_max_future:.1f}<span class="muted"> °C predicted</span></div>{actual}'
        + bar("P(BOOST needed)", r.p_boost, RED)), unsafe_allow_html=True)
    verdict = ""
    if not np.isnan(r.required):
        verdict = ('<div class="ok">✔ Cooling sufficient</div>' if lvl >= int(r.required)
                   else f'<div class="bad">✖ Under-cooling (needed {LEVELS[int(r.required)]})</div>')
    c[3].markdown(card("❄️ HVAC decision",
        f'<div class="decision" style="--c:{LEVEL_COLORS[lvl]}">{LEVELS[lvl]}</div>'
        f'<div class="muted" style="margin-top:.7rem">{p_rated * frac[lvl]:.1f} kW instead of {p_rated * frac[2]:.1f} kW</div>{verdict}'),
        unsafe_allow_html=True)
    st.write("")
    w = view.iloc[max(0, i - 60): i + 1]
    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.axhspan(T_WARM, T_HOT, color=AMBER, alpha=.07, lw=0); ax.axhline(T_HOT, color=RED, ls=(0, (4, 4)), lw=.9)
    ax.plot(w.index, w.hot_max, color=CYAN, lw=1.8); ax.plot(w.index, w.cold_mean, color=VIOLET, lw=1.4)
    ax.scatter([w.index[-1]], [w.hot_max.iloc[-1]], s=70, color=LEVEL_COLORS[lvl], zorder=5, edgecolor="white", linewidth=1.2)
    ax.set_title("Last 60 minutes", loc="left"); ax.set_ylabel("°C"); time_axis(ax)
    show(fig)

# ================================================================= TAB 3 : what-if
with tab3:
    st.write("Describe a situation and ask the trained Random Forest what it would do.")
    mdl = load_model(str(DATA))
    if isinstance(mdl, str):
        st.warning(f"Model file could not be loaded ({mdl}). Re-run rf_hvac.py with the same scikit-learn version "
                   f"as the app (trained with {meta['sklearn_version']}).")
    else:
        base = sim[mdl["features"]].median()
        s1, s2, s3 = st.columns(3)
        hot_max = s1.slider("Hot-side max (°C)", float(sim.hot_max.min()) - 2, float(sim.hot_max.max()) + 3, float(base.hot_max), .1)
        cold = s1.slider("Cold-side mean (°C)", float(sim.cold_mean.min()) - 2, float(sim.cold_mean.max()) + 3, float(base.cold_mean), .1)
        trend = s2.slider("Temperature change, last 5 min (°C)", -2.0, 2.0, float(base.d5), .05)
        loadv = s2.slider("Forecast CPU load (%)", 0.0, float(sim.load.max()) * 1.5, float(base.load), .1)
        ltr = s2.slider(f"Load change expected in {H} min", -2.0, 2.0, float(base.load_trend), .05)
        phot = s3.slider("CNN  P(hotspot)", 0.0, 1.0, float(base.p_hotspot), .01)
        pwarm = s3.slider("CNN  P(warm)", 0.0, 1.0 - phot, min(float(base.p_warm), 1.0 - phot), .01)
        x = base.copy()
        x["hot_max"], x["cold_mean"], x["d5"], x["load"], x["load_trend"] = hot_max, cold, trend, loadv, ltr
        x["p_hotspot"], x["p_warm"] = phot, pwarm
        x["hot_mean"] = base.hot_mean + (hot_max - base.hot_max) * .8
        x["delta_T"] = x.hot_mean - cold
        x["d1"] = trend / 5
        X = pd.DataFrame([x[mdl["features"]]])
        Pw = np.zeros((1, 3)); Pw[:, mdl["clf"].classes_.astype(int)] = mdl["clf"].predict_proba(X)
        a = int(decide(Pw, eps)[0])
        f = float(hot_max + mdl["reg"].predict(X)[0])
        st.write("")
        o1, o2 = st.columns(2)
        o1.markdown(card("Decision",
            f'<div class="decision" style="--c:{LEVEL_COLORS[a]}">{LEVELS[a]}</div>'
            f'<div class="muted" style="margin-top:.8rem">Predicted hot-side max in {H} min</div>'
            f'<div class="big">{f:.1f}<span class="muted"> °C</span></div>'), unsafe_allow_html=True)
        o2.markdown(card("Probability of the level required",
            "".join(bar(LEVELS[j], Pw[0, j], LEVEL_COLORS[j]) for j in range(3))), unsafe_allow_html=True)
        st.caption("Features not shown here (humidity, short-term std) are kept at their median value.")

# ================================================================= TAB 4 : models
with tab4:
    m = meta["forecast_metrics"]
    q = st.columns(3)
    q[0].markdown(kpi("Forecast error (RF)", f"{m['rf_mae_C']:.2f}", "°C", f"+{H} min · MAE", GREEN), unsafe_allow_html=True)
    q[1].markdown(kpi("'Stays the same' error", f"{m['persistence_mae_C']:.2f}", "°C", "baseline MAE", AMBER), unsafe_allow_html=True)
    q[2].markdown(kpi("R² of RF forecast", f"{m['rf_r2']:.2f}", "", f"baseline {m['persistence_r2']:.2f}", CYAN), unsafe_allow_html=True)
    st.write("")
    st.markdown("##### Energy and thermal risk on unseen test data")
    tt = pd.DataFrame(meta["test_table"]).T.rename(columns={
        "energy_kWh": "Energy (kWh)", "co2_kg": "CO₂ (kg)", "cost": "Cost", "undercool_pct": "Under-cooling (%)",
        "severe_undercool_pct": "Severe (%)", "saving_vs_boost_pct": "Saving vs BOOST (%)"})
    st.dataframe(tt.round(2), use_container_width=True)
    cA, cB = st.columns(2)
    with cA:
        st.markdown("##### What do CNN and LSTM bring? (ablation)")
        st.dataframe(pd.DataFrame(meta["ablation"]).T.round(3), use_container_width=True)
    with cB:
        imp = pd.Series(meta["feature_importance"]).sort_values()
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        ax.barh(imp.index, imp.values, color=CYAN, height=.62); ax.grid(axis="y", alpha=0)
        ax.set_title("Feature importance", loc="left")
        show(fig)
    for f_, cap in [("fig_rf_tradeoff.png", "Energy saving vs thermal risk"), ("fig_rf_timeline.png", "Test timeline")]:
        if (DATA / f_).exists():
            st.image(str(DATA / f_), caption=cap)
    if lstm is not None:
        fig, ax = plt.subplots(figsize=(12, 3))
        l2 = lstm.iloc[:576]
        ax.plot(l2.index, l2.actual_load, color=VIOLET, lw=1.4, label="Actual")
        ax.plot(l2.index, l2.predicted_load, color=CYAN, lw=1.4, label="LSTM forecast")
        ax.set_title("LSTM · CPU load forecast (2 days)", loc="left"); ax.set_ylabel("CPU load (%)"); ax.legend()
        show(fig)

# ================================================================= TAB 5 : about
with tab5:
    a1, a2 = st.columns(2)
    a1.markdown(card("How it works", f"""
<div class="muted" style="font-size:.95rem;line-height:1.7">
<b style="color:#cfe3ff">CNN</b> classifies the thermal map of the racks (Normal / Warm / Hotspot).<br>
<b style="color:#cfe3ff">LSTM</b> forecasts the total CPU load of the servers.<br>
<b style="color:#cfe3ff">Random Forest</b> predicts the cooling level needed in {H} minutes and picks the cheapest level
that stays safe (risk ≤ ε).</div>"""), unsafe_allow_html=True)
    a2.markdown(card("Honest limits", """
<div class="muted" style="font-size:.95rem;line-height:1.7">
• Workload (2013) and sensors (2018) are linked <b>by time of day only</b>.<br>
• No HVAC actions in the data → replay is <b>open-loop</b>.<br>
• Cooling power per level, CO₂ factor and price are <b>assumptions</b>.<br>
• Recording is short (≈30 h): confirm on longer data.</div>"""), unsafe_allow_html=True)
    st.write("")
    st.json({k_: meta[k_] for k_ in ["horizon_min", "eps", "thresholds_C", "sklearn_version", "test_period"]}, expanded=False)
