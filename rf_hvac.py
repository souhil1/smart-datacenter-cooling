"""
rf_hvac.py  -  Projet 19 : Random Forest HVAC controller  (part 3/3 of the project)

PIPELINE
    CNN  (friend)  -> cnn_output_1min.csv       thermal state of the racks (p_normal / p_warm / p_hotspot ...)
    LSTM (you)     -> lstm_predictions.csv      forecast of the total CPU load
    RF   (this)    -> decides the cooling level for the next minutes (ECO / NORMAL / BOOST)

WHAT THE SCRIPT DOES
    1. Loads the CNN + LSTM outputs (auto-search in Kaggle / local folders)
    2. Builds features (current temps, humidity, CNN probabilities, LSTM load + load trend, short history)
    3. Trains 2 Random Forests on a chronological split (train / validation / test, no shuffling):
         - RandomForestRegressor  : hottest hot-side temperature in HORIZON minutes
         - RandomForestClassifier : cooling level REQUIRED in HORIZON minutes (0 ECO, 1 NORMAL, 2 BOOST)
    4. Turns the classifier into a risk-aware controller: choose the cheapest level whose
       probability of being insufficient is <= EPS   (EPS is picked on VALIDATION)
    5. Replays the test period and compares energy / CO2 / cost / thermal risk against
       "always BOOST", "always NORMAL" and a reactive rule without ML
    6. Saves everything the interface (app.py) needs into OUT_DIR

RUN  (Kaggle notebook cell or local terminal):      python rf_hvac.py
"""
import os, glob, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib, sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (mean_absolute_error, r2_score, f1_score, accuracy_score,
                             confusion_matrix, classification_report)

warnings.filterwarnings("ignore")

# =============================== CONFIG (edit here only) ===============================
SEED = 42
HORIZON = 5                      # minutes ahead the controller anticipates
MAX_UNDERCOOL = 0.02             # accepted share of minutes with insufficient cooling (used to pick EPS)
EPS_GRID = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]

LEVELS = ["ECO", "NORMAL", "BOOST"]
POWER_FRAC = [0.45, 0.70, 1.00]  # share of rated cooling power used by each level   (ASSUMPTION)
P_RATED_KW = 20.0                # rated electrical power of the cooling system (kW)   (ASSUMPTION)
EF_KG_PER_KWH = 0.50             # grid emission factor kgCO2/kWh                      (ASSUMPTION)
PRICE_PER_KWH = 0.10             # electricity price, any currency                     (ASSUMPTION)

OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "outputs"
os.makedirs(OUT_DIR, exist_ok=True)
SEARCH_ROOTS = [".", OUT_DIR, "/kaggle/input", "/kaggle/working", "outputs", "data"]
rng = np.random.default_rng(SEED)


def find(name, required=True):
    """Look for a file by name in the usual places (Kaggle datasets are in /kaggle/input/...)."""
    for root in SEARCH_ROOTS:
        hits = glob.glob(os.path.join(root, "**", name), recursive=True)
        if hits:
            return sorted(hits, key=len)[0]
    if required:
        raise FileNotFoundError(f"{name} not found. Add it to the Kaggle notebook as a dataset, "
                                f"or copy it next to this script.")
    return None


# =============================== 1. LOAD ===============================
f_cnn, f_lstm = find("cnn_output_1min.csv"), find("lstm_predictions.csv")
f_meta = find("cnn_handoff_meta.json", required=False)
print("CNN  :", f_cnn, "\nLSTM :", f_lstm, "\nMeta :", f_meta)

cnn = pd.read_csv(f_cnn, index_col=0, parse_dates=True).sort_index()
lstm = pd.read_csv(f_lstm, index_col=0, parse_dates=True).sort_index()
cnn = cnn[~cnn.index.duplicated()]
cnn = cnn.dropna(subset=["hot_max_C", "hot_mean_C", "cold_mean_C"])
cnn = cnn.asfreq("1min").interpolate(limit=3).dropna()             # regular 1-minute grid
print(f"CNN output : {cnn.shape}  {cnn.index[0]} -> {cnn.index[-1]}")
print(f"LSTM output: {lstm.shape}  {lstm.index[0]} -> {lstm.index[-1]}")

# ---- IMPORTANT: the two datasets were recorded at different dates (workload 2013, sensors 2018).
# They are coupled through the TIME OF DAY: the LSTM load forecast is averaged per time-of-day (daily
# profile) and read at the sensor timestamp. State this assumption in the report.
tod = lstm.index.hour * 60 + lstm.index.minute
profile = lstm.groupby(tod)["predicted_load"].mean()
def load_at(ts):
    minute = (ts.hour * 60 + ts.minute).to_numpy()
    return np.interp(minute, profile.index.values, profile.values, period=1440)

# =============================== 2. FEATURES ===============================
idx = cnn.index
hot = cnn["hot_max_C"]
F = pd.DataFrame(index=idx)
F["hot_max"] = hot
F["hot_mean"] = cnn["hot_mean_C"]
F["cold_mean"] = cnn["cold_mean_C"]
F["delta_T"] = cnn["hot_mean_C"] - cnn["cold_mean_C"]
F["rh_hot"] = cnn.get("rh_hot_mean", np.nan)
F["rh_cold"] = cnn.get("rh_cold_mean", np.nan)
F["p_warm"] = cnn["p_warm"]                       # CNN output
F["p_hotspot"] = cnn["p_hotspot"]                 # CNN output
F["d1"] = hot.diff(1)                             # temperature trend
F["d5"] = hot.diff(5)
F["std5"] = hot.rolling(5).std()
F["load"] = load_at(idx)                          # LSTM output (time-of-day profile)
F["load_trend"] = load_at(idx + pd.Timedelta(minutes=HORIZON)) - F["load"]   # LSTM: load change expected

FEATS_ALL = list(F.columns)
FEATS_NO_LSTM = [c for c in FEATS_ALL if not c.startswith("load")]
FEATS_SENSORS = [c for c in FEATS_NO_LSTM if c not in ("p_warm", "p_hotspot")]

# thresholds shared with the CNN (so labels mean the same thing in both parts)
n = len(F)
a, b = int(.70 * n), int(.85 * n)
if f_meta:
    th = json.load(open(f_meta))["thresholds_C"]
    T_WARM, T_HOT = float(th["warm"]), float(th["hotspot"])
else:
    T_WARM, T_HOT = np.quantile(hot.iloc[:a], [0.60, 0.90])
print(f"Thresholds: warm >= {T_WARM:.2f} C | hotspot >= {T_HOT:.2f} C")

# targets
fut = hot.shift(-HORIZON)
required = pd.Series(np.digitize(fut, [T_WARM, T_HOT]), index=idx).where(fut.notna())
delta_target = fut - hot

valid = F.dropna().index.intersection(fut.dropna().index)
pos = pd.Series(np.arange(n), index=idx)
tr = valid[pos[valid] < a]
va = valid[(pos[valid] >= a + HORIZON) & (pos[valid] < b)]          # HORIZON gap: no label leakage
te = valid[pos[valid] >= b + HORIZON]
print(f"Samples  train={len(tr)}  val={len(va)}  test={len(te)}")
print("Required level share (train):", np.bincount(required[tr].astype(int), minlength=3) / len(tr))

# =============================== 3. MODELS ===============================
def make_clf():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features="sqrt",
                                  class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)
def make_reg():
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.6,
                                 n_jobs=-1, random_state=SEED)

def to3(proba, classes):
    """probabilities over the classes seen in training -> always 3 columns (ECO, NORMAL, BOOST)"""
    P = np.zeros((len(proba), 3)); P[:, np.asarray(classes, int)] = proba
    return P

def decide(P, eps):
    """Cheapest level L such that P(required level > L) <= eps."""
    tail = 1.0 - np.cumsum(P, axis=1)
    return (tail <= eps + 1e-9).argmax(axis=1)

clf = make_clf().fit(F.loc[tr, FEATS_ALL], required[tr].astype(int))
reg = make_reg().fit(F.loc[tr, FEATS_ALL], delta_target[tr])

# ---- forecast quality (TEST) : RF vs "temperature stays the same" ----
pred_future = F.loc[te, "hot_max"] + reg.predict(F.loc[te, FEATS_ALL])
reg_metrics = {
    "rf_mae_C": mean_absolute_error(fut[te], pred_future),
    "persistence_mae_C": mean_absolute_error(fut[te], F.loc[te, "hot_max"]),
    "rf_r2": r2_score(fut[te], pred_future),
    "persistence_r2": r2_score(fut[te], F.loc[te, "hot_max"]),
}
print("\nForecast hot_max(t+%d min), TEST:" % HORIZON, {k: round(v, 4) for k, v in reg_metrics.items()})

# ---- classification quality (TEST) + ablation : what do CNN / LSTM bring? ----
def clf_scores(feats):
    m = make_clf().fit(F.loc[tr, feats], required[tr].astype(int))
    p = m.predict(F.loc[te, feats])
    y = required[te].astype(int)
    return {"accuracy": accuracy_score(y, p), "macro_f1": f1_score(y, p, average="macro", zero_division=0)}
ablation = {"sensors only": clf_scores(FEATS_SENSORS),
            "sensors + CNN": clf_scores(FEATS_NO_LSTM),
            "sensors + CNN + LSTM (full)": clf_scores(FEATS_ALL)}
print("\nAblation (TEST):"); print(pd.DataFrame(ablation).T.round(3))

y_te = required[te].astype(int)
P_te = to3(clf.predict_proba(F.loc[te, FEATS_ALL]), clf.classes_)
print("\nClassification report (argmax, TEST):")
print(classification_report(y_te, P_te.argmax(1), labels=[0, 1, 2], target_names=LEVELS, zero_division=0))

# =============================== 4. CONTROL + ENERGY ===============================
DT_H = 1 / 60                                       # one sample = 1 minute

def simulate(act, req):
    kw = P_RATED_KW * np.asarray(POWER_FRAC)[np.asarray(act)]
    kwh = kw.sum() * DT_H
    short = np.maximum(np.asarray(req) - np.asarray(act), 0)
    return {"energy_kWh": kwh, "co2_kg": kwh * EF_KG_PER_KWH, "cost": kwh * PRICE_PER_KWH,
            "undercool_pct": 100 * float((short > 0).mean()),
            "severe_undercool_pct": 100 * float((short == 2).mean())}

def table(P, req, hot_now, eps_list):
    k = len(req)
    rows = {"Always BOOST (baseline)": simulate(np.full(k, 2), req),
            "Always NORMAL": simulate(np.full(k, 1), req),
            "Reactive rule (no ML)": simulate(np.digitize(hot_now, [T_WARM, T_HOT]), req)}
    for e in eps_list:
        rows[f"RF controller eps={e}"] = simulate(decide(P, e), req)
    t = pd.DataFrame(rows).T
    base = t.loc["Always BOOST (baseline)", "energy_kWh"]
    t["saving_vs_boost_pct"] = 100 * (1 - t["energy_kWh"] / base)
    return t

# pick eps on VALIDATION: cheapest controller with under-cooling <= MAX_UNDERCOOL
P_va = to3(clf.predict_proba(F.loc[va, FEATS_ALL]), clf.classes_)
t_val = table(P_va, required[va].astype(int).to_numpy(), F.loc[va, "hot_max"].to_numpy(), EPS_GRID)
ok = [e for e in EPS_GRID if t_val.loc[f"RF controller eps={e}", "undercool_pct"] <= 100 * MAX_UNDERCOOL]
EPS = max(ok) if ok else min(EPS_GRID)             # largest eps meeting the safety target = cheapest
print(f"\nChosen EPS (validation, under-cooling <= {100*MAX_UNDERCOOL:.0f}%): {EPS}")

t_test = table(P_te, y_te.to_numpy(), F.loc[te, "hot_max"].to_numpy(), EPS_GRID)
print("\nENERGY / RISK on TEST period (%d min):" % len(te)); print(t_test.round(2).to_string())
t_test.to_csv(f"{OUT_DIR}/rf_energy_table_test.csv")

# =============================== 5. FULL REPLAY FILE FOR THE INTERFACE ===============================
Fa = F.dropna()
Pa = to3(clf.predict_proba(Fa[FEATS_ALL]), clf.classes_)
sim = pd.DataFrame(index=Fa.index)
sim["split"] = "none"
sim.loc[Fa.index.intersection(tr), "split"] = "train"
sim.loc[Fa.index.intersection(va), "split"] = "val"
sim.loc[Fa.index.intersection(te), "split"] = "test"
for c in FEATS_ALL:                                  # every model input (the interface needs them)
    sim[c] = Fa[c]
sim["pred_hot_max_future"] = Fa["hot_max"] + reg.predict(Fa[FEATS_ALL])
sim["hot_max_future"] = fut.reindex(Fa.index)
sim["required"] = required.reindex(Fa.index)
sim["p_eco"], sim["p_normal"], sim["p_boost"] = Pa[:, 0], Pa[:, 1], Pa[:, 2]
sim["action"] = decide(Pa, EPS)
sim.index.name = "timestamp"
sim.to_csv(f"{OUT_DIR}/hvac_simulation.csv")

imp = pd.Series(clf.feature_importances_, index=FEATS_ALL).sort_values(ascending=False)

meta = {"horizon_min": HORIZON, "eps": EPS, "eps_grid": EPS_GRID, "levels": LEVELS,
        "power_frac": POWER_FRAC, "p_rated_kw": P_RATED_KW, "ef_kg_per_kwh": EF_KG_PER_KWH,
        "price_per_kwh": PRICE_PER_KWH, "thresholds_C": {"warm": T_WARM, "hotspot": T_HOT},
        "features": FEATS_ALL, "forecast_metrics": reg_metrics, "ablation": ablation,
        "feature_importance": imp.round(4).to_dict(), "sklearn_version": sklearn.__version__,
        "n_test": int(len(te)), "test_period": [str(te[0]), str(te[-1])],
        "test_table": json.loads(t_test.round(4).to_json(orient="index")),
        "assumptions": ["Workload (2013) and sensor (2018) datasets are coupled by time of day",
                        "Cooling power fractions / rated power / emission factor / price are assumptions",
                        "Open-loop replay: no HVAC actuator data exists, temperatures are assumed independent of the action"]}
json.dump(meta, open(f"{OUT_DIR}/rf_metrics.json", "w"), indent=2)
joblib.dump({"clf": clf, "reg": reg, "features": FEATS_ALL, "eps": EPS, "meta": meta},
            f"{OUT_DIR}/rf_hvac.joblib", compress=3)

# =============================== 6. FIGURES ===============================
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .3})

fig, ax = plt.subplots(figsize=(6, 4))
imp[::-1].plot.barh(ax=ax, color="#2a7"); ax.set_title("Random Forest - feature importance")
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig_rf_importance.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4))
r = t_test.loc[[f"RF controller eps={e}" for e in EPS_GRID]]
ax.plot(r["undercool_pct"], r["saving_vs_boost_pct"], "o-", label="RF controller (eps sweep)")
for nm, mk in [("Always NORMAL", "s"), ("Reactive rule (no ML)", "^")]:
    ax.plot(t_test.loc[nm, "undercool_pct"], t_test.loc[nm, "saving_vs_boost_pct"], mk, ms=9, label=nm)
ax.set_xlabel("Minutes with insufficient cooling (%)"); ax.set_ylabel("Energy saving vs always BOOST (%)")
ax.set_title("Energy saving vs thermal risk (test)"); ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig_rf_tradeoff.png", dpi=200); plt.close(fig)

s = sim[sim.split == "test"]
fig, axs = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
axs[0].plot(s.index, s.hot_max, label="hot_max (actual)", lw=1)
axs[0].plot(s.index, s.pred_hot_max_future.shift(HORIZON), label=f"RF forecast (made {HORIZON} min earlier)", lw=1, alpha=.8)
axs[0].axhline(T_WARM, color="orange", ls="--", lw=.8); axs[0].axhline(T_HOT, color="red", ls="--", lw=.8)
axs[0].set_ylabel("C"); axs[0].legend(loc="upper right"); axs[0].set_title("Test period : temperature and RF decision")
axs[1].step(s.index, s.action, where="post", label="RF level"); axs[1].step(s.index, s.required.shift(HORIZON), where="post", alpha=.6, label="required")
axs[1].set_yticks([0, 1, 2]); axs[1].set_yticklabels(LEVELS); axs[1].legend(loc="upper right")
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig_rf_timeline.png", dpi=200); plt.close(fig)

cm = confusion_matrix(y_te, P_te.argmax(1), labels=[0, 1, 2])
fig, ax = plt.subplots(figsize=(4.5, 4)); ax.imshow(cm, cmap="Blues"); ax.grid(False)
ax.set_xticks(range(3)); ax.set_xticklabels(LEVELS); ax.set_yticks(range(3)); ax.set_yticklabels(LEVELS)
ax.set_xlabel("Predicted level"); ax.set_ylabel("Required level"); ax.set_title("RF confusion matrix (test)")
for i in range(3):
    for j in range(3): ax.text(j, i, cm[i, j], ha="center", va="center", color="red")
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig_rf_confusion.png", dpi=200); plt.close(fig)

print("\nSaved in", OUT_DIR, ":", sorted(f for f in os.listdir(OUT_DIR) if f.startswith(("rf_", "fig_rf", "hvac_"))))
