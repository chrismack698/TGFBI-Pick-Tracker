import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="TGFBI Pick Tracker 2026", layout="wide")

# ---------------------------
# Config
# ---------------------------
DEFAULT_CSV_URL = "https://raw.githubusercontent.com/chrismack698/TGFBI-Pick-Tracker/main/data/pick_tracker_grid.csv"
CSV_URL = st.secrets.get("CSV_URL", DEFAULT_CSV_URL)

LEAGUE_START = 1062
LEAGUE_END = 1083
LEAGUE_COLS = [str(i) for i in range(LEAGUE_START, LEAGUE_END + 1)]

# ---------------------------
# Helpers
# ---------------------------
@st.cache_data(ttl=60 * 30)
def load_tracker(url: str) -> pd.DataFrame:
    return pd.read_csv(url)

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def build_metrics(df: pd.DataFrame) -> pd.DataFrame:
    for c in LEAGUE_COLS:
        df[c] = df[c].apply(safe_float) if c in df.columns else np.nan

    for col in ["ADP", "Min", "Max", "Range"]:
        df[col] = df[col].apply(safe_float) if col in df.columns else np.nan

    # backfill Min/Max/Range if needed
    if df["Min"].isna().all():
        df["Min"] = df[LEAGUE_COLS].min(axis=1, skipna=True)
    if df["Max"].isna().all():
        df["Max"] = df[LEAGUE_COLS].max(axis=1, skipna=True)
    if df["Range"].isna().all():
        df["Range"] = df["Max"] - df["Min"]

    df["AvgPick"] = df[LEAGUE_COLS].mean(axis=1, skipna=True)

    delta_cols = []
    for c in LEAGUE_COLS:
        dcol = f"d_{c}"
        df[dcol] = df[c] - df["ADP"]
        delta_cols.append(dcol)

    df["WorstReach"] = df[delta_cols].min(axis=1, skipna=True)
    df["BestValue"] = df[delta_cols].max(axis=1, skipna=True)
    df["PickStdDev"] = df[LEAGUE_COLS].std(axis=1, skipna=True)

    df["DiscrepancyScore"] = np.nanmax(
        np.vstack([np.abs(df["WorstReach"].values), np.abs(df["BestValue"].values)]),
        axis=0
    )

    df["Sample"] = df[LEAGUE_COLS].notna().sum(axis=1)

    return df

def sort_direction(sort_by: str) -> bool:
    return sort_by in {"ADP", "AvgPick", "Min", "Max"}

def fmt_int(x):
    if pd.isna(x):
        return ""
    try:
        return f"{int(round(float(x)))}"
    except Exception:
        return ""

def fmt_adp(x):
    if pd.isna(x):
        return ""
    try:
        return f"{float(x):.2f}"
    except Exception:
        return ""

def style_diverging(val, cap):
    if pd.isna(val) or cap <= 0:
        return ""
    v = float(val)
    t = min(abs(v) / cap, 1.0)
    if v < 0:
        alpha = 0.12 + 0.30 * t
        return f"background-color: rgba(255, 0, 0, {alpha});"
    else:
        alpha = 0.10 + 0.26 * t
        return f"background-color: rgba(0, 255, 0, {alpha});"

# ---------------------------
# UI
# ---------------------------
st.title("TGFBI Pick Tracker 2026")

df = build_metrics(load_tracker(CSV_URL))

c1, c2 = st.columns([1.2, 1.6])

with c1:
    show_mode = st.selectbox("Grid shows", ["Picks", "Delta vs ADP"], index=0)
with c2:
    sort_by = st.selectbox(
        "Sort by",
        ["ADP", "DiscrepancyScore", "AvgPick", "Min", "Max", "Range", "PickStdDev", "BestValue", "WorstReach", "Sample"],
        index=0
    )

view = df.copy()
view = view[view["Sample"] >= 3]

asc = sort_direction(sort_by)
view = view.sort_values(sort_by, ascending=asc, na_position="last").reset_index(drop=True)

# ===========================
# Per-league grid (TOP)
# ===========================
st.subheader("Per-league grid")

base_cols = ["Player", "Pos", "ADP", "Min", "Max"]
grid_num = view[base_cols + LEAGUE_COLS].copy()

# Precompute deltas ONCE (numeric), to avoid any string/object issues in styling
delta_matrix = grid_num[LEAGUE_COLS].sub(grid_num["ADP"], axis=0)

# Build display frame (strings)
grid_display = grid_num.copy()
grid_display["ADP"] = grid_num["ADP"].apply(fmt_adp)
grid_display["Min"] = grid_num["Min"].apply(fmt_int)
grid_display["Max"] = grid_num["Max"].apply(fmt_int)

if show_mode == "Delta vs ADP":
    for c in LEAGUE_COLS:
        grid_display[c] = delta_matrix[c].apply(fmt_int)
else:
    for c in LEAGUE_COLS:
        grid_display[c] = grid_num[c].apply(fmt_int)

def grid_styles(row):
    i = row.name
    styles = [""] * len(grid_display.columns)
    offset = 5
    for j, c in enumerate(LEAGUE_COLS):
        styles[offset + j] = style_diverging(delta_matrix.iloc[i, j], 50)
    return styles

styled_grid = grid_display.style.apply(grid_styles, axis=1)
st.dataframe(styled_grid, use_container_width=True, height=800)

st.caption("Coloring: red = drafted earlier than ADP (reach), green = drafted later than ADP (value).")
