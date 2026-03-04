import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="ADP Discrepancy Tracker", layout="wide")

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
    # Ensure league cols exist and are numeric
    for c in LEAGUE_COLS:
        if c in df.columns:
            df[c] = df[c].apply(safe_float)
        else:
            df[c] = np.nan

    # Base numeric cols (Min/Max/Range may already exist from your CSV)
    for col in ["ADP", "Min", "Max", "Range"]:
        if col in df.columns:
            df[col] = df[col].apply(safe_float)
        else:
            df[col] = np.nan

    # If Min/Max weren't present or are empty, compute them from league cols
    if df["Min"].isna().all():
        df["Min"] = df[LEAGUE_COLS].min(axis=1, skipna=True)
    if df["Max"].isna().all():
        df["Max"] = df[LEAGUE_COLS].max(axis=1, skipna=True)
    if df["Range"].isna().all():
        df["Range"] = df["Max"] - df["Min"]

    # Average pick across leagues (ignoring NaNs)
    df["AvgPick"] = df[LEAGUE_COLS].mean(axis=1, skipna=True)

    # Deltas per league: Pick - ADP
    delta_cols = []
    for c in LEAGUE_COLS:
        dcol = f"d_{c}"
        df[dcol] = df[c] - df["ADP"]
        delta_cols.append(dcol)

    # Reach/value summaries
    df["WorstReach"] = df[delta_cols].min(axis=1, skipna=True)   # most negative (picked earlier)
    df["BestValue"]  = df[delta_cols].max(axis=1, skipna=True)   # most positive (picked later)
    df["PickStdDev"] = df[LEAGUE_COLS].std(axis=1, skipna=True)

    # Discrepancy score
    df["DiscrepancyScore"] = np.nanmax(
        np.vstack([np.abs(df["WorstReach"].values), np.abs(df["BestValue"].values)]),
        axis=0
    )

    # How many leagues have a pick recorded
    df["Sample"] = df[LEAGUE_COLS].notna().sum(axis=1)

    # Human-readable “most extreme” direction + magnitude
    def label_extreme(row):
        wr = row["WorstReach"]
        bv = row["BestValue"]
        if pd.isna(wr) and pd.isna(bv):
            return ("", np.nan, np.nan)

        # Decide which is more extreme in magnitude; store signed "NetReach"
        # Reach -> negative, Value -> positive
        if pd.isna(bv) or (not pd.isna(wr) and abs(wr) >= abs(bv)):
            return ("Reach", abs(wr), -abs(wr))
        return ("Value", abs(bv), abs(bv))

    tmp = df.apply(label_extreme, axis=1, result_type="expand")
    df["ExtremeType"] = tmp[0]
    df["ExtremeBy"] = tmp[1]   # magnitude in picks (always positive)
    df["NetReach"] = tmp[2]    # signed: negative reach, positive value

    return df

def sort_direction(sort_by: str) -> bool:
    """
    Return True for ascending sort, False for descending.
    """
    ascending = {"ADP", "AvgPick", "Min", "Max"}  # smaller = earlier
    return sort_by in ascending

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
    """
    Diverging red/green based on signed value.
    val < 0 => red (reach), val > 0 => green (value)
    cap controls saturation: abs(val)>=cap -> strongest color.
    """
    if pd.isna(val) or cap <= 0:
        return ""

    v = float(val)
    t = min(abs(v) / cap, 1.0)  # 0..1

    if v < 0:
        # red with alpha scaled by magnitude
        alpha = 0.12 + 0.30 * t
        return f"background-color: rgba(255, 0, 0, {alpha});"
    else:
        # green with alpha scaled by magnitude
        alpha = 0.10 + 0.26 * t
        return f"background-color: rgba(0, 255, 0, {alpha});"

# ---------------------------
# UI
# ---------------------------
st.title("Pick Tracker — Reaches & Values vs ADP")

df = load_tracker(CSV_URL)
df = build_metrics(df)

# Controls
c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1.2, 1.6])

with c1:
    pos_filter = st.multiselect(
        "Position",
        sorted([p for p in df["Pos"].dropna().unique()]),
        default=[]
    )
with c2:
    min_sample = st.number_input("Min leagues sampled", min_value=0, max_value=len(LEAGUE_COLS), value=3)
with c3:
    threshold = st.slider("Highlight threshold (picks)", min_value=1, max_value=50, value=10)
with c4:
    show_mode = st.selectbox("Grid shows", ["Picks", "Delta vs ADP"], index=0)
with c5:
    sort_by = st.selectbox(
        "Sort by",
        ["ADP", "DiscrepancyScore", "ExtremeBy", "AvgPick", "Min", "Max", "Range", "PickStdDev", "BestValue", "WorstReach", "Sample"],
        index=0
    )

view = df.copy()
if pos_filter:
    view = view[view["Pos"].isin(pos_filter)]

view = view[view["Sample"] >= min_sample]

asc = sort_direction(sort_by)
view = view.sort_values(sort_by, ascending=asc, na_position="last").reset_index(drop=True)

# ---------------------------
# Summary table (formatted)
# ---------------------------
st.subheader("Most extreme reaches/values")

summary_cols = [
    "Player", "Pos", "ADP", "Min", "Max", "Range", "AvgPick", "Sample",
    "ExtremeType", "ExtremeBy", "NetReach",
    "WorstReach", "BestValue",
    "DiscrepancyScore", "PickStdDev"
]

summary = view[summary_cols].copy()

# Display formatting: ints everywhere except ADP (2 decimals)
int_cols = ["Min", "Max", "Range", "AvgPick", "Sample", "ExtremeBy", "NetReach", "WorstReach", "BestValue", "DiscrepancyScore", "PickStdDev"]
for col in int_cols:
    summary[col] = summary[col].apply(fmt_int)
summary["ADP"] = view["ADP"].apply(fmt_adp)

# Style: diverging on NetReach (overall reach/value)
# (We need the underlying numeric series for styling, so style from view not summary)
styled_summary = summary.style.apply(
    lambda row: [""] * len(summary.columns),
    axis=1
)

# Apply cell-level style for NetReach and ExtremeBy using underlying numeric 'view'
netreach_idx = summary.columns.get_loc("NetReach")
extremeby_idx = summary.columns.get_loc("ExtremeBy")

def summary_row_styles(i):
    # i is row index in the styled frame order
    nr = view.iloc[i]["NetReach"]
    styles = [""] * len(summary.columns)
    styles[netreach_idx] = style_diverging(nr, threshold)
    styles[extremeby_idx] = style_diverging(nr, threshold)  # same direction color
    return styles

styled_summary = summary.style.apply(lambda r: summary_row_styles(r.name), axis=1)

st.dataframe(styled_summary.head(75), use_container_width=True)

# ---------------------------
# Per-league grid
# ---------------------------
st.subheader("Per-league grid")

base_cols = ["Player", "Pos", "ADP", "Min", "Max"]
grid = view[base_cols + LEAGUE_COLS].head(200).copy()

# Display formatting
grid_display = grid.copy()
grid_display["ADP"] = grid["ADP"].apply(fmt_adp)
grid_display["Min"] = grid["Min"].apply(fmt_int)
grid_display["Max"] = grid["Max"].apply(fmt_int)

if show_mode == "Delta vs ADP":
    for c in LEAGUE_COLS:
        grid_display[c] = (grid[c] - grid["ADP"]).apply(fmt_int)
else:
    for c in LEAGUE_COLS:
        grid_display[c] = grid[c].apply(fmt_int)

# Styling: diverging based on delta (signed)
def grid_styles(row):
    # row is from grid (numeric), but we output styles for grid_display columns
    styles = [""] * len(grid_display.columns)

    # base col count: Player, Pos, ADP, Min, Max => 5
    offset = 5

    for j, c in enumerate(LEAGUE_COLS):
        if show_mode == "Delta vs ADP":
            delta = row[c] - row["ADP"]
        else:
            delta = row[c] - row["ADP"]  # still color by delta even when showing picks
        styles[offset + j] = style_diverging(delta, threshold)

    return styles

styled_grid = grid_display.style.apply(grid_styles, axis=1)

st.dataframe(styled_grid, use_container_width=True)

st.caption("Coloring: red = drafted earlier than ADP (reach), green = drafted later than ADP (value).")
