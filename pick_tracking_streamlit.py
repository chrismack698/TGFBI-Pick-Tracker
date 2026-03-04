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

    # Base numeric cols
    for col in ["ADP", "Min", "Max", "Range"]:
        if col in df.columns:
            df[col] = df[col].apply(safe_float)

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
            return ("", np.nan)
        # worst reach is negative; best value positive
        if pd.isna(bv) or (not pd.isna(wr) and abs(wr) >= abs(bv)):
            return ("Reach", abs(wr))
        return ("Value", abs(bv))

    tmp = df.apply(label_extreme, axis=1, result_type="expand")
    df["ExtremeType"] = tmp[0]
    df["ExtremeBy"] = tmp[1]  # magnitude in picks

    return df

def style_delta(val, threshold):
    if pd.isna(val):
        return ""
    if val <= -threshold:
        return "background-color: rgba(255, 0, 0, 0.25);"  # reach
    if val >= threshold:
        return "background-color: rgba(0, 255, 0, 0.18);"  # value
    return ""

def sort_direction(sort_by: str) -> bool:
    """
    Return True for ascending sort, False for descending.
    """
    # ascending metrics where "smaller is better/earlier"
    ascending = {"ADP", "AvgPick", "Min"}
    return sort_by in ascending

# ---------------------------
# UI
# ---------------------------
st.title("Pick Tracker — Reaches & Values vs ADP")
st.caption(f"Data source: {CSV_URL}")

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
        ["DiscrepancyScore", "ExtremeBy", "AvgPick", "ADP", "Range", "PickStdDev", "BestValue", "WorstReach", "Sample"],
        index=0
    )

view = df.copy()
if pos_filter:
    view = view[view["Pos"].isin(pos_filter)]

view = view[view["Sample"] >= min_sample]

asc = sort_direction(sort_by)
view = view.sort_values(sort_by, ascending=asc, na_position="last").reset_index(drop=True)

# Summary table
summary_cols = [
    "Player", "Pos", "ADP", "AvgPick", "Sample",
    "ExtremeType", "ExtremeBy",
    "WorstReach", "BestValue",
    "DiscrepancyScore", "Range", "PickStdDev"
]
st.subheader("Most extreme reaches/values")
st.dataframe(view[summary_cols].head(75), use_container_width=True)

# Detailed grid
st.subheader("Per-league grid")

base_cols = ["Player", "Pos", "ADP"]
grid = view[base_cols + LEAGUE_COLS].head(200).copy()

if show_mode == "Delta vs ADP":
    for c in LEAGUE_COLS:
        grid[c] = grid[c] - grid["ADP"]

# Styling based on delta either way:
styled = grid.style.apply(
    lambda row: [""]*3 + [
        style_delta((row[c] if show_mode == "Delta vs ADP" else (row[c] - row["ADP"])), threshold)
        for c in LEAGUE_COLS
    ],
    axis=1
)

st.dataframe(styled, use_container_width=True)

st.caption("Coloring: red = drafted earlier than ADP (reach), green = drafted later than ADP (value).")
