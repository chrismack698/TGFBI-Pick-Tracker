import time
import requests
import pandas as pd

BASE_URL = "https://draft.shgn.com/api/public/players/dp/{}"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://draft.shgn.com/",
    "Origin": "https://draft.shgn.com",
}

def fetch_league_players(dp_id: int) -> list[dict]:
    url = BASE_URL.format(dp_id)
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()  # list[dict]

def parse_league(df_json: list[dict], dp_id: int) -> pd.DataFrame:
    rows = []
    for x in df_json:
        full_name = f"{(x.get('f') or '').strip()} {(x.get('l') or '').strip()}".strip()
        player_id = x.get("playerId")
        pick = x.get("pick") or {}
        overall_pick = pick.get("tu")  # overall pick number

        rows.append({
            "LeagueId": dp_id,
            "PlayerId": player_id,
            "Player": full_name,
            "OverallPick": overall_pick,
            "Pos": x.get("e") or x.get("p"),
            "ADP": x.get("adp"),
            "Team": x.get("t"),
        })

    df = pd.DataFrame(rows)
    return df.sort_values(["OverallPick", "ADP"], na_position="last").reset_index(drop=True)

def build_pick_tracker(all_leagues_df: pd.DataFrame) -> pd.DataFrame:
    # Use PlayerId as the true unique key; keep Player for display
    grid = all_leagues_df.pivot_table(
        index=["PlayerId", "Player", "Pos", "ADP"],
        columns="LeagueId",
        values="OverallPick",
        aggfunc="min"
    ).reset_index()

    league_cols = [c for c in grid.columns if isinstance(c, int)]  # leagueId columns are ints

    grid["Min"] = grid[league_cols].min(axis=1, skipna=True)
    grid["Max"] = grid[league_cols].max(axis=1, skipna=True)
    grid["Range"] = grid["Max"] - grid["Min"]

    # Order columns like your screenshot
    grid = grid[["Player", "Pos", "ADP", "Min", "Max", "Range"] + league_cols]
    grid = grid.sort_values(["ADP", "Min"], na_position="last").reset_index(drop=True)
    return grid

def main(start_id: int = 1062, end_id: int = 1083, pause_s: float = 0.15):
    league_dfs = []

    for dp_id in range(start_id, end_id + 1):
        try:
            data = fetch_league_players(dp_id)
            league_dfs.append(parse_league(data, dp_id))
        except requests.HTTPError as e:
            print(f"[WARN] dp_id={dp_id} HTTP error: {e}")
        except Exception as e:
            print(f"[WARN] dp_id={dp_id} error: {e}")

        time.sleep(pause_s)  # be polite

    combined = pd.concat(league_dfs, ignore_index=True) if league_dfs else pd.DataFrame()
    tracker = build_pick_tracker(combined) if not combined.empty else pd.DataFrame()

    return combined, tracker

if __name__ == "__main__":
    combined_df, tracker_df = main(1062, 1083)
    combined_df.to_csv("league_picks_long.csv", index=False)
    tracker_df.to_csv("pick_tracker_grid.csv", index=False)

    print("Wrote league_picks_long.csv and pick_tracker_grid.csv")
    print(tracker_df.head(20))
