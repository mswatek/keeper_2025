import requests
from requests_oauthlib import OAuth2Session
from requests.auth import HTTPBasicAuth
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import time
import random
import os

csv_path = 'daily_team_combined_stats.csv'

def get_latest_csv_date(csv_file):
    if not os.path.exists(csv_file):
        return None  # No data yet
    df = pd.read_csv(csv_file)
    if df.empty:
        return None
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df['date'].max().date()

#csv_cutoff_date = datetime(2025, 5, 30).date() #use this when I don't want to go all the way to current day
csv_cutoff_date = datetime.today().date() - timedelta(days=1)
latest_in_csv = get_latest_csv_date(csv_path)

if latest_in_csv and latest_in_csv >= csv_cutoff_date:
    print(f"✅ CSV already has data through {csv_cutoff_date}. No update needed.")
else:
    start_from = latest_in_csv + timedelta(days=1) if latest_in_csv else datetime(2025, 3, 18).date()
    end_at = csv_cutoff_date
    print(f"📅 Running update from {start_from} to {end_at}")


def append_to_csv(dataframe, filename):
    file_exists = os.path.isfile(filename)
    dataframe.to_csv(
        filename,
        mode='a',
        header=not file_exists,
        index=False
    )


def update_team_names(existing_csv_path, latest_data):
    # Load the existing dataset
    df_existing = pd.read_csv(existing_csv_path)

    # Create a lookup dictionary: latest team_name per team_key
    latest_lookup = (
        latest_data.sort_values("date")
        .groupby("team_key")
        .tail(1)[["team_key", "team_name"]]
        .drop_duplicates(subset="team_key", keep="last")
        .set_index("team_key")["team_name"]
        .to_dict()
    )

    # Update team_name using the mapping, keeping unmatched names as-is
    df_existing["team_name"] = df_existing["team_key"].map(latest_lookup).fillna(df_existing["team_name"])

    # Overwrite the original CSV
    df_existing.to_csv(existing_csv_path, index=False)
    print(f"✅ Updated team_name values in {existing_csv_path}")


class YahooTokenManager:
    def __init__(self, client_id, client_secret, refresh_token):
        self.client_id = client_id
        self.client_secret = client_secret
        self.refresh_token = refresh_token
        self.token_url = 'https://api.login.yahoo.com/oauth2/get_token'
        self.token = None
        self.token_acquired_at = None
        self.expires_in = 3600  # Yahoo access tokens usually expire after 1 hour

        self.refresh_access_token()

    def refresh_access_token(self):
        payload = {
            'grant_type': 'refresh_token',
            'refresh_token': self.refresh_token,
            'redirect_uri': 'oob'
        }

        response = requests.post(
            self.token_url,
            data=payload,
            auth=HTTPBasicAuth(self.client_id, self.client_secret),
            headers={'Content-Type': 'application/x-www-form-urlencoded'}
        )

        if response.status_code == 200:
            self.token = response.json()
            self.token_acquired_at = time.time()
            print("🔄 Access token refreshed.")
        else:
            raise Exception(f"Token refresh failed: {response.text}")

    def get_headers(self):
        if self.token is None or (time.time() - self.token_acquired_at > self.expires_in - 60):
            print("⚠️ Token nearing expiration — refreshing...")
            self.refresh_access_token()

        return {
            'Authorization': f'Bearer {self.token["access_token"]}',
            'Accept': 'application/json'
        }

# Replace with your own credentials
client_id = "dj0yJmk9VEtpWVNNQzd1TVRtJmQ9WVdrOVRUQkpObXRuTjJrbWNHbzlNQS0tJnM9Y29uc3VtZXJzZWNyZXQmc3Y9MCZ4PTcy"
client_secret = "23f4d294641cc580d381c647f8932711f19a50e8"
refresh_token = "AMooCWXYBYMcjT_AzcfJWIeedRC4~000~nJZ43mNIt2q7pdxBi3U-"

token_manager = YahooTokenManager(client_id, client_secret, refresh_token)


# Use your actual team key and desired date
league_key = '458.l.9833'
team_key = '458.l.9833.t.2'  # Example format
date = '2025-06-24'
endpoint = f'https://fantasysports.yahooapis.com/fantasy/v2/team/{team_key}/stats;type=date;date={date}'


# -- Date range --
#start_date = datetime(2025, 3, 18) #started march 18
#end_date = datetime(2025, 6, 27)
delta = timedelta(days=1)

# -- Get list of teams in league --
league_url = f'https://fantasysports.yahooapis.com/fantasy/v2/league/{league_key}/teams'
headers = token_manager.get_headers()
league_resp = requests.get(league_url, headers=headers)
league_root = ET.fromstring(league_resp.text)
ns = {'fantasy': 'http://fantasysports.yahooapis.com/fantasy/v2/base.rng'}

teams = []
for team in league_root.findall(".//fantasy:team", ns):
    team_key = team.find("fantasy:team_key", ns).text
    team_name = team.find("fantasy:name", ns).text
    teams.append((team_key, team_name))


############################# get individual player stats #####################################

stat_map = {
    "0": "GP", "1": "GP2", "2": "GS", "3": "AVG", "4": "OBP", "5": "SLG", "6": "AB", "7": "R", "8": "H",
    "9": "1B", "10": "2B", "11": "3B", "12": "HR", "13": "RBI", "14": "SH", "15": "SF", "16": "SB",
    "17": "CS", "18": "BB", "19": "IBB", "20": "HBP", "21": "SO", "22": "GIDP", "23": "TB", "24": "APP",
    "25": "GS", "26": "ERA", "27": "WHIP", "28": "W", "29": "L", "30": "CG", "31": "SHO", "32": "SV",
    "33": "OUT", "34": "HA", "35": "TBF", "36": "RA", "37": "ER", "38": "HRA", "39": "BBA", "40": "IBBA",
    "41": "HBPA", "42": "K", "43": "WP", "44": "BLK", "45": "SBA", "46": "GIDPA", "47": "SVOP", "48": "HLD",
    "49": "TBA", "50": "IP", "51": "PO", "52": "A", "53": "E", "54": "FPCT", "55": "OPS", "56": "K/BB",
    "57": "K/9", "58": "TEAM", "59": "LEAGUE", "60": "H/AB", "61": "XBH", "62": "NSB", "63": "SB%",
    "64": "CYC", "65": "PA", "66": "SLAM", "67": "PC", "68": "2BA", "69": "3BA", "70": "RW", "71": "RL",
    "72": "PICK", "73": "RAPP", "74": "OBPA", "75": "WIN%", "76": "1BA", "77": "H/9", "78": "BB/9",
    "79": "NH", "80": "PG", "81": "SV%", "82": "IRA", "83": "QS", "84": "BSV", "85": "NSV", "86": "OFA",
    "87": "DPT", "88": "CI", "89": "SVH", "90": "NSVH", "91": "NW"
}


def get_roster_for_day(team_key, date_str, headers):
    import time
    start_t = time.time()
    roster_url = f"https://fantasysports.yahooapis.com/fantasy/v2/team/{team_key}/roster;date={date_str}"

    try:
        headers = token_manager.get_headers()
        response = requests.get(roster_url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        print(f"⏰ Timeout while fetching roster for {team_key} on {date_str}")
        return []
    except requests.exceptions.RequestException as e:
        print(f"🚨 Error fetching roster: {e}")
        return []

    print(f"📦 Roster fetched for {team_key} on {date_str} in {time.time() - start_t:.2f}s")

    roster_root = ET.fromstring(response.text)
    players = []

    for player in roster_root.findall(".//fantasy:player", ns):
        player_id = player.find("fantasy:player_id", ns).text.strip()
        name = player.find("fantasy:name/fantasy:full", ns).text
        #print(f"📋 From roster: {name} — player_id = '{player_id}'")
        position_type = player.find("fantasy:position_type", ns).text
        selected_position_elem = player.find("fantasy:selected_position/fantasy:position", ns)
        slot = selected_position_elem.text if selected_position_elem is not None else "NA"

        players.append({
            "player_id": player_id,
            "name": name,
            "position_type": position_type,
            "roster_slot": slot
        })

    print(f"🧍 {len(players)} players found for team {team_key}")
    return players



MAX_RETRIES = 5
RETRY_WAIT = 600  # 10 minutes
rate_limit_triggered = False
throttle_delay = 0.3

failed_attempts = {}
player_stats = []
final_retry_queue = []

def fetch_player_stats_with_retry(stat_url, headers):
    global rate_limit_triggered
    for attempt in range(1, MAX_RETRIES + 1):
        time.sleep(throttle_delay)
        try:
            headers = token_manager.get_headers()
            resp = requests.get(stat_url, headers=headers, timeout=10)

            if resp.status_code == 200:
                return resp
            elif resp.status_code == 401:
                print(f"🔐 Attempt {attempt}: status code 401 — Unauthorized")
                return "unauthorized"
            elif resp.status_code == 999:
                print(f"🧱 Attempt {attempt}: status code 999 — Rate limited")
                if not rate_limit_triggered:
                    print("⏸️ Pausing for 2 minutes...")
                    time.sleep(120)
                    rate_limit_triggered = True
            elif resp.status_code in [429, 500, 502, 503, 504]:
                print(f"🔁 Attempt {attempt}: transient error {resp.status_code}")
                time.sleep(60)
            else:
                print(f"⚠️ Attempt {attempt}: status code {resp.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"❗ Attempt {attempt}: exception - {e}")
        time.sleep(2 ** attempt)
    print("⛔ Exceeded max retries:", stat_url)
    return None

def parse_and_append_stats(resp, player, team_name, team_key, date_str):
    try:
        player_root = ET.fromstring(resp.text)
        stat_nodes = player_root.findall(".//fantasy:stat", ns)
    except ET.ParseError as e:
        print(f"❌ XML parsing error for {player['name']} on {date_str}: {e}")
        return False

    if not stat_nodes:
        print(f"🛑 No stats for {player['name']} on {date_str}")
        return False

    for stat in stat_nodes:
        stat_id = stat.find("fantasy:stat_id", ns).text
        value = stat.find("fantasy:value", ns)
        stat_val = value.text if value is not None else None

        player_stats.append({
            "date": date_str,
            "team_name": team_name,
            "team_key": team_key,
            "player_name": player["name"],
            "player_id": player["player_id"],
            "roster_slot": player["roster_slot"],
            "stat_id": stat_id,
            "stat_name": stat_map.get(stat_id, f"Stat {stat_id}"),
            "value": stat_val
        })
    return True

def process_retry_queue(queue_name, queue):
    print(f"🔁 Processing {len(queue)} entries in {queue_name}...")
    new_queue = []

    for team_key, team_name, player, stat_url, date_str in queue:
        pid = player["player_id"]
        key = (pid, date_str)
        failed_attempts[key] = failed_attempts.get(key, 0) + 1

        if failed_attempts[key] > MAX_RETRIES:
            print(f"⚠️ Skipping {player['name']} on {date_str} after {MAX_RETRIES} retries.")
            log_failed_attempt(pid, player['name'], date_str, stat_url)
            continue

        resp = fetch_player_stats_with_retry(stat_url, headers)
        if resp is None or resp == "unauthorized" or resp.status_code != 200:
            new_queue.append((team_key, team_name, player, stat_url, date_str))
            continue

        parse_and_append_stats(resp, player, team_name, team_key, date_str)

    return new_queue

def log_failed_attempt(pid, name, date_str, url):
    with open("failed_stat_urls.txt", "a") as log:
        log.write(f"{pid},{name},{date_str},{url}\n")

# --- Looping through dates and teams ---
current_date = start_from
while current_date <= end_at:
    date_str = current_date.strftime('%Y-%m-%d')
    print(f"\n📅 Processing {date_str}...")
    retry_queue = []

    for team_key, team_name in teams:
        print(f"🔹 Team: {team_name}")
        roster = get_roster_for_day(team_key, date_str, headers)

        for i, player in enumerate(roster):
            if i > 0 and i % 17 == 0:
                print("⏳ Pause after 17 players...")
                time.sleep(5)

            if player["roster_slot"] == "IL":
                print(f"🚑 Skipping IL: {player['name']}")
                continue

            pid = player["player_id"]
            player_key = f"mlb.p.{pid}"
            stat_url = f"https://fantasysports.yahooapis.com/fantasy/v2/player/{player_key}/stats;type=date;date={date_str}"

            resp = fetch_player_stats_with_retry(stat_url, headers)
            if resp == "unauthorized":
                print(f"🔐 Skipping {player['name']} on {date_str}")
                continue
            if resp is None or resp.status_code != 200:
                retry_queue.append((team_key, team_name, player, stat_url, date_str))
                continue

            parse_and_append_stats(resp, player, team_name, team_key, date_str)

    # Retry loop per day
    if retry_queue:
        print(f"🕒 Waiting 10 minutes before retrying {len(retry_queue)} fetches...")
        time.sleep(RETRY_WAIT)
        retry_queue = process_retry_queue("retry_queue", retry_queue)

    # Final fallback to global queue
    final_retry_queue.extend(retry_queue)
    print("🧊 Cooling off before next day...")
    time.sleep(15)
    current_date += delta

# Final retry at end
if final_retry_queue:
    print(f"\n🔁 Final retry round for {len(final_retry_queue)} failures...")
    final_retry_queue = process_retry_queue("final_retry_queue", final_retry_queue)

# Save the final dataset
player_df = pd.DataFrame(player_stats)
#player_df.to_csv("daily_player_stats.csv", index=False)
#print("✅ Saved daily player stats to CSV.")


# Ensure 'value' is numeric
player_df["value"] = pd.to_numeric(player_df["value"], errors="coerce")

# Pivot to wide: one row per (date, team, player), one column per stat
player_wide = player_df.pivot_table(
    index=["date", "team_name", "team_key", "player_name", "player_id", "roster_slot"],
    columns="stat_name",
    values="value",
    aggfunc="sum"
).reset_index()

# Update IP

# Assume the 'Innings Pitched' column contains strings or floats like: 5.1, 6.2, etc.
def convert_ip(ip):
    if pd.isna(ip):
        return None
    whole = int(float(ip))
    frac = round(float(ip) - whole, 2)
    if frac == 0.1:
        return whole + 0.333
    elif frac == 0.2:
        return whole + 0.667
    else:
        return float(ip)
    
player_wide["IP"] = player_wide["IP"].apply(convert_ip)

# Save it
#player_wide.to_csv("daily_player_stats_wide.csv", index=False) #when running for first time
append_to_csv(player_wide, 'daily_player_stats_wide.csv')
update_team_names('daily_player_stats_wide.csv', player_wide)
print("✅ Saved player-wide stats: daily_player_stats_wide.csv")


# 🧼 Filter out bench players — only include active contributors
active_wide = player_wide[player_wide["roster_slot"] != "BN"].copy()

# 🪣 Step 1: Group and sum at the team-day level
group_cols = ["date", "team_name", "team_key"]
value_cols = [
    col for col in active_wide.columns 
    if col not in group_cols + ["player_name", "player_id", "roster_slot"]
]

summed_stats = (
    active_wide
    .groupby(group_cols)[value_cols]
    .sum()
    .reset_index()
)


# 🧮 Step 2: Calculate derived metrics using formulas
def safe_div(num, denom):
    return num / denom if denom else None

def has_cols(df, cols):
    return all(col in df.columns for col in cols)

if has_cols(summed_stats, ["H", "AB"]):
    summed_stats["AVG"] = summed_stats.apply(lambda row: safe_div(row["H"], row["AB"]), axis=1)

if has_cols(summed_stats, ["H", "BB", "HBP", "PA"]):
    summed_stats["OBP"] = summed_stats.apply(
        lambda row: safe_div(row["H"] + row["BB"] + row["HBP"], row["PA"]), axis=1
    )

if has_cols(summed_stats, ["TB", "AB"]):
    summed_stats["SLG"] = summed_stats.apply(lambda row: safe_div(row["TB"], row["AB"]), axis=1)

if has_cols(summed_stats, ["OBP", "SLG"]):
    summed_stats["OPS"] = summed_stats["OBP"] + summed_stats["SLG"]

if has_cols(summed_stats, ["K", "BBA"]):
    summed_stats["KBB"] = summed_stats.apply(lambda row: safe_div(row["K"], row["BBA"]), axis=1)

if has_cols(summed_stats, ["ER", "IP"]):
    summed_stats["ERA"] = summed_stats.apply(lambda row: safe_div(row["ER"] * 9, row["IP"]), axis=1)

if has_cols(summed_stats, ["BBA", "HA", "IP"]):
    summed_stats["WHIP"] = summed_stats.apply(lambda row: safe_div(row["BBA"] + row["HA"], row["IP"]), axis=1)

# 🧭 Reorder columns for presentation
desired_order = ["date", "team_name", "team_key", "R", "HR", "RBI", "SB", "AVG", "OPS", "K", "ERA", "WHIP", "KBB", "QS", "SVH"]
all_columns = list(summed_stats.columns)
unordered_cols = [col for col in all_columns if col not in desired_order]
final_cols = [col for col in desired_order if col in all_columns] + unordered_cols
summed_stats = summed_stats[final_cols]

# 💾 Save the final version
#summed_stats.to_csv("daily_team_combined_stats.csv", index=False) #when running for first time
append_to_csv(summed_stats, 'daily_team_combined_stats.csv')
update_team_names("daily_team_combined_stats.csv", summed_stats)
print("📊 Saved full team stats (summed + derived) to daily_team_combined_stats.csv")
