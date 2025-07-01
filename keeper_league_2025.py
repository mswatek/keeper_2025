import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from datetime import timedelta

# === Global setup ===
st.set_page_config(page_title="Fantasy Baseball Dashboard", layout="wide")

# === Tabs ===
main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs(["🏆 League Summary", "📈 League Trends", "👥 Team Stats", "👥 Player Stats"])

# === LEAGUE TRENDS TAB ===
with main_tab2:

    # Load team-level data
    df = pd.read_csv("daily_team_combined_stats.csv", parse_dates=["date"], usecols=lambda col: col != "team_key")
    min_date, max_date = df["date"].min(), df["date"].max()

    # Stat categories
    roto_cats = {
        'R': True, 'HR': True, 'RBI': True, 'SB': True,
        'AVG': True, 'OPS': True, 'K': True, 'ERA': False,
        'WHIP': False, 'KBB': True, 'QS': True, 'SVH': True
    }
    batting_stats = ['R', 'HR', 'RBI', 'SB', 'AVG', 'OPS']
    pitching_stats = ['K', 'ERA', 'WHIP', 'KBB', 'QS', 'SVH']
    raw_stats = ['H', 'AB', 'BB', 'HBP', 'PA', 'TB', 'K', 'BBA', 'ER', 'IP', 'HA']
    rate_stats = ['AVG', 'OPS', 'ERA', 'WHIP', 'KBB']

    # Date range selector
    st.sidebar.header("🗓️ Date Range")
    presets = {
        "Full Season": (min_date, max_date),
        "Last 7 Days": (max_date - timedelta(days=6), max_date),
        "Last 14 Days": (max_date - timedelta(days=13), max_date),
        "Last 30 Days": (max_date - timedelta(days=29), max_date),
    }
    preset_choice = st.sidebar.radio("Quick Select", list(presets.keys()) + ["Custom"], index=4)
    start_date, end_date = presets.get(preset_choice, (min_date, max_date))
    start_date = st.sidebar.date_input("Start Date", value=start_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End Date", value=end_date, min_value=min_date, max_value=max_date)

    # Stat filtering
    st.header("📊 Roto Scoring Mode")
    roto_mode = st.radio("Choose Stat Types", ["All", "Batting", "Pitching"], horizontal=True)

    all_stats = list(roto_cats.keys())

    selected_stats = (
        batting_stats if roto_mode == "Batting"
        else pitching_stats if roto_mode == "Pitching"
        else list(roto_cats.keys())
    )

    # Filter data
    df_range = df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))].copy()
    numeric_cols = [col for col in df_range.columns if col not in ["date", "team_name"]]
    cumulative = df_range.groupby(["team_name", "date"])[numeric_cols].sum().reset_index()
    cumulative[numeric_cols] = cumulative.groupby("team_name")[numeric_cols].cumsum()

    # Safe division
    def safe_div(n, d): return n / d if d else None

    # Rate stat calculations
    cumulative["AVG"] = cumulative.apply(lambda r: safe_div(r["H"], r["AB"]), axis=1)
    cumulative["OBP"] = cumulative.apply(lambda r: safe_div(r["H"] + r["BB"] + r["HBP"], r["PA"]), axis=1)
    cumulative["SLG"] = cumulative.apply(lambda r: safe_div(r["TB"], r["AB"]), axis=1)
    cumulative["OPS"] = cumulative["OBP"] + cumulative["SLG"]
    cumulative["ERA"] = cumulative.apply(lambda r: safe_div(r["ER"] * 9, r["IP"]), axis=1)
    cumulative["WHIP"] = cumulative.apply(lambda r: safe_div(r["BBA"] + r["HA"], r["IP"]), axis=1)
    cumulative["KBB"] = cumulative.apply(lambda r: safe_div(r["K"], r["BBA"]), axis=1)

    # Roto ranking
    num_teams = cumulative["team_name"].nunique()
    roto_ranks = pd.DataFrame(index=cumulative.index)
    for stat in selected_stats:
        if stat in cumulative.columns:
            asc = not roto_cats[stat]
            colname = f"roto_{stat}"
            roto_ranks[colname] = cumulative.groupby("date")[stat].rank(ascending=asc, method="average")
            roto_ranks[colname] = num_teams + 1 - roto_ranks[colname]

    cumulative = pd.concat([cumulative, roto_ranks], axis=1)
    cumulative["Roto_Points"] = cumulative[roto_ranks.columns].sum(axis=1)

    # 📈 Roto Chart
    st.title("📈 Roto Points Over Time")
    fig = px.line(
        cumulative,
        x=np.array(cumulative["date"]),
        y="Roto_Points",
        color="team_name",
        title=f"{roto_mode} Roto Points from {start_date} to {end_date}",
        labels={"team_name": "Team", "Roto_Points": "Roto Points"}
    )
    st.plotly_chart(fig, use_container_width=True)

    # 📊 Cumulative Stat Chart
    st.header("📊 Cumulative Stat Totals Over Time")
    stat_choice = st.selectbox("Choose a Stat to View Totals", options=selected_stats)
    if stat_choice in cumulative.columns:
        fig2 = px.line(
            cumulative,
            x=np.array(cumulative["date"]),
            y=stat_choice,
            color="team_name",
            title=f"Cumulative {stat_choice} from {start_date} to {end_date}",
            labels={"team_name": "Team", stat_choice: f"{stat_choice} Total"}
        )
        st.plotly_chart(fig2, use_container_width=True)


# === ROTO SUMMARY TAB ===
with main_tab1:

    st.title("🏆 Overall Roto Standings")

    june_first = pd.to_datetime("2025-06-01")
    latest_date = df["date"].max()

    def compute_cumulative_roto(data, stats_subset):
        data = data.copy()

        # Cumulative build
        numeric_cols = [col for col in data.columns if col not in ["date", "team_name"]]
        cumulative = data.groupby(["team_name", "date"])[numeric_cols].sum().reset_index()
        cumulative[numeric_cols] = cumulative.groupby("team_name")[numeric_cols].cumsum()

        # Rate stats
        def safe_div(n, d): return n / d if d else None

        if {"H", "AB"}.issubset(cumulative.columns):
            cumulative["AVG"] = cumulative.apply(lambda row: safe_div(row["H"], row["AB"]), axis=1)
        if {"H", "BB", "HBP", "PA"}.issubset(cumulative.columns):
            cumulative["OBP"] = cumulative.apply(lambda r: safe_div(r["H"] + r["BB"] + r["HBP"], r["PA"]), axis=1)
        if {"TB", "AB"}.issubset(cumulative.columns):
            cumulative["SLG"] = cumulative.apply(lambda r: safe_div(r["TB"], r["AB"]), axis=1)
        if {"OBP", "SLG"}.issubset(cumulative.columns):
            cumulative["OPS"] = cumulative["OBP"] + cumulative["SLG"]
        if {"K", "BBA"}.issubset(cumulative.columns):
            cumulative["KBB"] = cumulative.apply(lambda r: safe_div(r["K"], r["BBA"]), axis=1)
        if {"ER", "IP"}.issubset(cumulative.columns):
            cumulative["ERA"] = cumulative.apply(lambda r: safe_div(r["ER"] * 9, r["IP"]), axis=1)
        if {"BBA", "HA", "IP"}.issubset(cumulative.columns):
            cumulative["WHIP"] = cumulative.apply(lambda r: safe_div(r["BBA"] + r["HA"], r["IP"]), axis=1)

        # Final-day only
        latest = cumulative[cumulative["date"] == cumulative["date"].max()]

        num_teams = latest["team_name"].nunique()
        roto_cols = []
        for stat in stats_subset:
            if stat in latest.columns:
                asc = not roto_cats[stat]
                colname = f"roto_{stat}"
                latest[colname] = latest[stat].rank(ascending=asc, method="average")
                latest[colname] = num_teams + 1 - latest[colname]
                roto_cols.append(colname)

        latest["Roto Points"] = latest[roto_cols].sum(axis=1)
        return latest[["team_name", "Roto Points"] + roto_cols]

    full_roto = compute_cumulative_roto(df, all_stats).rename(columns={"Roto Points": "Total Roto Points"})
    hitting_roto = compute_cumulative_roto(df, batting_stats).rename(columns={"Roto Points": "Hitting Roto Points"})
    pitching_roto = compute_cumulative_roto(df, pitching_stats).rename(columns={"Roto Points": "Pitching Roto Points"})

    # Before June 1st (First Half)
    before_june = df[df["date"] < june_first]
    before_roto = compute_cumulative_roto(before_june, all_stats).rename(
        columns={"Roto Points": "First Half Roto Points (Before June 1)"}
    )
    
    # Since June 1st (Second Half)
    since_june = df[df["date"] >= june_first]
    june_roto = compute_cumulative_roto(since_june, all_stats).rename(columns={"Roto Points": "Second Half Roto Points (Since June 1)"})



    # Merge all
    summary = full_roto[["team_name", "Total Roto Points"]].merge(
        hitting_roto[["team_name", "Hitting Roto Points"]],
        on="team_name"
    ).merge(
        pitching_roto[["team_name", "Pitching Roto Points"]],
        on="team_name"
    ).merge(
        before_roto[["team_name", "First Half Roto Points (Before June 1)"]],
        on="team_name"
    ).merge(
        june_roto[["team_name", "Second Half Roto Points (Since June 1)"]],
        on="team_name"
    )

    # ➕ Add improvement column
    summary["First vs Second Half"] = (
        summary["Second Half Roto Points (Since June 1)"] - summary["First Half Roto Points (Before June 1)"]
    )

   # 🔢 Round numeric columns
    numeric_cols = summary.select_dtypes(include=["float", "int"]).columns
    summary[numeric_cols] = summary[numeric_cols].round(1)

    # 🎨 Conditional styling for improvement column
    def highlight_improvement(val):
        if val > 0:
            return "background-color: #b6e2b6"  # light green
        elif val < 0:
            return "background-color: #f8c291"  # light orange
        else:
            return ""

    # 🎛️ Build final styled table
    styled_df = summary.sort_values(by="Total Roto Points", ascending=False).style\
        .format({col: "{:.1f}" for col in numeric_cols})\
        .map(highlight_improvement, subset=["First vs Second Half"])\
        .set_properties(**{"text-align": "center"})\
        .set_table_styles([{"selector": "th", "props": [("text-align", "center")]}])

    st.dataframe(styled_df, use_container_width=True, hide_index=True)


# === TEAM STATS TAB ===
with main_tab3:

    st.header("📋 Miscellaneous Stat Tables")

    def filter_by_date(df):
        return df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))]

    # Load and filter wide-format player data
    df = pd.read_csv("daily_player_stats_wide.csv", parse_dates=["date"])
    active_df = df[df["roster_slot"] != "BN"].copy()
    active_df = filter_by_date(active_df)

    batting_stats = ["1B", "2B", "3B", "TB", "SO", "GIDP", "HBP", "BB", "CS", "IBB", "SLAM"]
    fielding_stats = ["PO", "A", "E"]
    pitching_stats = ["PC", "TBF", "RAPP", "1BA", "2BA", "3BA", "BSV", "PICK", "SBA", "BBA"]

    batting_summary = active_df.groupby("team_name")[batting_stats].sum(min_count=1).reset_index()
    fielding_summary = active_df.groupby("team_name")[fielding_stats].sum(min_count=1).reset_index()
    pitching_summary = active_df.groupby("team_name")[pitching_stats].sum(min_count=1).reset_index()

    stat_tab1, stat_tab2, stat_tab3 = st.tabs(["⚾ Batting", "🧤 Fielding", "🔥 Pitching"])
    with stat_tab1:
        st.subheader("🟦 Batting Stats")
        st.dataframe(batting_summary)
    with stat_tab2:
        st.subheader("🟩 Fielding Stats")
        st.dataframe(fielding_summary)
    with stat_tab3:
        st.subheader("🟥 Pitching Stats")
        st.dataframe(pitching_summary)

    # === Active Roster Highlights ===
    df = pd.read_csv("daily_player_stats_wide.csv", parse_dates=["date"])
    df = df[df["roster_slot"] != "BN"].copy()
    df = filter_by_date(df)
    df["IP"] = pd.to_numeric(df["IP"], errors="coerce")
    df["ER"] = pd.to_numeric(df.get("ER"), errors="coerce")
    df["SO"] = pd.to_numeric(df.get("SO"), errors="coerce")

    st.header("🎯 Active Roster Highlights and Lowlights")

    teams = sorted(df["team_name"].unique())
    selected_team = st.selectbox("Select a team:", teams)

    st.subheader("🟪 Near-Quality Starts (= 5.667 IP & ≤ 3 ER)")
    qs_df = df[(df["team_name"] == selected_team) & (df["IP"] == 5.667) & (df["ER"] <= 3)][["date", "player_name", "IP", "ER"]]
    st.dataframe(qs_df, use_container_width=True)

    st.subheader("🟦 Golden Sombreros (4+ Strikeouts)")
    k_df = df[(df["team_name"] == selected_team) & (df["SO"] >= 4)][["date", "player_name", "SO", "AB"]]
    st.dataframe(k_df, use_container_width=True)

    # === Bench Events ===
    st.header("🚨 Bench Highlights and Lowlights")

    bench_df = pd.read_csv("daily_player_stats_wide.csv", parse_dates=["date"])
    bench_df = bench_df[bench_df["roster_slot"] == "BN"].copy()
    bench_df = filter_by_date(bench_df)

    cols_to_check = ["HR", "SB", "SO", "ER", "QS", "SVH", "SLAM", "H", "PC", "K", "AB"]
    for col in cols_to_check + ["IP"]:
        if col in bench_df.columns:
            bench_df[col] = pd.to_numeric(bench_df[col], errors="coerce")

    bench_teams = sorted(bench_df["team_name"].unique())
    bench_tabs = st.tabs([f"🪑 {team}" for team in bench_teams])

    for team, tab in zip(bench_teams, bench_tabs):
        with tab:
            team_df = bench_df[bench_df["team_name"] == team]
            summary = {
                "HR_SB": ((team_df["HR"] >= 1) & (team_df["SB"] >= 1)).sum(),
                "HR2": (team_df["HR"] >= 2).sum(),
                "QS": team_df["QS"].fillna(0).astype(int).sum() if "QS" in team_df.columns else 0,
                "SVH": team_df["SVH"].fillna(0).astype(int).sum() if "SVH" in team_df.columns else 0,
                "SLAM": team_df["SLAM"].fillna(0).astype(int).sum() if "SLAM" in team_df.columns else 0,
                "ER5": (team_df["ER"] >= 5).sum(),
                "SO4": (team_df["SO"] >= 4).sum(),
            }

            st.markdown(f"""
            🧾 **Summary**: {team} missed out on  
            • `{summary['HR_SB']}` HR+SB combos  
            • `{summary['HR2']}` multi-HR games  
            • `{summary['QS']}` Quality Starts  
            • `{summary['SVH']}` Saves+Holds  
            • `{summary['SLAM']}` Grand Slams  

            But they also dodged  
            • `{summary['ER5']}` outings with 5+ ER  
            • `{summary['SO4']}` 4+ strikeout appearances left on the bench.
            """, unsafe_allow_html=True)

            st.subheader("Combo Meals (💣 HR + SB on Bench)")
            st.dataframe(team_df[(team_df["HR"] >= 1) & (team_df["SB"] >= 1)][["date", "player_name", "HR", "SB"]])

            st.subheader("🔁 Multi-HR Games (2+)")
            st.dataframe(team_df[team_df["HR"] >= 2][["date", "player_name", "H", "HR"]])

            st.subheader("🧱 Quality Starts (QS)")
            st.dataframe(team_df[team_df.get("QS", 0) >= 1][["date", "player_name", "IP", "ER", "QS"]])

            st.subheader("🧯 Saves + Holds (SVH)")
            st.dataframe(team_df[team_df.get("SVH", 0) >= 1][["date", "player_name", "PC", "SVH"]])

            st.subheader("🔥 Double-Digit Strikeouts (K ≥ 10)")
            st.dataframe(team_df[team_df["SO"] >= 10][["date", "player_name", "K", "IP"]])

            st.subheader("🌊 5+ ER Allowed")
            st.dataframe(team_df[team_df["ER"] >= 5][["date", "player_name", "ER", "IP"]])

            st.subheader("🌀 4+ Strikeouts")
            st.dataframe(team_df[team_df["SO"] >= 4][["date", "player_name", "SO", "AB"]])

            st.subheader("🚀 Grand Slams (SLAM)")
            st.dataframe(team_df[team_df.get("SLAM", 0) >= 1][["date", "player_name", "SLAM"]])

# === PLAYER STATS TAB ===
with main_tab4:

    st.header("📋 Miscellaneous Stat Tables")

    def filter_by_date(df):
        return df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))]

    # Load and filter data
    df = pd.read_csv("daily_player_stats_wide.csv", parse_dates=["date"])
    active_df = df[df["roster_slot"] != "BN"].copy()
    active_df = filter_by_date(active_df)

    # Split by player roles
    batting_positions = ["C", "1B", "2B", "3B", "SS", "OF", "UTIL"]  # can expand as needed
    pitching_positions = ["SP", "RP", "P"]

    batters_df = active_df[active_df["roster_slot"].isin(batting_positions)]
    pitchers_df = active_df[active_df["roster_slot"].isin(pitching_positions)]

    # Define stat categories
    batting_stats = ["1B", "2B", "3B", "TB", "SO", "GIDP", "HBP", "BB", "CS", "IBB", "SLAM"]
    fielding_stats = ["PO", "A", "E"]
    pitching_stats = ["PC", "TBF", "RAPP", "1BA", "2BA", "3BA", "BSV", "PICK", "SBA", "BBA"]

    # Summarize by player
    batting_summary = batters_df.groupby("player_name")[batting_stats].sum(min_count=1).reset_index()
    fielding_summary = active_df.groupby("player_name")[fielding_stats].sum(min_count=1).reset_index()
    pitching_summary = pitchers_df.groupby("player_name")[pitching_stats].sum(min_count=1).reset_index()

    # Display tabs
    stat_tab1, stat_tab2, stat_tab3 = st.tabs(["⚾ Batting", "🧤 Fielding", "🔥 Pitching"])
    with stat_tab1:
        st.subheader("🟦 Batting Stats (By Player)")
        st.dataframe(batting_summary, use_container_width=True)
    with stat_tab2:
        st.subheader("🟩 Fielding Stats (By Player)")
        st.dataframe(fielding_summary, use_container_width=True)
    with stat_tab3:
        st.subheader("🟥 Pitching Stats (By Player)")
        st.dataframe(pitching_summary, use_container_width=True)

    st.header("📈 Top Hitters by Z-Score Over Selected Date Range")

    # Load and filter active non-bench hitters
    hitters_df = pd.read_csv("daily_player_stats_wide.csv", parse_dates=["date"])
    hitters_df = hitters_df[(hitters_df["roster_slot"] != "BN") & (hitters_df["AB"] > 0)].copy()
    hitters_df = filter_by_date(hitters_df)

    # Choose which batting stats to include in Z-score
    zscore_stats = ["R", "HR", "RBI", "SB", "AVG", "OPS"]

    # Recalculate rate stats
    hitters_df["AVG"] = hitters_df.apply(lambda r: r["H"] / r["AB"] if r["AB"] > 0 else 0, axis=1)
    hitters_df["OBP"] = hitters_df.apply(lambda r: (r["H"] + r["BB"] + r["HBP"]) / r["PA"] if r["PA"] > 0 else 0, axis=1)
    hitters_df["SLG"] = hitters_df.apply(lambda r: r["TB"] / r["AB"] if r["AB"] > 0 else 0, axis=1)
    hitters_df["OPS"] = hitters_df["OBP"] + hitters_df["SLG"]

    raw_stats = ["R", "H", "HR", "RBI", "SB", "BB", "HBP", "AB", "PA", "TB"]
    grouped = hitters_df.groupby("player_name")[raw_stats].sum(min_count=1).reset_index()

    # Recalculate rate stats again after summing
    grouped["AVG"] = grouped["H"] / grouped["AB"]
    grouped["OBP"] = (grouped["H"] + grouped["BB"] + grouped["HBP"]) / grouped["PA"]
    grouped["SLG"] = grouped["TB"] / grouped["AB"]
    grouped["OPS"] = grouped["OBP"] + grouped["SLG"]

    # Compute z-scores
    z_scores = grouped[zscore_stats].apply(lambda x: (x - x.mean()) / x.std(ddof=0))

    # Weight AVG and OPS z-scores by sqrt(AB) to downweight small samples
    weight = np.sqrt(grouped["AB"]) / np.sqrt(grouped["AB"].max())  # scale from 0 to 1
    z_scores["AVG"] *= weight
    z_scores["OPS"] *= weight

    # Combine all z-scores
    grouped["Z-Score Sum"] = z_scores.sum(axis=1)

    # Get most recent team assignment for each player
    latest_team = (
        hitters_df.sort_values("date")
        .groupby("player_name")["team_name"]
        .last()
        .reset_index()
        .rename(columns={"team_name": "Current Team"})
    )

    # Merge with grouped Z-score table
    top20 = (
        grouped[["player_name", "Z-Score Sum"] + zscore_stats]
        .merge(latest_team, on="player_name", how="left")
        .sort_values(by="Z-Score Sum", ascending=False)
        .head(20)
    )

    st.dataframe(top20.reset_index(drop=True), use_container_width=True)

    st.header("📉 Top Bench Hitters by Z-Score Over Selected Date Range")

    # Load and filter bench hitters
    bench_df = pd.read_csv("daily_player_stats_wide.csv", parse_dates=["date"])
    bench_df = bench_df[(bench_df["roster_slot"] == "BN") & (bench_df["AB"] > 0)].copy()
    bench_df = filter_by_date(bench_df)

    # Stats to use for Z-score ranking
    zscore_stats = ["R", "HR", "RBI", "SB", "AVG", "OPS"]
    raw_stats = ["R", "H", "HR", "RBI", "SB", "BB", "HBP", "AB", "PA", "TB"]

    # Recalculate rate stats
    bench_df["AVG"] = bench_df.apply(lambda r: r["H"] / r["AB"] if r["AB"] > 0 else 0, axis=1)
    bench_df["OBP"] = bench_df.apply(lambda r: (r["H"] + r["BB"] + r["HBP"]) / r["PA"] if r["PA"] > 0 else 0, axis=1)
    bench_df["SLG"] = bench_df.apply(lambda r: r["TB"] / r["AB"] if r["AB"] > 0 else 0, axis=1)
    bench_df["OPS"] = bench_df["OBP"] + bench_df["SLG"]

    # Aggregate per player
    grouped = bench_df.groupby("player_name")[raw_stats].sum(min_count=1).reset_index()
    grouped["AVG"] = grouped["H"] / grouped["AB"]
    grouped["OBP"] = (grouped["H"] + grouped["BB"] + grouped["HBP"]) / grouped["PA"]
    grouped["SLG"] = grouped["TB"] / grouped["AB"]
    grouped["OPS"] = grouped["OBP"] + grouped["SLG"]

    # Compute z-scores
    z_scores = grouped[zscore_stats].apply(lambda x: (x - x.mean()) / x.std(ddof=0))

    # Weight AVG and OPS z-scores by sqrt(AB) to downweight small samples
    weight = np.sqrt(grouped["AB"]) / np.sqrt(grouped["AB"].max())  # scale from 0 to 1
    z_scores["AVG"] *= weight
    z_scores["OPS"] *= weight

    # Combine all z-scores
    grouped["Z-Score Sum"] = z_scores.sum(axis=1)

    # Get latest team for each player
    latest_team = (
        bench_df.sort_values("date")
        .groupby("player_name")["team_name"]
        .last()
        .reset_index()
        .rename(columns={"team_name": "Current Team"})
    )

    # Merge and display
    top20 = (
        grouped[["player_name", "Z-Score Sum"] + zscore_stats]
        .merge(latest_team, on="player_name", how="left")
        .sort_values(by="Z-Score Sum", ascending=False)
        .head(20)
    )

    st.dataframe(top20.reset_index(drop=True), use_container_width=True)

    st.header("📈 Top Pitchers by Z-Score Over Selected Date Range")

    # Load and filter active non-bench hitters
    pitchers_df = pd.read_csv("daily_player_stats_wide.csv", parse_dates=["date"])
    pitchers_df = pitchers_df[(pitchers_df["roster_slot"] != "BN") & (pitchers_df["PC"] > 0)].copy()
    pitchers_df = filter_by_date(pitchers_df)

    # Choose which batting stats to include in Z-score
    zscore_stats = ["K", "ERA", "WHIP", "K%-BB%", "QS", "SVH"]

    # Recalculate rate stats
    pitchers_df["ERA"] = pitchers_df.apply(lambda r: r["ER"] / r["IP"] * 9 if r["IP"] > 0 else 0, axis=1)
    pitchers_df["WHIP"] = pitchers_df.apply(lambda r: (r["HA"] + r["BBA"]) / r["IP"] if r["IP"] > 0 else 0, axis=1)
    pitchers_df["K%"] = pitchers_df.apply(lambda r: r["K"] / r["TBF"] if r["TBF"] > 0 else 0, axis=1)
    pitchers_df["BB%"] = pitchers_df.apply(lambda r: r["BBA"] / r["TBF"] if r["TBF"] > 0 else 0, axis=1)
    pitchers_df["K%-BB%"] = pitchers_df.apply(lambda r: r["K%"] - r["BB%"] if r["TBF"] > 0 else 0, axis=1)

    raw_stats = ["IP", "K", "HA", "BBA", "ER", "QS", "SVH", "TBF"]
    grouped = pitchers_df.groupby("player_name")[raw_stats].sum(min_count=1).reset_index()

    # Recalculate rate stats again after summing
    grouped["ERA"] = grouped["ER"] / grouped["IP"] *9
    grouped["WHIP"] = (grouped["HA"] + grouped["BBA"]) / grouped["IP"]
    grouped["K%"] = grouped["K"] / grouped["TBF"]
    grouped["BB%"] = grouped["BBA"] / grouped["TBF"]
    grouped["K%-BB%"] = grouped["K%"] - grouped["BB%"]

    # Compute z-scores
    z_scores = grouped[zscore_stats].apply(lambda x: (x - x.mean()) / x.std(ddof=0))

    # Adjust ERA and WHIP
    z_scores["ERA"] *= -1
    z_scores["WHIP"] *= -1

    # Weighting factor
    grouped["weight"] = np.sqrt(grouped["TBF"]) / np.sqrt(grouped["TBF"].max())

    # Apply weight to selected z-score columns
    for stat in ["ERA", "WHIP", "K%-BB%"]:
        if stat in z_scores.columns:
            z_scores[stat] *= grouped["weight"]

    # Combine all z-scores
    grouped["Z-Score Sum"] = z_scores.sum(axis=1)

    # Get most recent team assignment for each player
    latest_team = (
        pitchers_df.sort_values("date")
        .groupby("player_name")["team_name"]
        .last()
        .reset_index()
        .rename(columns={"team_name": "Current Team"})
    )

    # Merge with grouped Z-score table
    top20 = (
        grouped[["player_name", "Z-Score Sum"] + zscore_stats]
        .merge(latest_team, on="player_name", how="left")
        .sort_values(by="Z-Score Sum", ascending=False)
        .head(20)
    )

    st.dataframe(top20.reset_index(drop=True), use_container_width=True)

    st.header("📉 Top Bench Pitchers by Z-Score Over Selected Date Range")

    # Load and filter active non-bench hitters
    bench_df = pd.read_csv("daily_player_stats_wide.csv", parse_dates=["date"])
    bench_df = bench_df[(bench_df["roster_slot"] == "BN") & (bench_df["PC"] > 0)].copy()
    bench_df = filter_by_date(bench_df)

    # Choose which batting stats to include in Z-score
    zscore_stats = ["K", "ERA", "WHIP", "K%-BB%", "QS", "SVH"]

    # Recalculate rate stats
    bench_df["ERA"] = bench_df.apply(lambda r: r["ER"] / r["IP"] * 9 if r["IP"] > 0 else 0, axis=1)
    bench_df["WHIP"] = bench_df.apply(lambda r: (r["HA"] + r["BBA"]) / r["IP"] if r["IP"] > 0 else 0, axis=1)
    bench_df["K%"] = bench_df.apply(lambda r: r["K"] / r["TBF"] if r["TBF"] > 0 else 0, axis=1)
    bench_df["BB%"] = bench_df.apply(lambda r: r["BBA"] / r["TBF"] if r["TBF"] > 0 else 0, axis=1)
    bench_df["K%-BB%"] = bench_df.apply(lambda r: r["K%"] - r["BB%"] if r["TBF"] > 0 else 0, axis=1)

    raw_stats = ["IP", "K", "HA", "BBA", "ER", "QS", "SVH", "TBF"]
    grouped = bench_df.groupby("player_name")[raw_stats].sum(min_count=1).reset_index()

    # Recalculate rate stats again after summing
    grouped["ERA"] = grouped["ER"] / grouped["IP"] *9
    grouped["WHIP"] = (grouped["HA"] + grouped["BBA"]) / grouped["IP"]
    grouped["K%"] = grouped["K"] / grouped["TBF"]
    grouped["BB%"] = grouped["BBA"] / grouped["TBF"]
    grouped["K%-BB%"] = grouped["K%"] - grouped["BB%"]

    # Compute z-scores
    z_scores = grouped[zscore_stats].apply(lambda x: (x - x.mean()) / x.std(ddof=0))

    # Adjust ERA and WHIP
    z_scores["ERA"] *= -1
    z_scores["WHIP"] *= -1

    # Weighting factor
    grouped["weight"] = np.sqrt(grouped["TBF"]) / np.sqrt(grouped["TBF"].max())

    # Apply weight to selected z-score columns
    for stat in ["ERA", "WHIP", "K%-BB%"]:
        if stat in z_scores.columns:
            z_scores[stat] *= grouped["weight"]

    # Combine all z-scores
    grouped["Z-Score Sum"] = z_scores.sum(axis=1)

    # Get most recent team assignment for each player
    latest_team = (
        bench_df.sort_values("date")
        .groupby("player_name")["team_name"]
        .last()
        .reset_index()
        .rename(columns={"team_name": "Current Team"})
    )

    # Merge with grouped Z-score table
    top20 = (
        grouped[["player_name", "Z-Score Sum"] + zscore_stats]
        .merge(latest_team, on="player_name", how="left")
        .sort_values(by="Z-Score Sum", ascending=False)
        .head(20)
    )

    st.dataframe(top20.reset_index(drop=True), use_container_width=True)