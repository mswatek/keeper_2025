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

    # === 🧮 Best & Worst Team Days (Raw Stats Ranked by Composite Z-Score) ===

    player_df = pd.read_csv("daily_player_stats_wide.csv", parse_dates=["date"])
    player_df = player_df[(player_df["roster_slot"] != "BN") & 
                        (player_df["date"] >= pd.to_datetime(start_date)) & 
                        (player_df["date"] <= pd.to_datetime(end_date))].copy()

    # --- Recalculate rate stats at player level ---
    player_df["AVG"] = player_df.apply(lambda r: r["H"] / r["AB"] if r["AB"] > 0 else np.nan, axis=1)
    player_df["OBP"] = player_df.apply(lambda r: (r["H"] + r["BB"] + r["HBP"]) / r["PA"] if r["PA"] > 0 else np.nan, axis=1)
    player_df["SLG"] = player_df.apply(lambda r: r["TB"] / r["AB"] if r["AB"] > 0 else np.nan, axis=1)
    player_df["OPS"] = player_df["OBP"] + player_df["SLG"]
    player_df["ERA"] = player_df.apply(lambda r: (r["ER"] * 9) / r["IP"] if r["IP"] > 0 else np.nan, axis=1)
    player_df["WHIP"] = player_df.apply(lambda r: (r["BBA"] + r["HA"]) / r["IP"] if r["IP"] > 0 else np.nan, axis=1)

    # Estimate TBF if not provided
    if "TBF" not in player_df.columns:
        player_df["TBF"] = player_df["IP"] * 3 + player_df["HA"] + player_df["BBA"]

    # --- Aggregate to team-day level ---
    agg_dict = {
        "R": "sum", "HR": "sum", "RBI": "sum", "SB": "sum", "K": "sum",
        "QS": "sum", "SVH": "sum", "PA": "sum",
        "AVG": "mean", "OPS": "mean", "ERA": "mean", "WHIP": "mean",
        "AB": "sum", "IP": "sum", "TBF": "sum", "BBA": "sum"
    }
    team_day = player_df.groupby(["team_name", "date"]).agg(agg_dict).reset_index()

    # Calculate K%-BB% from team totals
    team_day["K%-BB%"] = team_day.apply(
        lambda r: (r["K"] - r["BBA"]) / r["TBF"] if r["TBF"] > 0 else np.nan,
        axis=1
    )

    # List of roto stats for scoring
    zscore_stats = ['R', 'HR', 'RBI', 'SB', 'AVG', 'OPS', 'K', 'ERA', 'WHIP', 'K%-BB%', 'QS', 'SVH']
    team_day = team_day.dropna(subset=zscore_stats)

    # --- Z-score calculation ---
    z_scores = team_day[zscore_stats].apply(lambda x: (x - x.mean()) / x.std(ddof=0))

    # Invert ERA and WHIP (lower is better)
    for stat in ["ERA", "WHIP"]:
        if stat in z_scores.columns:
            z_scores[stat] *= -1

    # Weighting by volume
    weight_AB = np.sqrt(team_day["AB"]) / np.sqrt(team_day["AB"].max())
    weight_IP = np.sqrt(team_day["IP"]) / np.sqrt(team_day["IP"].max())
    weight_TBF = np.sqrt(team_day["TBF"]) / np.sqrt(team_day["TBF"].max())

    if "AVG" in z_scores:       z_scores["AVG"] *= weight_AB
    if "OPS" in z_scores:       z_scores["OPS"] *= weight_AB
    if "ERA" in z_scores:       z_scores["ERA"] *= weight_IP
    if "WHIP" in z_scores:      z_scores["WHIP"] *= weight_IP
    if "K%-BB%" in z_scores:    z_scores["K%-BB%"] *= weight_TBF

    # Composite z-score
    team_day["z_total"] = z_scores.sum(axis=1)

    # Find best and worst days
    best_days = team_day.loc[team_day.groupby("team_name")["z_total"].idxmax()].sort_values("z_total", ascending=False)
    
    # Apply PA threshold for worst day evaluation
    team_day["PA"] = player_df.groupby(["team_name", "date"])["PA"].sum().reindex(team_day.set_index(["team_name", "date"]).index).values
    eligible_for_worst = team_day[team_day["PA"] >= 30].copy()

    # Now identify worst days only among those that meet the PA threshold
    worst_days = eligible_for_worst.loc[eligible_for_worst.groupby("team_name")["z_total"].idxmin()].sort_values("z_total")

    display_cols = ["team_name", "date", "PA", "R", "HR", "RBI", "SB", "AVG", "OPS", "IP", "K", "ERA", "WHIP", "K%-BB%", "QS", "SVH"]

    # --- Display raw stats, ranked by z_total ---
    st.header("💫 Best and Worst Team Days (Raw Stats Ranked by Composite Z-Score)")

    tabs = st.tabs(["📈 Best Day Per Team", "📉 Worst Day Per Team"])

    with tabs[0]:
        st.subheader("📈 Best Day Per Team")
        st.dataframe(best_days[display_cols].reset_index(drop=True), use_container_width=True)

    with tabs[1]:
        st.subheader("📉 Worst Day Per Team (Min 30 PA)")
        st.dataframe(worst_days[display_cols].reset_index(drop=True), use_container_width=True)

    st.header("🏅 Team Highlights")

    # Load & clean data
    df = pd.read_csv("daily_player_stats_wide.csv", parse_dates=["date"])
    df = filter_by_date(df)

    # Ensure numeric columns
    for col in ["IP", "ER", "SO", "QS", "SVH", "BSV", "SLAM", "HR", "SB", "H", "PC", "K", "AB"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    teams = sorted(df["team_name"].unique())
    selected_team = st.selectbox("Select a team:", teams)

    active_df = df[(df["team_name"] == selected_team) & (df["roster_slot"] != "BN")].copy()
    bench_df = df[(df["team_name"] == selected_team) & (df["roster_slot"] == "BN")].copy()

    # =======================
    # Helper function
    # =======================
    def paired_stat(title, active_filter, bench_filter, active_cols, bench_cols=None):
        bench_cols = bench_cols or active_cols
        col1, col2 = st.columns(2)

        try:
            active_filtered = active_df.loc[active_filter].copy()
            bench_filtered = bench_df.loc[bench_filter].copy()

            active_table = active_filtered[["date", "player_name"] + active_cols].dropna(subset=active_cols)
            bench_table = bench_filtered[["date", "player_name"] + bench_cols].dropna(subset=bench_cols)

        except Exception as e:
            st.error(f"Error in '{title}': {e}")
            return

        with col1:
            st.markdown(f"**Active: {title}**")
            if not active_table.empty:
                st.dataframe(active_table.head(100), use_container_width=True)
            else:
                st.info("No results on active roster.")

        with col2:
            st.markdown(f"**Bench: {title}**")
            if not bench_table.empty:
                st.dataframe(bench_table.head(100), use_container_width=True)
            else:
                st.info("No results on bench.")

    # =======================
    # Custom layout blocks
    # =======================

    # 1️⃣ Near-Quality Starts + Bench QS
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Active: Near-Quality Starts (5.667 IP & ≤ 3 ER)**")
        nqs = active_df[(active_df["IP"] == 5.667) & (active_df["ER"] <= 3)][["date", "player_name", "IP", "ER"]]
        if not nqs.empty:
            st.dataframe(nqs.head(100), use_container_width=True)
        else:
            st.info("No near-quality starts on active roster.")
    with col2:
        st.markdown("**Bench: Quality Starts (QS ≥ 1)**")
        qs_bench = bench_df[bench_df["QS"] >= 1][["date", "player_name", "IP", "ER", "QS"]]
        if not qs_bench.empty:
            st.dataframe(qs_bench.head(100), use_container_width=True)
        else:
            st.info("No quality starts on bench.")

    # 2️⃣ Blown Saves + Bench SVH
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Active: Blown Saves (BSV ≥ 1)**")
        bsv = active_df[active_df.get("BSV", 0) >= 1][["date", "player_name", "BSV"]]
        if not bsv.empty:
            st.dataframe(bsv.head(100), use_container_width=True)
        else:
            st.info("No blown saves on active roster.")
    with col2:
        st.markdown("**Bench: Saves + Holds (SVH ≥ 1)**")
        svh = bench_df[bench_df.get("SVH", 0) >= 1][["date", "player_name", "PC", "SVH"]]
        if not svh.empty:
            st.dataframe(svh.head(100), use_container_width=True)
        else:
            st.info("No saves or holds on bench.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Active: Combo Meals (HR + SB)**")
        combo_active = active_df[(active_df["HR"] >= 1) & (active_df["SB"] >= 1)][["date", "player_name", "HR", "SB"]]
        if not combo_active.empty:
            st.dataframe(combo_active.head(100), use_container_width=True)
        else:
            st.info("No combo meals on active roster.")

    with col2:
        st.markdown("**Bench: Combo Meals (HR + SB)**")
        combo_bench = bench_df[(bench_df["HR"] >= 1) & (bench_df["SB"] >= 1)][["date", "player_name", "HR", "SB"]]
        if not combo_bench.empty:
            st.dataframe(combo_bench.head(100), use_container_width=True)
        else:
            st.info("No combo meals on bench.")

    # =======================
    # Core stat comparisons
    # =======================

    paired_stat("Golden Sombreros (4+ SO)",
        active_filter=(active_df["SO"] >= 4),
        bench_filter=(bench_df["SO"] >= 4),
        active_cols=["SO", "AB"]
    )

    paired_stat("Multi-HR Games (2+ HR)",
        active_filter=(active_df["HR"] >= 2),
        bench_filter=(bench_df["HR"] >= 2),
        active_cols=["H", "HR"]
    )

    paired_stat("Grand Slams (SLAM)",
        active_filter=(active_df["SLAM"] >= 1),
        bench_filter=(bench_df["SLAM"] >= 1),
        active_cols=["SLAM"]
    )

    paired_stat("4+ Strikeouts",
        active_filter=(active_df["SO"] >= 4),
        bench_filter=(bench_df["SO"] >= 4),
        active_cols=["SO", "AB"]
    )

    paired_stat("Double-Digit Ks (K ≥ 10)",
        active_filter=(active_df["SO"] >= 10),
        bench_filter=(bench_df["SO"] >= 10),
        active_cols=["K", "IP"]
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Active: 5+ ER Allowed**")
        try:
            er_active = active_df[(active_df["ER"] >= 5)][["date", "player_name", "ER", "IP"]].dropna()
            st.dataframe(er_active.head(100), use_container_width=True)
        except Exception as e:
            st.error(f"Active ER error: {e}")

    with col2:
        st.markdown("**Bench: 5+ ER Allowed**")
        try:
            er_bench = bench_df[(bench_df["ER"] >= 5)][["date", "player_name", "ER", "IP"]].dropna()
            st.dataframe(er_bench.head(100), use_container_width=True)
        except Exception as e:
            st.error(f"Bench ER error: {e}")

    # === Optional Bench Summary (if you still want it) ===
    st.markdown("🧾 **Bench Summary**")
    summary = {
        "HR_SB": ((bench_df["HR"] >= 1) & (bench_df["SB"] >= 1)).sum(),
        "HR2": (bench_df["HR"] >= 2).sum(),
        "QS": bench_df["QS"].fillna(0).astype(int).sum(),
        "SVH": bench_df["SVH"].fillna(0).astype(int).sum(),
        "SLAM": bench_df["SLAM"].fillna(0).astype(int).sum(),
        "ER5": (bench_df["ER"] >= 5).sum(),
        "SO4": (bench_df["SO"] >= 4).sum(),
    }
    st.markdown(
        f"""
    • `{summary['HR_SB']}` HR+SB combos  
    • `{summary['HR2']}` multi-HR games  
    • `{summary['QS']}` Quality Starts  
    • `{summary['SVH']}` Saves+Holds  
    • `{summary['SLAM']}` Grand Slams  
    • `{summary['ER5']}` 5+ ER outings  
    • `{summary['SO4']}` 4+ strikeout games  
    """)

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

    # === Utility: Z-Score Leaderboard Generator ===
    def zscore_leaderboard(df, role, is_bench=False):
        df = df.copy()
        df = filter_by_date(df)

        if role == "hitter":
            df = df[(df["AB"] > 0)]
            raw_stats = ["R", "H", "HR", "RBI", "SB", "BB", "HBP", "AB", "PA", "TB"]
            rate_stats = ["AVG", "OPS"]
            zscore_stats = ["R", "HR", "RBI", "SB", "AVG", "OPS"]

            # Rate stats
            df["AVG"] = df.apply(lambda r: r["H"] / r["AB"] if r["AB"] > 0 else 0, axis=1)
            df["OBP"] = df.apply(lambda r: (r["H"] + r["BB"] + r["HBP"]) / r["PA"] if r["PA"] > 0 else 0, axis=1)
            df["SLG"] = df.apply(lambda r: r["TB"] / r["AB"] if r["AB"] > 0 else 0, axis=1)
            df["OPS"] = df["OBP"] + df["SLG"]

            grouped = df.groupby("player_name")[raw_stats].sum(min_count=1).reset_index()
            grouped["AVG"] = grouped["H"] / grouped["AB"]
            grouped["OBP"] = (grouped["H"] + grouped["BB"] + grouped["HBP"]) / grouped["PA"]
            grouped["SLG"] = grouped["TB"] / grouped["AB"]
            grouped["OPS"] = grouped["OBP"] + grouped["SLG"]

            # Z-score
            z = grouped[zscore_stats].apply(lambda x: (x - x.mean()) / x.std(ddof=0))
            weight = np.sqrt(grouped["AB"]) / np.sqrt(grouped["AB"].max())
            z["AVG"] *= weight
            z["OPS"] *= weight

        else:  # pitcher
            df = df[(df["PC"] > 0)]
            raw_stats = ["IP", "K", "HA", "BBA", "ER", "QS", "SVH", "TBF"]
            zscore_stats = ["K", "ERA", "WHIP", "K%-BB%", "QS", "SVH"]

            df["ERA"] = df.apply(lambda r: r["ER"] / r["IP"] * 9 if r["IP"] > 0 else 0, axis=1)
            df["WHIP"] = df.apply(lambda r: (r["HA"] + r["BBA"]) / r["IP"] if r["IP"] > 0 else 0, axis=1)
            df["K%"] = df.apply(lambda r: r["K"] / r["TBF"] if r["TBF"] > 0 else 0, axis=1)
            df["BB%"] = df.apply(lambda r: r["BBA"] / r["TBF"] if r["TBF"] > 0 else 0, axis=1)
            df["K%-BB%"] = df["K%"] - df["BB%"]

            grouped = df.groupby("player_name")[raw_stats].sum(min_count=1).reset_index()
            grouped["ERA"] = grouped["ER"] / grouped["IP"] * 9
            grouped["WHIP"] = (grouped["HA"] + grouped["BBA"]) / grouped["IP"]
            grouped["K%"] = grouped["K"] / grouped["TBF"]
            grouped["BB%"] = grouped["BBA"] / grouped["TBF"]
            grouped["K%-BB%"] = grouped["K%"] - grouped["BB%"]

            z = grouped[zscore_stats].apply(lambda x: (x - x.mean()) / x.std(ddof=0))
            z["ERA"] *= -1
            z["WHIP"] *= -1

            weight = np.sqrt(grouped["TBF"]) / np.sqrt(grouped["TBF"].max())
            for stat in ["ERA", "WHIP", "K%-BB%"]:
                z[stat] *= weight

        grouped["Z-Score Sum"] = z.sum(axis=1)

        # Latest team info
        latest_team = (
            df.sort_values("date")
            .groupby("player_name")["team_name"]
            .last()
            .reset_index()
            .rename(columns={"team_name": "Current Team"})
        )

        result = (
            grouped[["player_name", "Z-Score Sum"] + zscore_stats]
            .merge(latest_team, on="player_name", how="left")
            .sort_values(by="Z-Score Sum", ascending=False)
            .head(20)
            .reset_index(drop=True)
        )

        return result

    # === Tabbed Layout for Z-Score Tables ===
    tabs = st.tabs([
        "📈 Top Hitters",
        "📉 Bench Hitters",
        "💪 Top Pitchers",
        "🛋️ Bench Pitchers"
    ])

    with tabs[0]:
        df = pd.read_csv("daily_player_stats_wide.csv", parse_dates=["date"])
        df = df[df["roster_slot"] != "BN"]
        st.dataframe(zscore_leaderboard(df, role="hitter"), use_container_width=True)

    with tabs[1]:
        df = pd.read_csv("daily_player_stats_wide.csv", parse_dates=["date"])
        df = df[df["roster_slot"] == "BN"]
        st.dataframe(zscore_leaderboard(df, role="hitter", is_bench=True), use_container_width=True)

    with tabs[2]:
        df = pd.read_csv("daily_player_stats_wide.csv", parse_dates=["date"])
        df = df[df["roster_slot"] != "BN"]
        st.dataframe(zscore_leaderboard(df, role="pitcher"), use_container_width=True)

    with tabs[3]:
        df = pd.read_csv("daily_player_stats_wide.csv", parse_dates=["date"])
        df = df[df["roster_slot"] == "BN"]
        st.dataframe(zscore_leaderboard(df, role="pitcher", is_bench=True), use_container_width=True)
