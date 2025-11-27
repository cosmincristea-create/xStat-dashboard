import streamlit as st
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="xStatistica: Team Data Explorer", 
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. STARTUP & LOADING FEEDBACK
# -----------------------------------------------------------------------------
status_text = st.empty()
progress_bar = st.progress(0)

try:
    status_text.text("booting_system > Importing tactical libraries...")
    
    # --- IMPORTS ---
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import seaborn as sns
    from mplsoccer import PyPizza, Pitch
    from scipy import stats
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    import math
    
    # [STYLE UPGRADE] Activate Dark Background globally for Matplotlib
    plt.style.use('dark_background')
    
    progress_bar.progress(25)

    # --- IBM CARBON DESIGN SYSTEM (DARK MODE) ---
    IBM_DARK_BG = '#161616'     # Main Background
    IBM_LIGHT_BG = '#262626'    # Card/Widget Background
    IBM_TEXT = '#F4F4F4'        # Main Text
    IBM_BLUE = '#0F62FE'        # Action/Good (Above Avg)
    IBM_MAGENTA = '#F50E68'     # Highlight
    IBM_TEAL = '#009D9A'        # Secondary
    IBM_RED = '#DA1E28'         # Bad/Warning (Below Avg)
    IBM_GREEN = '#24A148'       # Success (Elite)
    IBM_YELLOW = '#F1C21B'      # Average
    
    # Palette for distinct H2H verdict colors
    H2H_COLORS = [IBM_GREEN, IBM_MAGENTA, IBM_TEAL, IBM_YELLOW, '#FF8C00']
    
    # [CRITICAL FIX] Define 'background' globally
    background = IBM_DARK_BG
    
    # --- GLOBAL CSS OVERRIDES (NEW FONT: INTER) ---
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        
        .stApp {{
            background-color: {IBM_DARK_BG};
            color: {IBM_TEXT};
            font-family: 'Inter', sans-serif;
        }}
        
        h1, h2, h3, h4, h5, h6 {{
            color: {IBM_TEXT} !important;
            font-family: 'Inter', sans-serif;
            font-weight: 700;
            letter-spacing: -0.5px;
        }}
        
        p, li, label, div {{
            font-family: 'Inter', sans-serif;
            color: {IBM_TEXT};
        }}
        
        [data-testid="stSidebar"] {{
            background-color: {IBM_LIGHT_BG};
            border-right: 1px solid #393939;
        }}
        
        div[data-testid="stMetricValue"] {{
            color: {IBM_BLUE} !important;
            font-weight: 700;
        }}
        
        /* Tab Styling */
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {{
            font-size: 15px;
            color: #A0A0A0; 
            font-weight: 600 !important; 
        }}

        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
            background-color: transparent !important;
            border-top: 3px solid {IBM_RED} !important;
        }}
        
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] [data-testid="stMarkdownContainer"] p {{
            color: {IBM_TEXT} !important;
            font-weight: 700 !important;
        }}

        .stTabs [data-baseweb="tab-list"] button:hover [data-testid="stMarkdownContainer"] p {{
            color: {IBM_RED} !important;
        }}
        
        [data-testid="stDataFrame"] {{
            background-color: {IBM_LIGHT_BG};
        }}
        
        .info-box {{
            background-color: #262626;
            padding: 15px;
            border-left: 5px solid {IBM_BLUE};
            border-radius: 4px;
            margin-bottom: 15px;
        }}
        </style>
    """, unsafe_allow_html=True)
    
    progress_bar.progress(50)

    # --- DATA LOADING ---
    @st.cache_data
    def load_data():
        file_path = 'Romanian Liga 1 25-26.csv' 
        return pd.read_csv(file_path)

    status_text.text("booting_system > Scouting the players (Loading CSV)...")
    
    try:
        df = load_data()
        progress_bar.progress(75)
        
        # Ensure Date is datetime for Trend Analysis sorting
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        team_df = df.groupby('Team')[numeric_cols].mean().reset_index()
    except Exception as e:
        st.error(f"❌ Error processing data: {e}")
        st.stop()

    progress_bar.progress(100)
    status_text.empty()
    progress_bar.empty()

    # --- METRIC DEFINITIONS ---
    metrics_db = {
        'xG': [True, "Expected Goals. The probability that a shot will result in a goal based on the characteristics of that shot (distance, angle, body part). (Source: Opta)"],
        'xGA': [False, "Expected Goals Against. The sum of xG of all shots conceded. A measure of defensive strength (lower is better). (Source: Opta)"],
        'xGD': [True, "Expected Goal Difference. Calculated as xG minus xGA. Positive values indicate dominance. (Source: Opta)"],
        'Possession': [True, "The percentage of time the team possessed the ball relative to the total time the ball was in play. (Source: Wyscout)"],
        'Field Tilt': [True, "The share of a team's passes made in the attacking third compared to the opponent's. Measures territorial dominance. (Source: Wyscout)"],
        'PPDA': [False, "Passes Per Defensive Action. A metric for pressing intensity. It counts opponent passes allowed per defensive action (tackle, interception, foul) in the attacking 60% of the pitch. Lower values = More intense pressing. (Source: Wyscout)"],
        'Passes in Opposition Half': [True, "Total number of completed passes performed in the opponent's half. Indicates active possession rather than passive circulation. (Source: Wyscout)"],
        'Game Control': [True, "A composite metric derived by Ben Griffis indicating overall match dominance based on possession, territory (Field Tilt), and chance creation (xG)."],
        'xT': [True, "Expected Threat. Measures how much a team's actions (passes/carries) increase the probability of a goal by moving the ball to more dangerous zones. (Source: Socceraction)"],
        'High Recoveries': [True, "Ball recoveries made within the attacking third of the pitch (High Zone). A key indicator of effective high pressing. (Source: Wyscout)"],
        'Shots': [True, "Total attempts to score, including shots on target, off target, and blocked. (Source: Opta)"],
        'Shots Faced': [False, "Total shots allowed to the opponent. (Source: Opta)"],
        'Avg Pass Height': [False, "Average height of passes (meters). Lower values indicate ground-based/tiki-taka play; higher values indicate aerial/direct play. (Source: Wyscout)"],
        'Crosses': [True, "Total crosses attempted. Indicates reliance on wing play. (Source: Wyscout)"],
        'Passes into Box': [True, "Completed passes into the opponent's 18-yard box (Deep Completions). A strong proxy for creating danger. (Source: Wyscout)"],
        'Corners': [True, "Total corner kicks won. (Source: Opta)"],
        'Fouls': [False, "Total fouls committed. High values may indicate a disjointed defense or a tactical strategy to break play. (Source: Opta)"],
        'Offsides': [False, "Times caught offside. High values may indicate aggressive forward runs or poor timing. (Source: Opta)"],
        'Aerial Duels Won %': [True, "Percentage of aerial duels won. Indicates physical dominance in the air. (Source: Wyscout)"],
        'Challenge Intensity': [True, "Duels, tackles and interceptions per minute of opponent possession. Higher = More aggressive defending. (Source: Wyscout)"],
        'Clearances': [True, "Defensive actions to clear the ball from danger. High values often correlate with teams that sit deep. (Source: Opta)"],
        'Interceptions': [True, "Reading of the game to cut off opponent passes. (Source: Opta)"]
    }

    # -----------------------------------------------------------------------------
    # 4. MAIN APPLICATION
    # -----------------------------------------------------------------------------
    
    st.title("xStatistica: Team Data Explorer")
    
    st.sidebar.title("App Controls")
    st.sidebar.markdown("---")
    team_list = sorted(team_df['Team'].unique())
    selected_team = st.sidebar.selectbox("Select Focus Team", team_list, key="sb_team_select")
    
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    st.sidebar.markdown("---")
    
    with st.sidebar.expander("ℹ️ Disclaimer & Credits"):
        st.markdown("""
        * All the data for this app comes from **Ben Griffis** via his GitHub Repo.
        * If you want to understand Ben's incredible work and thought process, he explains his xPts model [here](https://cafetactiques.com/2023/04/15/creating-an-expected-points-model-inspired-by-pythagorean-expectation/).
        * This app takes inspiration from Ben's Match Report app, which you can find [here](https://football-match-reports.streamlit.app/).
        """)

    # --- TABS ---
    tab_overview, tab_trend, tab_ai, tab_profiler, tab_h2h, tab_match, tab_event, tab_zscore, tab_master, tab_glossary = st.tabs([
        "Tactical Landscape",
        "📈 Season Trends", 
        "Tactical Engine",
        "Team Profiler", 
        "Head to Head", 
        "Match Inspector",
        "Event Lab",
        "Z-Score Benchmark", 
        "Master Table",
        "Glossary"
    ])

    # === TAB 1: OVERVIEW SCATTER ===
    with tab_overview:
        st.subheader("League Tactical Landscape")
        c1, c2 = st.columns(2)
        with c1: x_metric = st.selectbox("X-Axis", numeric_cols, index=numeric_cols.index('Field Tilt') if 'Field Tilt' in numeric_cols else 0, key="t1_x")
        with c2: y_metric = st.selectbox("Y-Axis", numeric_cols, index=numeric_cols.index('PPDA') if 'PPDA' in numeric_cols else 1, key="t1_y")

        plot_data = team_df.copy()
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor(IBM_DARK_BG)
        ax.set_facecolor(IBM_DARK_BG)
        
        ax.grid(True, linestyle=':', alpha=0.3, color='#555555', zorder=0)
        
        avg_x = plot_data[x_metric].mean()
        avg_y = plot_data[y_metric].mean()
        ax.axvline(avg_x, color='#888888', linestyle='--', alpha=0.6, linewidth=1, zorder=1)
        ax.axhline(avg_y, color='#888888', linestyle='--', alpha=0.6, linewidth=1, zorder=1)
        
        sns.scatterplot(data=plot_data, x=x_metric, y=y_metric, color=IBM_BLUE, s=150, alpha=0.7, edgecolor=IBM_DARK_BG, ax=ax, zorder=2)
        
        focus = plot_data[plot_data['Team'] == selected_team]
        if not focus.empty:
            ax.scatter(focus[x_metric], focus[y_metric], color=IBM_MAGENTA, s=350, edgecolors='white', linewidth=2.5, zorder=4)
            ax.text(focus[x_metric].values[0], focus[y_metric].values[0], f"  {selected_team}", 
                    color=IBM_MAGENTA, fontweight='bold', fontsize=14, va='center', zorder=5)

        for i, row in plot_data.iterrows():
            if row['Team'] != selected_team:
                ax.text(row[x_metric], row[y_metric], f"  {row['Team']}", fontsize=9, alpha=0.5, color='#CCCCCC', zorder=3)

        for spine in ['top', 'right']: ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']: ax.spines[spine].set_color('white'); ax.spines[spine].set_alpha(0.5)
        ax.tick_params(colors='#CCCCCC', labelsize=10)

        if 'PPDA' in y_metric or 'Against' in y_metric:
            ax.invert_yaxis()
            y_label = f"{y_metric} (Inverted)"
        else:
            y_label = y_metric
            
        ax.set_ylabel(y_label, color='white', fontweight='bold', fontsize=12)
        ax.set_xlabel(x_metric, color='white', fontweight='bold', fontsize=12)
        ax.set_title(f"{y_metric} vs. {x_metric}", fontsize=20, fontweight='bold', color='white', loc='left', pad=20)
        
        st.pyplot(fig)

    # === TAB 2: SEASON TRENDS (UPDATED) ===
    with tab_trend:
        st.subheader(f"Season Trend Analysis: {selected_team}")
        
        # --- ANALYST EXPLAINER ---
        st.markdown("""
        ### 📈 Performance Trajectory: Reading the Lines
        This visualisation tracks the team's underlying performance across the season, match by match. 
        
        * **The Magenta Line (xG Created):** Represents the quality of chances created. A rising line indicates attacking fluidity.
        * **The Blue Line (xGA Conceded):** Represents defensive vulnerability. Ideally, this line should remain low and flat.
        * **The Gap:** Look for the **"Jaws of Dominance"**—where the Magenta line rises significantly above the Blue line. Conversely, if the Blue line eclipses the Magenta, the team is conceding more quality chances than they are creating, regardless of the match result. This helps identify momentum shifts and dips in form against specific opponents.
        """)
        st.markdown("---")
        
        # Get team matches sorted by date
        trend_df = df[df['Team'] == selected_team].sort_values(by='Date')
        
        if not trend_df.empty:
            fig, ax = plt.subplots(figsize=(14, 7)) # Increased width for match labels
            fig.patch.set_facecolor(IBM_DARK_BG)
            ax.set_facecolor(IBM_DARK_BG)
            
            # Create a sequence index for the X axis (0, 1, 2, 3...) so steps are even
            x_seq = range(len(trend_df))
            
            # Use 'Step' plot
            ax.step(x_seq, trend_df['xG'], where='mid', label='xG (Created)', color=IBM_MAGENTA, linewidth=3)
            ax.step(x_seq, trend_df['xGA'], where='mid', label='xGA (Conceded)', color=IBM_BLUE, linewidth=3, linestyle='--')
            
            # Fill area between
            ax.fill_between(x_seq, trend_df['xG'], trend_df['xGA'], alpha=0.15, color='gray', step='mid')
            
            # Set X-Ticks to be the Match Names
            ax.set_xticks(x_seq)
            ax.set_xticklabels(trend_df['Match'], rotation=45, ha='right', fontsize=9, color='#CCCCCC')
            
            ax.set_title(f"Expected Goals (xG) vs Expected Goals Against (xGA)", color='white', fontweight='bold', loc='left', fontsize=16)
            ax.legend(facecolor=IBM_DARK_BG, labelcolor='white', loc='upper left')
            ax.grid(True, axis='y', linestyle=':', alpha=0.3)
            
            # Spines
            for spine in ['top', 'right']: ax.spines[spine].set_visible(False)
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.tick_params(axis='y', colors='white')
            
            st.pyplot(fig)
        else:
            st.warning("No match history data found for this team.")

    # === TAB 3: AI TACTICAL ENGINE ===
    with tab_ai:
        st.subheader("Tactical Cluster Analysis")
        st.markdown("""
        ### Tactical groups uncovered by AI clustering algorithm
        **Think of this engine as an automated tactical style spotter.** While the league table tells you *what* happened, this algorithm tells you *how* it happened. 
        It analyses **14 different metrics** simultaneously to group teams into "tactical styles."
        """)
        st.markdown("---")
        
        cluster_metrics_list = ['xG', 'xGA', 'Possession', 'Field Tilt', 'PPDA', 'Passes in Opposition Half', 'High Recoveries', 'Game Control', 'xT', 'Shots', 'Shots Faced', 'Avg Pass Height', 'Crosses', 'Passes into Box']
        available_cluster_metrics = [m for m in cluster_metrics_list if m in team_df.columns]
        
        with st.expander("⚙️ Engine Configuration (Metrics & Logic)"):
            st.write(f"**Metrics Analysed ({len(available_cluster_metrics)}):** {', '.join(available_cluster_metrics)}")
        
        c_k, c_x, c_y = st.columns(3)
        with c_k: k_clusters_ai = st.slider("Number of Tactical Groups", 2, 5, 3, key="ai_k")
        with c_x: ai_x = st.selectbox("Visual X-Axis", numeric_cols, index=numeric_cols.index('Field Tilt') if 'Field Tilt' in numeric_cols else 0, key="ai_x")
        with c_y: ai_y = st.selectbox("Visual Y-Axis", numeric_cols, index=numeric_cols.index('PPDA') if 'PPDA' in numeric_cols else 1, key="ai_y")

        if available_cluster_metrics:
            X = team_df[available_cluster_metrics].dropna()
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            kmeans = KMeans(n_clusters=k_clusters_ai, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            ai_df = team_df.copy()
            ai_df.loc[X.index, 'ClusterID'] = clusters
            
            cluster_summaries = {}
            X_z = pd.DataFrame(X_scaled, columns=available_cluster_metrics, index=X.index)
            X_z['ClusterID'] = clusters
            z_means = X_z.groupby('ClusterID').mean()
            
            for cid in range(k_clusters_ai):
                traits = z_means.loc[cid].sort_values(ascending=False)
                top_traits = traits.head(3)
                desc_parts = []
                for m, z in top_traits.items():
                    if z > 0.5: desc_parts.append(f"High {m}")
                
                if 'PPDA' in z_means.columns and z_means.loc[cid, 'PPDA'] < -0.5: desc_parts.append("Intense Pressing")
                if 'xGA' in z_means.columns and z_means.loc[cid, 'xGA'] < -0.5: desc_parts.append("Solid Defense")
                if 'Avg Pass Height' in z_means.columns and z_means.loc[cid, 'Avg Pass Height'] < -0.5: desc_parts.append("Short Passing")

                if not desc_parts: label = "Balanced Profile"
                else: label = " & ".join(desc_parts[:2])
                teams_in_cluster = ai_df[ai_df['ClusterID'] == cid]['Team'].tolist()
                cluster_summaries[cid] = {"label": label, "teams": teams_in_cluster}

            cols = st.columns(k_clusters_ai)
            for i in range(k_clusters_ai):
                info = cluster_summaries[i]
                with cols[i]:
                    st.info(f"**Group {i+1}: {info['label']}**")
                    st.markdown(f"*{', '.join(info['teams'])}*")

            st.markdown("---")
            fig, ax = plt.subplots(figsize=(12, 8))
            fig.patch.set_facecolor(IBM_DARK_BG)
            ax.set_facecolor(IBM_DARK_BG)
            ax.grid(True, linestyle=':', alpha=0.3, color='#555555')
            
            for spine in ['top', 'right']: ax.spines[spine].set_visible(False)
            
            sns.scatterplot(data=ai_df, x=ai_x, y=ai_y, hue='ClusterID', palette='viridis', s=200, alpha=0.9, ax=ax, edgecolor=IBM_DARK_BG, legend=False)
            focus = ai_df[ai_df['Team'] == selected_team]
            if not focus.empty:
                ax.scatter(focus[ai_x], focus[ai_y], color=IBM_MAGENTA, s=400, edgecolors='white', linewidth=3, zorder=10)
                ax.text(focus[ai_x].values[0], focus[ai_y].values[0], f"  {selected_team}", color=IBM_MAGENTA, fontweight='bold', fontsize=14, va='center')
            for i, row in ai_df.iterrows():
                if row['Team'] != selected_team:
                    ax.text(row[ai_x], row[ai_y], f"  {row['Team']}", fontsize=10, alpha=0.5, color='#CCCCCC')
            
            if 'PPDA' in ai_y or 'Against' in ai_y:
                ax.invert_yaxis()
                ax.set_ylabel(f"{ai_y} (Inverted)", color='white', fontweight='bold', fontsize=12)
            else:
                ax.set_ylabel(ai_y, color='white', fontweight='bold', fontsize=12)
            ax.set_xlabel(ai_x, color='white', fontweight='bold', fontsize=12)
            ax.set_title(f"Cluster Visualization: {ai_y} vs {ai_x}", color='white', fontweight='bold', fontsize=18, loc='left', pad=20)
            st.pyplot(fig)

    # === TAB 4: TEAM PROFILER ===
    with tab_profiler:
        c_head, c_toggle = st.columns([3, 1])
        with c_head: st.subheader(f"Performance Profile: {selected_team}")
        with c_toggle: display_mode = st.radio("Value Labels:", ["Percentile Rank", "Raw Data"], horizontal=True, key="t2_display_mode")

        default_params = ['xG', 'xGA', 'Possession', 'Field Tilt', 'PPDA', 'Shots', 'High Recoveries', 'Game Control']
        valid_defaults = [p for p in default_params if p in team_df.columns]
        selected_params = st.multiselect("Select Metrics", numeric_cols, default=valid_defaults, key="t2_params")
        
        if len(selected_params) < 3:
            st.warning("⚠️ Select at least 3 metrics.")
        else:
            team_values = team_df[team_df['Team'] == selected_team][selected_params].iloc[0].values.tolist()
            percentiles = []
            for p, val in zip(selected_params, team_values):
                lower_is_better = any(x in p.lower() for x in ['ppda', 'xga', 'conceded', 'against'])
                rank = stats.percentileofscore(team_df[p], val, kind='weak')
                if lower_is_better: percentiles.append(100 - rank)
                else: percentiles.append(rank)
            
            slice_colors = []
            for p in percentiles:
                if p >= 80: slice_colors.append(IBM_GREEN)
                elif p >= 60: slice_colors.append(IBM_BLUE)
                elif p >= 40: slice_colors.append(IBM_YELLOW)
                else: slice_colors.append(IBM_RED)
            
            if display_mode == "Raw Data": values_to_show = [round(v, 2) for v in team_values]
            else: values_to_show = [int(p) for p in percentiles]

            baker = PyPizza(
                params=selected_params, background_color=IBM_DARK_BG, straight_line_color="#444444",
                straight_line_lw=1, last_circle_lw=1, other_circle_lw=1, other_circle_ls="-.",
                inner_circle_size=20, last_circle_color="#444444", other_circle_color="#444444"
            )
            fig, ax = baker.make_pizza(
                percentiles, figsize=(8, 8), param_location=110, slice_colors=slice_colors,
                value_colors=["#000000"] * len(selected_params), value_bck_colors=slice_colors,
                kwargs_slices=dict(edgecolor=IBM_DARK_BG, zorder=2, linewidth=1),
                kwargs_params=dict(color="white", fontsize=11, fontfamily="sans-serif", va="center"),
                kwargs_values=dict(color="#000000", fontsize=11, fontfamily="sans-serif", zorder=3,
                    bbox=dict(edgecolor="#000000", facecolor="#FFFFFF", boxstyle="round,pad=0.2", lw=1))
            )
            if display_mode == "Raw Data":
                for i, text_obj in enumerate(ax.texts[len(selected_params):]): text_obj.set_text(str(values_to_show[i]))

            fig.text(0.515, 0.975, f"{selected_team} - {display_mode}", size=16, ha="center", fontweight="bold", color="white")
            st.pyplot(fig)
            st.caption(f"🟢 Elite (>80%) | 🔵 Good (>60%) | 🟡 Average (>40%) | 🔴 Below Avg (<40%)")

    # === TAB 5: HEAD TO HEAD ===
    with tab_h2h:
        st.subheader("Head to Head Comparison")
        default_teams = [selected_team]
        remaining = [t for t in team_list if t != selected_team]
        if remaining: default_teams.append(remaining[0])
        h2h_teams = st.multiselect("Select Teams (Max 4)", team_list, default=default_teams, key="t3_teams")
        
        if len(h2h_teams) < 2:
            st.info("Select at least 2 teams.")
        else:
            if len(h2h_teams) > 4:
                st.warning("⚠️ Max 4 teams. Showing first 4.")
                h2h_teams = h2h_teams[:4]
                
            avail_defs = [m for m in metrics_db.keys() if m in team_df.columns]
            h2h_metrics = st.multiselect("Select Metrics", numeric_cols, default=avail_defs, key="t3_metrics")
            
            h2h_data = []
            for metric in h2h_metrics:
                row = {"Metric": metric}
                vals = {}
                for t in h2h_teams:
                    v = team_df[team_df['Team'] == t][metric].values[0]
                    row[t] = round(v, 2)
                    vals[t] = v
                
                is_high_good = metrics_db[metric][0] if metric in metrics_db else True
                winner = max(vals, key=vals.get) if is_high_good else min(vals, key=vals.get)
                row["Verdict"] = "Draw" if len(set(vals.values())) == 1 else winner
                row["Definition"] = metrics_db[metric][1] if metric in metrics_db else ""
                h2h_data.append(row)
            
            df_h2h = pd.DataFrame(h2h_data)
            cols = ["Metric"] + h2h_teams + ["Verdict", "Definition"]
            
            h2h_palette = [IBM_BLUE, IBM_MAGENTA, IBM_TEAL, IBM_YELLOW, '#FF8C00']
            team_color_map = {team: h2h_palette[i % len(h2h_palette)] for i, team in enumerate(h2h_teams)}
            
            st.caption("Team Color Legend:")
            legend_cols = st.columns(len(h2h_teams))
            for i, team in enumerate(h2h_teams):
                c = team_color_map[team]
                legend_cols[i].markdown(f"**<span style='color:{c}'>● {team}</span>**", unsafe_allow_html=True)

            def highlight_winner(row):
                styles = [''] * len(row)
                winner = row['Verdict']
                if winner == "Draw": return styles
                winner_color = team_color_map.get(winner, IBM_GREEN)
                for i, col_name in enumerate(row.index):
                    if col_name == winner: styles[i] = f'color: {winner_color}; font-weight: bold'
                    elif col_name == "Verdict": styles[i] = f'color: {winner_color}; font-weight: bold'
                return styles

            st.dataframe(df_h2h[cols].style.apply(highlight_winner, axis=1), use_container_width=True, height=500)

    # === TAB 6: MATCH INSPECTOR ===
    with tab_match:
        st.subheader("Match Specific Analysis")
        match_focus_team = st.selectbox("Select Team to Filter Games", team_list, key="match_team_select")
        available_matches = df[df['Team'] == match_focus_team]['Match'].unique()
        selected_match = st.selectbox("Select Match", available_matches, key="match_select")
        
        if selected_match:
            match_data = df[df['Match'] == selected_match]
            if not match_data.empty and len(match_data) == 2:
                teams_in_match = match_data['Team'].tolist()
                avail_defs = [m for m in metrics_db.keys() if m in match_data.columns]
                match_metrics = st.multiselect("Select Metrics", numeric_cols, default=avail_defs, key="match_metrics")
                
                match_rows = []
                for metric in match_metrics:
                    val_0 = match_data.iloc[0][metric]
                    val_1 = match_data.iloc[1][metric]
                    name_0 = match_data.iloc[0]['Team']
                    name_1 = match_data.iloc[1]['Team']
                    
                    row = {"Metric": metric, name_0: round(val_0, 2), name_1: round(val_1, 2)}
                    is_high_good = metrics_db[metric][0] if metric in metrics_db else True
                    
                    if val_0 == val_1: row["Verdict"] = "Draw"
                    elif is_high_good: row["Verdict"] = name_0 if val_0 > val_1 else name_1
                    else: row["Verdict"] = name_0 if val_0 < val_1 else name_1
                    
                    row["Definition"] = metrics_db[metric][1] if metric in metrics_db else ""
                    match_rows.append(row)
                
                df_match = pd.DataFrame(match_rows)
                cols_match = ["Metric", teams_in_match[0], teams_in_match[1], "Verdict", "Definition"]
                
                h2h_palette = [IBM_BLUE, IBM_MAGENTA]
                team_color_map = {teams_in_match[0]: h2h_palette[0], teams_in_match[1]: h2h_palette[1]}
                
                st.caption("Team Color Legend:")
                legend_cols = st.columns(2)
                for i, team in enumerate(teams_in_match):
                    c = team_color_map[team]
                    legend_cols[i].markdown(f"**<span style='color:{c}'>● {team}</span>**", unsafe_allow_html=True)

                def highlight_match_winner(row):
                    styles = [''] * len(row)
                    winner = row['Verdict']
                    if winner == "Draw": return styles
                    winner_color = team_color_map.get(winner, IBM_GREEN)
                    for i, col_name in enumerate(row.index):
                        if col_name == winner or col_name == "Verdict":
                            styles[i] = f'color: {winner_color}; font-weight: bold'
                    return styles

                st.dataframe(df_match[cols_match].style.apply(highlight_match_winner, axis=1), use_container_width=True, height=500)

    # === TAB 7: EVENT LAB ===
    with tab_event:
        st.subheader("⚽ Event Lab: Visual Analysis")
        st.markdown("**Upload a Wyscout Event CSV (Coordinate Data) to visualise shots.**")
        
        uploaded_event_file = st.file_uploader("Upload Wyscout Event CSV", type=["csv"], key="event_upload")
        
        if uploaded_event_file:
            try:
                events_df = pd.read_csv(uploaded_event_file)
                if 'location.x' in events_df.columns and 'location.y' in events_df.columns:
                    st.success("Events loaded successfully!")
                    shots_df = events_df[events_df['type.primary'] == 'shot'].copy()
                    
                    if not shots_df.empty:
                        teams_in_file = shots_df['team.name'].unique()
                        selected_event_team = st.selectbox("Select Team to Visualize", teams_in_file, key="event_team_sel")
                        team_shots = shots_df[shots_df['team.name'] == selected_event_team]
                        
                        pitch = Pitch(pitch_type='wyscout', pitch_color=IBM_DARK_BG, line_color='white')
                        fig, ax = pitch.draw(figsize=(10, 6))
                        
                        goals = team_shots[team_shots['shot.isGoal'] == True]
                        pitch.scatter(goals['location.x'], goals['location.y'], ax=ax, s=200, c=IBM_GREEN, edgecolors='white', label='Goal')
                        misses = team_shots[team_shots['shot.isGoal'] == False]
                        pitch.scatter(misses['location.x'], misses['location.y'], ax=ax, s=100, c=IBM_RED, alpha=0.6, label='No Goal')
                        
                        ax.legend(facecolor=IBM_DARK_BG, labelcolor='white')
                        ax.set_title(f"Shot Map: {selected_event_team}", color='white', fontsize=16)
                        st.pyplot(fig)
                    else:
                        st.warning("No shot data found in this file.")
                else:
                    st.error("CSV does not contain 'location.x' and 'location.y' columns (Wyscout format required).")
            except Exception as e:
                st.error(f"Error reading file: {e}")

    # === TAB 8: Z-SCORE ===
    with tab_zscore:
        st.subheader("Z-Score Deviation")
        z_defs = [m for m in metrics_db.keys() if m in team_df.columns]
        z_metrics = st.multiselect("Select Metrics", numeric_cols, default=z_defs, key="t4_metrics")
        
        if z_metrics:
            z_data = []
            team_row = team_df[team_df['Team'] == selected_team].iloc[0]
            for m in z_metrics:
                mean = team_df[m].mean()
                std = team_df[m].std()
                z = (team_row[m] - mean) / std
                if any(x in m.lower() for x in ['ppda', 'xga', 'conceded', 'against', 'shots faced']): z *= -1
                z_data.append({'Metric': m, 'Z-Score': z})
            
            z_df = pd.DataFrame(z_data).sort_values(by='Z-Score')
            fig, ax = plt.subplots(figsize=(10, len(z_metrics)*0.6))
            fig.patch.set_facecolor(IBM_DARK_BG)
            ax.set_facecolor(IBM_DARK_BG)
            
            colors = [IBM_GREEN if x >= 0 else IBM_RED for x in z_df['Z-Score']]
            bars = ax.barh(z_df['Metric'], z_df['Z-Score'], color=colors, edgecolor=IBM_DARK_BG)
            
            ax.axvline(0, color='white')
            ax.grid(axis='x', linestyle='--', alpha=0.3, color='gray')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_visible(False)
            ax.tick_params(colors='white')
            
            for bar in bars:
                w = bar.get_width()
                align = 'left' if w > 0 else 'right'
                ax.text(w + (0.1 if w > 0 else -0.1), bar.get_y()+bar.get_height()/2, f'{w:.2f}', 
                        va='center', ha=align, color='white', fontweight='bold', fontsize=9)
            st.pyplot(fig)

    # === TAB 9: MASTER TABLE ===
    with tab_master:
        st.subheader("League Master Table")
        c1, c2 = st.columns(2)
        with c1: m_rank = st.selectbox("Rank By", numeric_cols, index=0, key="t5_rank")
        with c2: order = st.radio("Order", ["Descending", "Ascending"], index=0, key="t5_order")
        
        asc = True if "Ascending" in order else False
        ranked = team_df[['Team'] + numeric_cols].sort_values(by=m_rank, ascending=asc).reset_index(drop=True)
        ranked.index += 1
        
        def style_sel(s): return [f'background-color: {IBM_RED}; color: white; font-weight: bold' if s['Team'] == selected_team else '' for _ in s]
        st.dataframe(ranked.style.apply(style_sel, axis=1).background_gradient(subset=[m_rank], cmap='RdYlGn_r' if asc else 'RdYlGn').format(precision=2), use_container_width=True)

    # === TAB 10: GLOSSARY ===
    with tab_glossary:
        st.subheader("📚 Metric Glossary")
        gloss_df = pd.DataFrame([{"Metric": m, "Definition": d[1]} for m, d in metrics_db.items()])
        def make_bold(val): return "font-weight: 700; color: #0F62FE;"
        st.dataframe(gloss_df.style.map(make_bold, subset=['Metric']), use_container_width=True, hide_index=True)

except Exception as e:
    st.error("🚨 SYSTEM ERROR")
    st.code(str(e))