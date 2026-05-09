import streamlit as st
import json
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Classroom AI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# CUSTOM CSS FOR DARK GLASS MORPHISM
# =========================
st.markdown("""
<style>
    /* Root variables */
    :root {
        --primary: #00d4ff;
        --secondary: #ff006e;
        --accent: #ff9500;
        --success: #00ff88;
        --dark-bg: #0a0e27;
        --card-bg: rgba(20, 33, 61, 0.35);
        --card-border: rgba(0, 212, 255, 0.15);
        --text-primary: #e8f0ff;
        --text-secondary: #a0b0c0;
    }
    
    /* Main container */
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1428 100%);
        color: var(--text-primary);
    }
    
    /* Remove default padding */
    .main > div:first-child {
        padding-top: 0;
    }
    
    /* Glass morphism card effect */
    .glass-card {
        background: var(--card-bg);
        backdrop-filter: blur(10px) saturate(180%);
        border: 1px solid var(--card-border);
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .glass-card:hover {
        background: rgba(20, 33, 61, 0.5);
        border-color: rgba(0, 212, 255, 0.25);
        box-shadow: 0 12px 48px rgba(0, 212, 255, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.08);
        transform: translateY(-2px);
    }
    
    /* Metric container styling */
    [data-testid="metric-container"] {
        background: var(--card-bg) !important;
        backdrop-filter: blur(10px) saturate(180%) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 20px !important;
        padding: 24px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
    }
    
    [data-testid="metric-container"] label {
        color: var(--text-secondary) !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    [data-testid="metric-container"] > div:last-child {
        color: var(--primary) !important;
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
    }
    
    /* Headers */
    h1 {
        color: var(--primary) !important;
        font-size: 2.8rem !important;
        font-weight: 900 !important;
        text-shadow: 0 0 30px rgba(0, 212, 255, 0.25);
        margin-bottom: 0.5rem !important;
        letter-spacing: -1px;
    }
    
    h2 {
        color: var(--primary) !important;
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.2);
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
    }
    
    h3 {
        color: var(--primary) !important;
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 0.8rem !important;
    }
    
    /* Caption */
    .stCaption {
        color: var(--text-secondary) !important;
        font-size: 0.9rem !important;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.2), transparent);
        margin: 2rem 0 !important;
    }
    
    /* Info box */
    .stInfo {
        background: rgba(0, 212, 255, 0.1) !important;
        border: 1px solid rgba(0, 212, 255, 0.3) !important;
        border-radius: 15px !important;
        color: var(--primary) !important;
        padding: 16px 20px !important;
    }
    
    .stWarning {
        background: rgba(255, 149, 0, 0.1) !important;
        border: 1px solid rgba(255, 149, 0, 0.3) !important;
        border-radius: 15px !important;
        color: var(--accent) !important;
        padding: 16px 20px !important;
    }
    
    /* Dataframe styling */
    [data-testid="dataframe"] {
        background: var(--card-bg) !important;
        border-radius: 15px !important;
        border: 1px solid var(--card-border) !important;
    }
    
    /* Plotly charts */
    .plotly-graph-div {
        background: transparent !important;
    }
    
    /* Text styling */
    .stMarkdown {
        color: var(--text-primary) !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(10, 14, 39, 0.7) !important;
        backdrop-filter: blur(10px) !important;
    }
    
    /* Custom text classes */
    .metric-label {
        color: var(--text-secondary);
        font-size: 0.9rem;
        font-weight: 600;
    }
    
    .metric-value {
        color: var(--primary);
        font-size: 2.2rem;
        font-weight: 800;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
    }
    
    /* Engagement status colors */
    .status-attentive {
        color: #00d4ff;
        font-weight: 700;
    }
    
    .status-distracted {
        color: #ff006e;
        font-weight: 700;
    }
    
    .status-engaged {
        color: #00ff88;
        font-weight: 700;
    }
    
    /* Column container */
    [data-testid="column"] {
        padding: 0 8px;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# HELPER FUNCTIONS
# =========================
def load_data():
    """Load student data from data.json"""
    try:
        with open("data.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}

def get_status_color(status):
    """Get status color"""
    colors = {
        "Attentive": "#00d4ff",
        "Distracted": "#ff006e",
        "Engaged": "#00ff88"
    }
    return colors.get(status, "#a0b0c0")

def get_status_emoji(status):
    """Get status emoji"""
    emojis = {
        "Attentive": "👀",
        "Distracted": "😵",
        "Engaged": "✅"
    }
    return emojis.get(status, "❓")

def get_behavior_emoji(behavior):
    """Get behavior emoji"""
    emojis = {
        "reading_writing": "📖",
        "raise_hand": "✋",
        None: "—"
    }
    return emojis.get(behavior, "—")

def calculate_engagement_level(status, behavior):
    """Calculate engagement level"""
    if status == "Engaged":
        if behavior == "reading_writing":
            return "🌟 Active Reading"
        elif behavior == "raise_hand":
            return "🙋 Hand Raised"
        else:
            return "✅ Engaged"
    elif status == "Attentive":
        return "👀 Attentive"
    else:
        return "😴 Distracted"

# =========================
# MAIN APP
# =========================
def main():
    # Header
    col1, col2 = st.columns([0.85, 0.15])
    with col1:
        st.markdown("## 📊 Classroom AI Dashboard")
        st.caption(f"🕐 Last Updated: {datetime.now().strftime('%H:%M:%S')}")
    
    # Load data
    data = load_data()
    
    if not data:
        st.warning("⚠️ No data available currently. Make sure the AI model is running.")
        return
    
    # Get lesson duration
    lesson_duration = next(iter(data.values())).get("lesson_duration", 0)
    lesson_minutes = round(lesson_duration / 60, 1)
    
    # Display lesson duration
    st.info(f"⏱️ **Lesson Duration:** {lesson_minutes} minutes")
    
    # Calculate statistics
    total_students = len(data)
    attentive_count = sum(1 for d in data.values() if d.get("status") == "Attentive")
    distracted_count = sum(1 for d in data.values() if d.get("status") == "Distracted")
    engaged_count = sum(1 for d in data.values() if d.get("status") == "Engaged")
    
    # Summary metrics
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("👥 Total Students", total_students)
    
    with col2:
        st.metric("👀 Attentive", attentive_count)
    
    with col3:
        st.metric("😵 Distracted", distracted_count)
    
    with col4:
        st.metric("✅ Engaged", engaged_count)
    
    st.markdown("---")
    
    # Prepare data for charts
    df_data = []
    for sid, d in data.items():
        df_data.append({
            "Student": f"Student {sid}",
            "Status": d.get("status", "Unknown"),
            "Behavior": d.get("behavior"),
            "Last Seen": d.get("last_seen", 0)
        })
    
    df = pd.DataFrame(df_data)
    
    # Chart 1: Status distribution (Pie Chart)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Status Distribution")
        status_counts = df["Status"].value_counts()
        
        colors_map = {
            "Attentive": "#00d4ff",
            "Distracted": "#ff006e",
            "Engaged": "#00ff88"
        }
        
        fig_pie = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="",
            color_discrete_map=colors_map,
            hole=0.4
        )
        
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e8f0ff', size=12),
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.05,
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(0, 212, 255, 0.2)",
                borderwidth=1
            ),
            margin=dict(l=0, r=150, t=0, b=0)
        )
        
        fig_pie.update_traces(
            textposition='inside',
            textinfo='label+percent',
            marker=dict(line=dict(color='rgba(10, 14, 39, 0.8)', width=2))
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Chart 2: Status bar chart
    with col2:
        st.markdown("### 📊 Number of Students by Status")
        
        fig_bar = px.bar(
            x=status_counts.index,
            y=status_counts.values,
            color=status_counts.index,
            color_discrete_map=colors_map,
            text=status_counts.values,
            title=""
        )
        
        fig_bar.update_traces(
            textposition='outside',
            marker=dict(line=dict(color='rgba(0, 212, 255, 0.3)', width=2))
        )
        
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(20, 33, 61, 0.2)',
            font=dict(color='#e8f0ff', size=12),
            showlegend=False,
            xaxis_title="Status",
            yaxis_title="Number of Students",
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showline=True,
                linewidth=1,
                linecolor='rgba(0, 212, 255, 0.2)'
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(0, 212, 255, 0.1)',
                zeroline=False
            ),
            margin=dict(l=50, r=20, t=20, b=50)
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
    
    st.markdown("---")
    
    # Student details table
    st.markdown("### 📋 Student Details")
    
    table_data = []
    for sid, d in data.items():
        status = d.get("status", "Unknown")
        behavior = d.get("behavior")
        
        table_data.append({
            "🆔 Student": f"Student {sid}",
            "📊 Status": f"{get_status_emoji(status)} {status}",
            "🎯 Behavior": f"{get_behavior_emoji(behavior)} {behavior or 'No Behavior'}",
            "⏱️ Last Seen": datetime.fromtimestamp(d.get("last_seen", 0)).strftime("%H:%M:%S") if d.get("last_seen") else "—",
            "📈 Engagement Level": calculate_engagement_level(status, behavior)
        })
    
    df_table = pd.DataFrame(table_data)
    
    # Display table with custom styling
    st.dataframe(
        df_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "🆔 Student": st.column_config.TextColumn(width="small"),
            "📊 Status": st.column_config.TextColumn(width="medium"),
            "🎯 Behavior": st.column_config.TextColumn(width="medium"),
            "⏱️ Last Seen": st.column_config.TextColumn(width="small"),
            "📈 Engagement Level": st.column_config.TextColumn(width="medium")
        }
    )
    
    st.markdown("---")
    
    # Statistics summary
    st.markdown("### 📈 Statistics Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        attentive_pct = (attentive_count / total_students * 100) if total_students > 0 else 0
        st.metric("👀 Attention Rate", f"{attentive_pct:.1f}%")
    
    with col2:
        distracted_pct = (distracted_count / total_students * 100) if total_students > 0 else 0
        st.metric("😵 Distraction Rate", f"{distracted_pct:.1f}%")
    
    with col3:
        engaged_pct = (engaged_count / total_students * 100) if total_students > 0 else 0
        st.metric("✅ Engagement Rate", f"{engaged_pct:.1f}%")

# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    main()
    
    # Auto-refresh
    time.sleep(2)
    st.rerun()
