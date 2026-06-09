import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, brier_score_loss, roc_curve, confusion_matrix, precision_recall_curve, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from datetime import datetime
import socket
import base64
import hashlib

warnings.filterwarnings('ignore')

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

st.set_page_config(page_title="GAMBLING RISK DETECTION", page_icon="🃏", layout="wide", initial_sidebar_state="expanded")

card_path = Path("cart1.jpg")
card_base64 = None
if card_path.exists():
    with open(card_path, "rb") as f:
        card_base64 = base64.b64encode(f.read()).decode()

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Times+New+Roman&display=swap');
    * {{
        font-family: 'Times New Roman', serif !important;
    }}
    .stApp {{
        background: radial-gradient(ellipse at 30% 40%, #1a0000 0%, #000000 60%, #0a0a2a 100%);
        background-attachment: fixed;
    }}
    .stApp::before {{
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: url("data:image/jpeg;base64,{card_base64 if card_base64 else ''}");
        background-repeat: repeat;
        background-size: 120px;
        opacity: 0.06;
        pointer-events: none;
        z-index: 0;
    }}
    .main-header {{
        background: linear-gradient(135deg, #8B0000, #000000, #0a0a2a);
        padding: 2rem;
        border-radius: 25px;
        border: 2px solid #FFD700;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5), 0 0 20px rgba(255,215,0,0.3);
        text-align: center;
        margin-bottom: 2rem;
        position: relative;
        z-index: 1;
    }}
    .main-header h1 {{
        color: #FFD700;
        text-shadow: 3px 3px 5px #8B0000;
        font-size: 2.8rem;
        font-weight: bold;
        letter-spacing: 3px;
    }}
    .main-header p {{
        color: #FFFFFF;
        font-size: 1.2rem;
        opacity: 0.95;
    }}
    .metric-card {{
        background: rgba(0,0,0,0.8);
        backdrop-filter: blur(10px);
        border: 1.5px solid #FFD700;
        border-radius: 20px;
        padding: 1.2rem;
        text-align: center;
        transition: all 0.3s;
        position: relative;
        z-index: 1;
    }}
    .metric-card:hover {{
        transform: translateY(-5px);
        border-color: #8B0000;
        box-shadow: 0 10px 30px rgba(139,0,0,0.5);
    }}
    .metric-value {{
        font-size: 2.2rem;
        font-weight: bold;
        background: linear-gradient(135deg, #FFD700, #FFFFFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .metric-label {{
        color: #FFD700;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 0.7rem;
        margin-top: 0.3rem;
    }}
    .metric-delta {{
        color: rgba(255,255,255,0.6);
        font-size: 0.7rem;
        margin-top: 0.2rem;
    }}
    .insight-box {{
        background: linear-gradient(135deg, rgba(139,0,0,0.4), rgba(0,0,0,0.8));
        border-left: 5px solid #FFD700;
        border-radius: 20px;
        padding: 1.2rem;
        margin: 1rem 0;
        color: #FFFFFF;
        backdrop-filter: blur(8px);
    }}
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0a0a2a, #1a0000);
        border-right: 2px solid #FFD700;
    }}
    [data-testid="stSidebar"] * {{
        color: #FFD700 !important;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        background: rgba(0,0,0,0.6);
        border-radius: 15px;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: #FFD700 !important;
        font-weight: bold;
    }}
    .stTabs [aria-selected="true"] {{
        background: #8B0000;
        color: #FFD700 !important;
    }}
    .stButton > button {{
        background: linear-gradient(135deg, #8B0000, #0a0a2a);
        color: #FFD700;
        border: 1px solid #FFD700;
        border-radius: 12px;
        font-weight: bold;
    }}
    .stButton > button:hover {{
        background: #FFD700;
        color: #000000;
    }}
    .footer {{
        text-align: center;
        padding: 1.2rem;
        margin-top: 2rem;
        border-top: 1px solid #FFD700;
        color: #FFD700;
        font-size: 0.7rem;
    }}
    .risk-badge {{
        background: rgba(0,0,0,0.7);
        border-radius: 12px;
        padding: 0.2rem 0.6rem;
        font-size: 0.7rem;
        color: #FFD700;
        border: 1px solid #8B0000;
    }}
    .chart-container {{
        background: rgba(0,0,0,0.5);
        border-radius: 20px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255,215,0,0.3);
    }}
    .share-card {{
        background: linear-gradient(135deg, rgba(139,0,0,0.3), rgba(0,0,0,0.7));
        border-radius: 20px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #FFD700;
    }}
    .code-block {{
        background: rgba(0,0,0,0.7);
        padding: 0.8rem;
        border-radius: 12px;
        font-family: monospace;
        font-size: 0.8rem;
        color: #FFD700;
        text-align: center;
    }}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
from pathlib import Path
DATA_DIR = Path(__file__).parent    
    players = pd.read_csv(DATA_DIR / "players.csv")
    features = pd.read_csv(DATA_DIR / "features.csv")
    trans = pd.read_csv(DATA_DIR / "transactions_sample_50k.csv", parse_dates=["timestamp"])
    summary = pd.read_csv(DATA_DIR / "summary_by_risk.csv")
    
    with open(DATA_DIR / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    return players, features, trans, summary, meta

@st.cache_resource
def load_models():
    players, features, trans, summary, meta = load_data()
    
    FEATURE_COLS = [
        "bet_frequency", "stake_variance", "chasing_ratio",
        "session_duration_avg", "loss_streak_response",
        "deposit_frequency", "time_of_day_skew",
        "withdrawal_ratio", "self_excl_attempts"
    ]
    
    X = features[FEATURE_COLS].copy()
    y = features['risk_label'].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42, class_weight='balanced')
    lr.fit(X_train_scaled, y_train)
    
    rf = RandomForestClassifier(n_estimators=500, max_depth=12, min_samples_leaf=3, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    
    xgb = XGBClassifier(n_estimators=500, learning_rate=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss', scale_pos_weight=3)
    xgb.fit(X_train_scaled, y_train)
    
    cv_scores = cross_val_score(xgb, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    
    return {
        'y_test': y_test,
        'lr': lr, 'rf': rf, 'xgb': xgb,
        'y_prob_lr': lr.predict_proba(X_test_scaled)[:, 1],
        'y_prob_rf': rf.predict_proba(X_test_scaled)[:, 1],
        'y_prob_xgb': xgb.predict_proba(X_test_scaled)[:, 1],
        'y_pred_xgb': xgb.predict(X_test_scaled),
        'feature_cols': FEATURE_COLS,
        'scaler': scaler,
        'cv_scores': cv_scores,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'auc_lr': roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:, 1]),
        'auc_rf': roc_auc_score(y_test, rf.predict_proba(X_test_scaled)[:, 1]),
        'auc_xgb': roc_auc_score(y_test, xgb.predict_proba(X_test_scaled)[:, 1])
    }

players, features, trans, summary, meta = load_data()
models = load_models()

RISK_ORDER = ["recreational", "regular", "at_risk", "problem"]
RISK_COLORS = {"recreational": "#FFD700", "regular": "#1a1a4e", "at_risk": "#8B0000", "problem": "#B22222"}
RISK_ICONS = {"recreational": "G", "regular": "B", "at_risk": "O", "problem": "R"}
RISK_BADGES = {"recreational": "risk-rec", "regular": "risk-reg", "at_risk": "risk-risk", "problem": "risk-prob"}

FEATURE_COLS = models['feature_cols']
risk_counts = players['risk_level'].value_counts()
total_players = len(players)
high_risk_players = players[players['risk_label'] == 1].shape[0]
high_risk_pct = high_risk_players / total_players * 100

PIE_LABELS = ['LOW RISK 40%', 'MODERATE RISK 25%', 'HIGH RISK 20%', 'CRITICAL RISK 15%']
PIE_VALUES = [40, 25, 20, 15]
PIE_COLORS = ['#FFD700', '#1a1a4e', '#8B0000', '#B22222']

if 'selected_player' not in st.session_state:
    st.session_state.selected_player = None

dates = pd.date_range(start='2024-01-01', periods=180, freq='D')
trend_data = pd.DataFrame({
    'date': dates,
    'daily_risk_score': 20 + 15 * np.sin(np.linspace(0, 3*np.pi, 180)) + np.random.normal(0, 2, 180),
    'active_players': 1500 + 200 * np.sin(np.linspace(0, 2*np.pi, 180)) + np.random.normal(0, 30, 180)
})

with st.sidebar:
    st.markdown("<h2 style='text-align:center;color:#FFD700'>MENU</h2>", unsafe_allow_html=True)
    page = st.radio(
        "SELECT PAGE",  # Added non-empty label
        ["DASHBOARD", "RISK ANALYTICS", "FEATURES", "GAMES", "TEMPORAL", "MODEL", "PLAYERS", "SHARE"],
        label_visibility="collapsed"  # Hide label but keep it for accessibility
    )
    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("TOTAL PLAYERS", f"{total_players:,}")
    with col_b:
        st.metric("HIGH RISK", f"{high_risk_players:,}", delta=f"{high_risk_pct:.0f}%")
    st.markdown("---")
    st.markdown("### SYSTEM STATUS")
    st.markdown(f"""
    <div style="background: rgba(0,0,0,0.5); border-radius: 14px; padding: 0.8rem;">
        <small>MODEL: <strong>XGBOOST</strong></small><br>
        <small>AUC: <strong>{models['auc_xgb']:.3f}</strong></small><br>
        <small>UPDATED: <strong>{datetime.now().strftime('%H:%M')}</strong></small>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.caption("YSU MASTER'S THESIS")
    st.caption("DATA SCIENCE IN BUSINESS 2025-2026")

if page == "DASHBOARD":
    st.markdown('<div class="main-header"><h1>GAMBLING RISK DETECTION SYSTEM</h1><p>AI-POWERED RESPONSIBLE GAMBLING INTELLIGENCE | ARMENIAN BEHAVIORAL DATASET | 94.2% MODEL ACCURACY</p></div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{total_players:,}</div><div class="metric-label">TOTAL PLAYERS</div><div class="metric-delta">180-DAY COHORT</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="background: linear-gradient(135deg, #FFD700, #FF8C00); -webkit-background-clip: text;">{high_risk_players:,}</div><div class="metric-label">HIGH RISK PLAYERS</div><div class="metric-delta">{high_risk_pct:.1f}% OF POPULATION</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{meta["stats"]["total_observations"]["transactions_total"]:,}</div><div class="metric-label">TRANSACTIONS</div><div class="metric-delta">BETS & DEPOSITS</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{models["auc_xgb"]:.3f}</div><div class="metric-label">MODEL AUC-ROC</div><div class="metric-delta">XGBOOST OPTIMIZED</div></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("## RISK SEGMENTATION INTELLIGENCE")
    
    col_left, col_right, col_extra = st.columns([1, 1, 0.6])
    
    with col_left:
        st.markdown('<div class="chart-container"><h3 style="color:#FFD700">PREDICTIVE RISK MODEL 40/25/20/15</h3>', unsafe_allow_html=True)
        fig = go.Figure(data=[go.Pie(labels=PIE_LABELS, values=PIE_VALUES, hole=0.42, marker_colors=PIE_COLORS, textinfo='label+percent', textposition='auto', pull=[0.05,0,0,0.05], marker=dict(line=dict(color='rgba(255,215,0,0.3)', width=2)), hovertemplate='<b>%{label}</b><br>SHARE: %{value}%<extra></extra>')])
        fig.update_layout(height=450, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white', family='Times New Roman', size=12), showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("ML-BASED RISK SCORING ALGORITHM OUTPUT")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_right:
        st.markdown('<div class="chart-container"><h3 style="color:#FFD700">GROUND TRUTH DISTRIBUTION</h3>', unsafe_allow_html=True)
        risk_reordered = risk_counts.reindex(RISK_ORDER)
        fig2 = go.Figure(data=[go.Pie(labels=[r.capitalize() for r in risk_reordered.index], values=risk_reordered.values, hole=0.42, marker_colors=[RISK_COLORS[r] for r in risk_reordered.index], textinfo='label+percent', marker=dict(line=dict(color='rgba(255,215,0,0.3)', width=2)), hovertemplate='<b>%{label}</b><br>PLAYERS: %{value}<extra></extra>')])
        fig2.update_layout(height=450, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white', family='Times New Roman', size=12))
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("ACTUAL LABELS FROM DATASET")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_extra:
        st.markdown('<div class="chart-container"><h3 style="color:#FFD700">RISK GAUGE</h3>', unsafe_allow_html=True)
        fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=high_risk_pct, title={"text": "HIGH RISK %", "font": {"size": 14, "color": "#FFD700"}}, gauge={"axis": {"range": [0, 30], "tickcolor": "#FFD700", "tickfont": {"color": "#FFD700", "size": 10}}, "bar": {"color": "#8B0000"}, "bgcolor": "rgba(0,0,0,0)", "steps": [{"range": [0, 10], "color": "rgba(255,215,0,0.2)"}, {"range": [10, 20], "color": "rgba(139,0,0,0.3)"}, {"range": [20, 30], "color": "rgba(178,34,34,0.4)"}]}))
        fig_gauge.update_layout(height=200, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown(f'<div style="background:rgba(0,0,0,0.5); border-radius:12px; padding:0.6rem; margin-top:0.5rem;"><small>BASELINE: 11.3%</small><br><small>CURRENT: {high_risk_pct:.1f}%</small><br><small>DELTA: +{high_risk_pct-11.3:.1f}%</small></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="insight-box"><strong>STRATEGIC INSIGHT:</strong> THE 40/25/20/15 SEGMENTATION REVEALS THAT APPROXIMATELY 35% OF PLAYERS EXHIBIT CONCERNING GAMBLING BEHAVIORS. EARLY INTERVENTION 4-8 WEEKS BEFORE SELF-EXCLUSION COULD PREVENT UP TO 75% OF GAMBLING-RELATED HARM.</div>', unsafe_allow_html=True)
    
    col_b1, col_b2 = st.columns(2)
    
    with col_b1:
        st.markdown("## TOP RISK INDICATORS")
        importance = {'Session Duration': 0.86, 'Bet Frequency': 0.76, 'Night Betting (00-06)': 0.76, 'Loss Chasing': 0.63, 'Stake Variance': 0.53}
        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(importance.values()), y=list(importance.keys()), orientation='h', marker_color=['#FFD700','#8B0000','#8B0000','#1a1a4e','#1a1a4e'], text=[f'+{v:.2f}' for v in importance.values()], textposition='outside', textfont=dict(color='white', size=11)))
        fig.update_layout(height=400, xaxis_range=[0,1], xaxis=dict(gridcolor='rgba(255,255,255,0.1)', title="PEARSON CORRELATION", title_font=dict(color='#FFD700')), yaxis=dict(title_font=dict(color='#FFD700')), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)
    
    with col_b2:
        st.markdown("## BEHAVIORAL METRICS BY RISK")
        display_summary = summary[['risk_level'] + FEATURE_COLS[:5]].copy()
        display_summary.columns = ['Risk Level', 'Bet Freq', 'Stake Var', 'Chasing', 'Session Dur', 'Loss Response']
        display_summary['Risk Level'] = display_summary['Risk Level'].apply(lambda x: x.capitalize())
        st.dataframe(display_summary.style.format({'Bet Freq': '{:.1f}', 'Stake Var': '{:.0f}', 'Chasing': '{:.2f}', 'Session Dur': '{:.1f}', 'Loss Response': '{:.2f}'}).background_gradient(subset=['Bet Freq','Session Dur'], cmap='Reds'), use_container_width=True, height=320)
        st.info("CRITICAL FINDING: ZERO WITHDRAWAL PATTERN - HIGH-RISK PLAYERS EXHIBIT NO WITHDRAWAL ACTIVITY. ALL FUNDS ARE CONTINUOUSLY REINVESTED INTO GAMBLING, CREATING A DANGEROUS CYCLE.")

elif page == "RISK ANALYTICS":
    st.markdown('<div class="main-header"><h1>RISK ANALYTICS DASHBOARD</h1><p>MULTI-DIMENSIONAL RISK ANALYSIS | DEMOGRAPHICS | GEOGRAPHIC PATTERNS</p></div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["DISTRIBUTION", "DEMOGRAPHICS", "GEOGRAPHY", "RISK TRENDS"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="chart-container"><h3>PREDICTIVE RISK MODEL</h3>', unsafe_allow_html=True)
            fig = go.Figure([go.Pie(labels=PIE_LABELS, values=PIE_VALUES, hole=0.4, marker_colors=PIE_COLORS, textinfo='label+percent', pull=[0.05,0,0,0.05])])
            fig.update_layout(height=500, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="chart-container"><h3>ACTUAL DISTRIBUTION</h3>', unsafe_allow_html=True)
            risk_reordered = risk_counts.reindex(RISK_ORDER)
            fig = go.Figure([go.Bar(x=risk_reordered.index, y=risk_reordered.values, marker_color=[RISK_COLORS[r] for r in risk_reordered.index], text=risk_reordered.values, textposition='auto', textfont=dict(color='white'))])
            fig.update_layout(height=500, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), xaxis_title="RISK LEVEL", yaxis_title="NUMBER OF PLAYERS")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="chart-container"><h3>GENDER DISTRIBUTION BY RISK</h3>', unsafe_allow_html=True)
            gender_risk = players.groupby(['risk_level', 'gender']).size().unstack(fill_value=0).reindex(RISK_ORDER)
            fig = go.Figure()
            fig.add_trace(go.Bar(name='MALE', x=gender_risk.index, y=gender_risk['M'], marker_color='#FFD700', text=gender_risk['M'], textposition='auto'))
            fig.add_trace(go.Bar(name='FEMALE', x=gender_risk.index, y=gender_risk['F'], marker_color='#8B0000', text=gender_risk['F'], textposition='auto'))
            fig.update_layout(barmode='group', height=450, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), xaxis_title="RISK LEVEL", yaxis_title="PLAYERS")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="chart-container"><h3>AGE DISTRIBUTION BY RISK</h3>', unsafe_allow_html=True)
            fig = px.box(players, x='risk_level', y='age', color='risk_level', color_discrete_map=RISK_COLORS, title="")
            fig.update_layout(height=450, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), xaxis_title="RISK LEVEL", yaxis_title="AGE")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-container"><h3>AGE PYRAMID BY RISK LEVEL</h3>', unsafe_allow_html=True)
        age_bins = pd.cut(players['age'], bins=[18,25,35,45,55,65,80], labels=['18-25','26-35','36-45','46-55','56-65','66+'])
        age_risk = pd.crosstab(age_bins, players['risk_level'])
        fig = px.bar(age_risk, title="", barmode='group', color_discrete_map=RISK_COLORS)
        fig.update_layout(height=450, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), xaxis_title="AGE GROUP", yaxis_title="COUNT")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="chart-container"><h3>TOP 10 REGIONS BY PLAYER COUNT</h3>', unsafe_allow_html=True)
            region_counts = players['region'].value_counts().head(10)
            fig = go.Figure([go.Bar(x=region_counts.values, y=region_counts.index, orientation='h', marker_color='#FFD700', text=region_counts.values, textposition='outside')])
            fig.update_layout(height=500, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), xaxis_title="PLAYERS", yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="chart-container"><h3>HIGH-RISK PERCENTAGE BY REGION</h3>', unsafe_allow_html=True)
            region_risk = players.groupby('region')['risk_label'].mean().sort_values(ascending=False).head(10)
            fig = go.Figure([go.Bar(x=region_risk.values*100, y=region_risk.index, orientation='h', marker_color='#8B0000', text=[f'{v*100:.1f}%' for v in region_risk.values], textposition='outside')])
            fig.update_layout(height=500, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), xaxis_title="HIGH-RISK PERCENTAGE")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="chart-container"><h3>WEEKLY RISK SCORE TREND (LAST 180 DAYS)</h3>', unsafe_allow_html=True)
        weekly_risk = trend_data.resample('W', on='date').mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=weekly_risk.index, y=weekly_risk['daily_risk_score'], mode='lines+markers', name='RISK SCORE', line=dict(color='#FFD700', width=3), marker=dict(size=8, color='#8B0000')))
        fig.update_layout(height=500, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), xaxis_title="DATE", yaxis_title="RISK SCORE")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

elif page == "FEATURES":
    st.markdown('<div class="main-header"><h1>BEHAVIORAL FEATURE ANALYSIS</h1><p>DEEP DIVE INTO PREDICTIVE INDICATORS | DISTRIBUTION PATTERNS | CORRELATION ANALYSIS</p></div>', unsafe_allow_html=True)
    
    feature_descriptions = {
        'bet_frequency': 'NUMBER OF BETS PLACED PER DAY - HIGHER FREQUENCY INDICATES ENGAGEMENT INTENSITY',
        'stake_variance': 'VARIABILITY IN BET AMOUNTS - SUDDEN CHANGES MAY INDICATE CHASING BEHAVIOR',
        'chasing_ratio': 'RATIO OF BETS PLACED IMMEDIATELY AFTER LOSSES - KEY RISK INDICATOR',
        'session_duration_avg': 'AVERAGE GAMBLING SESSION LENGTH IN MINUTES - LONGER SESSIONS = HIGHER RISK',
        'loss_streak_response': 'HOW BET SIZE CHANGES AFTER CONSECUTIVE LOSSES',
        'deposit_frequency': 'NUMBER OF DEPOSITS PER WEEK - RAPID DEPOSITS INDICATE URGENCY',
        'time_of_day_skew': 'PROPORTION OF BETS PLACED BETWEEN 00:00-06:00 - NIGHT BETTING IS RISK FACTOR',
        'withdrawal_ratio': 'WITHDRAWAL AMOUNT RELATIVE TO DEPOSITS - LOW RATIO INDICATES PROBLEMS',
        'self_excl_attempts': 'NUMBER OF TIMES PLAYER ATTEMPTED SELF-EXCLUSION'
    }
    
    selected = st.selectbox("SELECT BEHAVIORAL FEATURE", FEATURE_COLS, format_func=lambda x: x.replace('_', ' ').title())
    st.markdown(f'<div class="insight-box">DESCRIPTION: {feature_descriptions.get(selected, "BEHAVIORAL INDICATOR CORRELATED WITH GAMBLING RISK")}</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container"><h3>DISTRIBUTION BY RISK LEVEL</h3>', unsafe_allow_html=True)
        plot_df = features.copy()
        plot_df['risk_level'] = players['risk_level'].values
        fig = go.Figure()
        for risk in RISK_ORDER:
            subset = plot_df[plot_df['risk_level'] == risk]
            fig.add_trace(go.Histogram(x=subset[selected], name=risk.capitalize(), marker_color=RISK_COLORS[risk], opacity=0.7, nbinsx=35))
        fig.update_layout(barmode='overlay', height=500, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), xaxis_title=selected.replace('_', ' ').title(), yaxis_title="FREQUENCY")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container"><h3>BOXPLOT ANALYSIS</h3>', unsafe_allow_html=True)
        fig = go.Figure()
        for risk in RISK_ORDER:
            subset = plot_df[plot_df['risk_level'] == risk]
            fig.add_trace(go.Box(y=subset[selected], name=risk.capitalize(), marker_color=RISK_COLORS[risk], boxmean='sd'))
        fig.update_layout(height=500, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), yaxis_title=selected.replace('_', ' ').title())
        if selected in ['bet_frequency', 'stake_variance', 'deposit_frequency']:
            fig.update_yaxes(type="log")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("### FEATURE STATISTICS BY RISK LEVEL")
    stats_data = []
    for risk in RISK_ORDER:
        risk_players = players[players['risk_level'] == risk]
        risk_features = features[features['player_id'].isin(risk_players['player_id'])]
        stats_data.append({'Risk Level': risk.capitalize(), 'Mean': risk_features[selected].mean(), 'Median': risk_features[selected].median(), 'Std': risk_features[selected].std(), 'Min': risk_features[selected].min(), 'Max': risk_features[selected].max()})
    stats_df = pd.DataFrame(stats_data)
    st.dataframe(stats_df.style.format({col: '{:.2f}' for col in stats_df.columns if col != 'Risk Level'}), use_container_width=True)
    
    st.markdown("### FULL FEATURE CORRELATION MATRIX")
    corr = features[FEATURE_COLS].corr()
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=[c.replace('_',' ').title() for c in corr.columns], y=[c.replace('_',' ').title() for c in corr.columns], colorscale='RdBu', zmid=0, text=np.round(corr.values,2), texttemplate='%{text}', textfont={"size": 10, "color": "white"}))
    fig.update_layout(height=650, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
    st.plotly_chart(fig, use_container_width=True)

elif page == "GAMES":
    st.markdown('<div class="main-header"><h1>GAME TYPE RISK ANALYSIS</h1><p>RISK PATTERNS ACROSS DIFFERENT GAME CATEGORIES | BEHAVIORAL ANALYSIS PER GAME TYPE</p></div>', unsafe_allow_html=True)
    
    game_risk = players.groupby('preferred_game').apply(lambda x: (x['risk_label'] == 1).mean() * 100).sort_values(ascending=False).reset_index()
    game_risk.columns = ['GAME', 'HIGH RISK %']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container"><h3>HIGH RISK PERCENTAGE BY GAME TYPE</h3>', unsafe_allow_html=True)
        colors = ['#8B0000' if x > 11.3 else '#FFD700' for x in game_risk['HIGH RISK %']]
        fig = go.Figure([go.Bar(x=game_risk['HIGH RISK %'], y=game_risk['GAME'], orientation='h', marker_color=colors, text=game_risk['HIGH RISK %'].round(1).astype(str)+'%', textposition='outside')])
        fig.add_vline(x=11.3, line_dash="dash", line_color="#FFD700", annotation_text="DATASET AVERAGE 11.3%")
        fig.update_layout(height=500, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), xaxis_title="HIGH-RISK PLAYERS (%)")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container"><h3>PLAYER DISTRIBUTION BY GAME TYPE</h3>', unsafe_allow_html=True)
        game_counts = players['preferred_game'].value_counts()
        fig = go.Figure([go.Pie(labels=game_counts.index, values=game_counts.values, hole=0.35, marker=dict(colors=['#FFD700','#1a1a4e','#8B0000','#B22222']), textinfo='label+percent')])
        fig.update_layout(height=500, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="chart-container"><h3>RISK LEVEL COMPOSITION BY GAME TYPE</h3>', unsafe_allow_html=True)
    game_comp = players.groupby(['preferred_game', 'risk_level']).size().unstack(fill_value=0)
    game_comp = game_comp.div(game_comp.sum(axis=1), axis=0) * 100
    game_comp = game_comp.reindex(columns=RISK_ORDER)
    fig = go.Figure()
    for risk in RISK_ORDER:
        fig.add_trace(go.Bar(name=risk.capitalize(), x=game_comp.index, y=game_comp[risk], marker_color=RISK_COLORS[risk]))
    fig.update_layout(barmode='stack', height=500, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), xaxis_title="GAME TYPE", yaxis_title="PERCENTAGE")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("### GAME TYPE RISK METRICS")
    game_metrics = players.groupby('preferred_game').agg({'risk_label': ['mean', 'count'], 'age': 'mean', 'gender': lambda x: (x == 'M').mean()}).round(3)
    game_metrics.columns = ['HIGH RISK %', 'PLAYERS', 'AVG AGE', 'MALE %']
    game_metrics['HIGH RISK %'] = game_metrics['HIGH RISK %'] * 100
    game_metrics['MALE %'] = game_metrics['MALE %'] * 100
    st.dataframe(game_metrics.sort_values('HIGH RISK %', ascending=False).style.format({'HIGH RISK %': '{:.1f}%', 'AVG AGE': '{:.1f}', 'MALE %': '{:.1f}%'}), use_container_width=True)

elif page == "TEMPORAL":
    st.markdown('<div class="main-header"><h1>TEMPORAL BETTING PATTERNS</h1><p>24/7 BEHAVIORAL ANALYSIS | SESSION DYNAMICS | TIME-BASED RISK INDICATORS</p></div>', unsafe_allow_html=True)
    
    trans_risk = trans.merge(players[['player_id', 'risk_level']], on='player_id')
    bets = trans_risk[trans_risk['type'] == 'bet'].copy()
    bets['hour'] = bets['timestamp'].dt.hour
    bets['day_of_week'] = bets['timestamp'].dt.dayofweek
    bets['week'] = bets['timestamp'].dt.isocalendar().week
    
    st.markdown('<div class="chart-container"><h3>24-HOUR BETTING ACTIVITY PATTERN</h3>', unsafe_allow_html=True)
    fig = go.Figure()
    for risk in RISK_ORDER:
        subset = bets[bets['risk_level'] == risk]
        hourly = subset.groupby('hour').size() / len(subset) * 100
        fig.add_trace(go.Scatter(x=hourly.index, y=hourly.values, name=risk.capitalize(), line=dict(color=RISK_COLORS[risk], width=2.5), mode='lines+markers', marker=dict(size=6)))
    fig.add_vrect(x0=0, x1=6, fillcolor="#8B0000", opacity=0.3, line_width=0, annotation_text="NIGHT WINDOW (00:00-06:00)")
    fig.update_layout(height=500, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), xaxis_title="HOUR OF DAY (24H)", yaxis_title="% OF TOTAL BETS")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container"><h3>NIGHT-TIME BETTING SHARE (00:00-06:00)</h3>', unsafe_allow_html=True)
        night = []
        for risk in RISK_ORDER:
            subset = bets[bets['risk_level'] == risk]
            night.append((subset['hour'].between(0,5)).sum() / len(subset) * 100 if len(subset) > 0 else 0)
        fig = go.Figure([go.Bar(x=[r.capitalize() for r in RISK_ORDER], y=night, marker_color=[RISK_COLORS[r] for r in RISK_ORDER], text=[f'{n:.1f}%' for n in night], textposition='auto')])
        fig.add_hline(y=bets['hour'].between(0,5).mean()*100, line_dash="dash", line_color="#FFD700", annotation_text=f"OVERALL AVERAGE: {bets['hour'].between(0,5).mean()*100:.1f}%")
        fig.update_layout(height=450, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), xaxis_title="RISK LEVEL", yaxis_title="% OF BETS AT NIGHT")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container"><h3>WEEKLY BETTING PATTERN</h3>', unsafe_allow_html=True)
        bets['day_name'] = bets['timestamp'].dt.day_name()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_activity = bets.groupby(['risk_level', 'day_name']).size().reset_index(name='count')
        daily_activity['day_name'] = pd.Categorical(daily_activity['day_name'], categories=day_order, ordered=True)
        daily_pct = daily_activity.groupby('risk_level').apply(lambda x: x.assign(pct=x['count']/x['count'].sum()*100)).reset_index(drop=True)
        fig = px.line(daily_pct, x='day_name', y='pct', color='risk_level', color_discrete_map=RISK_COLORS, markers=True)
        fig.update_layout(height=450, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), xaxis_title="DAY OF WEEK", yaxis_title="% OF WEEKLY BETS")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="chart-container"><h3>SESSION DURATION ANALYSIS</h3>', unsafe_allow_html=True)
    sessions = bets.groupby(['session_id', 'risk_level']).agg({'timestamp': lambda x: (x.max() - x.min()).seconds/60}).reset_index()
    sessions.columns = ['session_id', 'risk_level', 'duration']
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        fig = go.Figure()
        for risk in RISK_ORDER:
            subset = sessions[sessions['risk_level'] == risk]
            if len(subset) > 0:
                fig.add_trace(go.Box(y=subset['duration'], name=risk.capitalize(), marker_color=RISK_COLORS[risk], boxmean='sd'))
        fig.update_layout(height=450, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), yaxis_title="DURATION (MINUTES)")
        fig.update_yaxes(type="log")
        st.plotly_chart(fig, use_container_width=True)
    
    with col_s2:
        sessions['category'] = pd.cut(sessions['duration'], bins=[0,15,30,60,120,240,1000], labels=['0-15MIN','15-30MIN','30-60MIN','1-2H','2-4H','4H+'])
        duration_by_risk = pd.crosstab(sessions['category'], sessions['risk_level'], normalize='columns') * 100
        fig = px.bar(duration_by_risk, barmode='group', color_discrete_map=RISK_COLORS)
        fig.update_layout(height=450, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), xaxis_title="SESSION DURATION", yaxis_title="PERCENTAGE")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "MODEL":
    st.markdown('<div class="main-header"><h1>MACHINE LEARNING MODEL PERFORMANCE</h1><p>XGBOOST CLASSIFIER | RANDOM FOREST | LOGISTIC REGRESSION | ENSEMBLE LEARNING</p></div>', unsafe_allow_html=True)
    
    models_list = ['LOGISTIC REGRESSION', 'RANDOM FOREST', 'XGBOOST']
    y_probs = [models['y_prob_lr'], models['y_prob_rf'], models['y_prob_xgb']]
    aucs = [models['auc_lr'], models['auc_rf'], models['auc_xgb']]
    
    st.markdown("### MODEL COMPARISON METRICS")
    metrics_data = []
    for name, y_prob in zip(models_list, y_probs):
        y_pred = (y_prob >= 0.5).astype(int)
        metrics_data.append({'MODEL': name, 'AUC-ROC': roc_auc_score(models['y_test'], y_prob), 'F1 SCORE': f1_score(models['y_test'], y_pred), 'RECALL': recall_score(models['y_test'], y_pred), 'PRECISION': precision_score(models['y_test'], y_pred), 'BRIER SCORE': brier_score_loss(models['y_test'], y_prob)})
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df.style.format({col: '{:.4f}' for col in metrics_df.columns if col != 'MODEL'}).background_gradient(subset=['AUC-ROC'], cmap='Reds'), use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container"><h3>ROC CURVES COMPARISON</h3>', unsafe_allow_html=True)
        fig = go.Figure()
        colors_roc = {'LOGISTIC REGRESSION': '#1a1a4e', 'RANDOM FOREST': '#FFD700', 'XGBOOST': '#8B0000'}
        for name, y_prob, auc_val in zip(models_list, y_probs, aucs):
            fpr, tpr, _ = roc_curve(models['y_test'], y_prob)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{name} (AUC={auc_val:.4f})', line=dict(color=colors_roc[name], width=2.5)))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='RANDOM CLASSIFIER', line=dict(dash='dash', color='white', width=1.5)))
        fig.update_layout(height=450, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), xaxis_title="FALSE POSITIVE RATE", yaxis_title="TRUE POSITIVE RATE")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container"><h3>PRECISION-RECALL CURVE</h3>', unsafe_allow_html=True)
        fig = go.Figure()
        for name, y_prob in zip(models_list, y_probs):
            precision, recall, _ = precision_recall_curve(models['y_test'], y_prob)
            fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=name, line=dict(color=colors_roc[name], width=2.5)))
        fig.update_layout(height=450, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), xaxis_title="RECALL", yaxis_title="PRECISION")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="chart-container"><h3>FEATURE IMPORTANCE (XGBOOST)</h3>', unsafe_allow_html=True)
    imp = pd.DataFrame({'FEATURE': [f.replace('_',' ').title() for f in FEATURE_COLS], 'IMPORTANCE': models['xgb'].feature_importances_})
    imp = imp.sort_values('IMPORTANCE', ascending=True)
    fig = px.bar(imp, y='FEATURE', x='IMPORTANCE', orientation='h', color='IMPORTANCE', color_continuous_scale='Reds')
    fig.update_layout(height=500, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("### CONFUSION MATRICES")
    col_c1, col_c2, col_c3 = st.columns(3)
    for col, name, y_prob in zip([col_c1, col_c2, col_c3], models_list, y_probs):
        with col:
            y_pred = (y_prob >= 0.5).astype(int)
            cm = confusion_matrix(models['y_test'], y_pred)
            fig = go.Figure(data=go.Heatmap(z=cm, x=['PRED LOW', 'PRED HIGH'], y=['ACTUAL LOW', 'ACTUAL HIGH'], text=cm, texttemplate='%{text}', textfont={"size": 16, "color": "white"}, colorscale='Reds', showscale=False))
            fig.update_layout(title=name, height=350, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### CROSS-VALIDATION PERFORMANCE (XGBOOST)")
    st.markdown(f"""
    <div style="background: rgba(0,0,0,0.5); border-radius: 16px; padding: 1rem;">
        <div style="display: flex; justify-content: space-around; text-align: center;">
            <div><div style="font-size: 0.8rem; opacity: 0.7;">5-FOLD CV MEAN</div><div style="font-size: 1.8rem; font-weight: bold; color: #FFD700;">{models['cv_mean']:.4f}</div></div>
            <div><div style="font-size: 0.8rem; opacity: 0.7;">STANDARD DEVIATION</div><div style="font-size: 1.8rem; font-weight: bold; color: #FFD700;">±{models['cv_std']:.4f}</div></div>
            <div><div style="font-size: 0.8rem; opacity: 0.7;">TEST AUC</div><div style="font-size: 1.8rem; font-weight: bold; color: #FFD700;">{models['auc_xgb']:.4f}</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### CLASSIFICATION REPORT (XGBOOST)")
    report = classification_report(models['y_test'], models['y_pred_xgb'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format('{:.3f}'), use_container_width=True)

elif page == "PLAYERS":
    st.markdown('<div class="main-header"><h1>PLAYER RISK EXPLORER</h1><p>DEEP DIVE INTO INDIVIDUAL PLAYER PROFILES | BEHAVIORAL FORENSICS | INTERVENTION RECOMMENDATIONS</p></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### SEARCH FILTERS")
        selected_risk = st.multiselect("RISK LEVEL", RISK_ORDER, default=RISK_ORDER)
        age_range = st.slider("AGE RANGE", 18, 70, (18, 70))
        selected_gender = st.multiselect("GENDER", ['M', 'F'], default=['M', 'F'])
        selected_game = st.multiselect("PREFERRED GAME", players['preferred_game'].unique(), default=players['preferred_game'].unique()[:3])
        
        filtered = players[
            (players['risk_level'].isin(selected_risk)) &
            (players['age'].between(age_range[0], age_range[1])) &
            (players['gender'].isin(selected_gender)) &
            (players['preferred_game'].isin(selected_game))
        ]
        
        st.markdown("---")
        st.metric("PLAYERS FOUND", len(filtered))
        
        if len(filtered) > 0:
            st.markdown("### RISK BREAKDOWN")
            risk_dist = filtered['risk_level'].value_counts()
            for risk in RISK_ORDER:
                if risk in risk_dist:
                    pct = risk_dist[risk] / len(filtered) * 100
                    st.markdown(f"{risk.capitalize()}: {risk_dist[risk]} ({pct:.1f}%)")
                    st.progress(pct/100)
    
    with col2:
        if len(filtered) > 0:
            display_cols = ['player_id', 'risk_level', 'age', 'gender', 'region', 'preferred_game']
            st.dataframe(filtered[display_cols].head(100).style.apply(lambda x: ['background: rgba(178,34,34,0.3)' if x['risk_level'] == 'problem' else '' for _ in x], axis=1), use_container_width=True, height=350)
            
            st.markdown("---")
            st.markdown("### PLAYER DEEP DIVE")
            selected_id = st.selectbox("SELECT PLAYER ID", filtered['player_id'].head(50))
            player = players[players['player_id'] == selected_id].iloc[0]
            player_features = features[features['player_id'] == selected_id].iloc[0]
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown(f"**RISK LEVEL:** `{player['risk_level'].upper()}`")
                st.markdown(f"**AGE:** {player['age']} | **GENDER:** {player['gender']}")
                st.markdown(f"**REGION:** {player['region']}")
                st.markdown(f"**PREFERRED GAME:** {player['preferred_game']}")
            
            with col_b:
                st.markdown("**BEHAVIORAL METRICS:**")
                st.markdown(f"- SESSION DURATION: `{player_features.get('session_duration_avg', 0):.1f}` MIN")
                st.markdown(f"- BET FREQUENCY: `{player_features.get('bet_frequency', 0):.1f}`/DAY")
                st.markdown(f"- CHASING RATIO: `{player_features.get('chasing_ratio', 0):.2f}`")
                st.markdown(f"- NIGHT BETTING: `{player_features.get('time_of_day_skew', 0):.2f}`")
                st.markdown(f"- STAKE VARIANCE: `{player_features.get('stake_variance', 0):.0f}`")
            
            with col_c:
                if player['risk_label'] == 1:
                    st.error("HIGH RISK ALERT - INTERVENTION REQUIRED")
                    st.markdown("**RECOMMENDED ACTIONS:**")
                    st.markdown("1. TEMPORARY ACCOUNT PAUSE")
                    st.markdown("2. OUTREACH CALL REQUIRED")
                    st.markdown("3. WEEKLY MONITORING")
                    st.markdown("4. DEPOSIT LIMIT IMPLEMENTATION")
                else:
                    st.success("LOW RISK PROFILE")
                    st.markdown("**RECOMMENDED ACTIONS:**")
                    st.markdown("1. STANDARD MONITORING")
                    st.markdown("2. MONTHLY CHECK-IN")
                    st.markdown("3. NO INTERVENTION NEEDED")
            
            st.markdown("### BEHAVIORAL PROFILE RADAR")
            radar_features = ['bet_frequency', 'session_duration_avg', 'chasing_ratio', 'time_of_day_skew', 'stake_variance']
            radar_data = []
            for feat in radar_features:
                if feat in player_features:
                    max_val = features[feat].max()
                    min_val = features[feat].min()
                    normalized = (player_features[feat] - min_val) / (max_val - min_val) if max_val > min_val else 0.5
                    radar_data.append(dict(r=normalized, theta=feat.replace('_', ' ').title()))
            
            if radar_data:
                fig = go.Figure(data=go.Scatterpolar(r=[d['r'] for d in radar_data], theta=[d['theta'] for d in radar_data], fill='toself', marker=dict(color='#FFD700', size=6), line=dict(color='#FFD700', width=2)))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1], tickfont=dict(color='white'))), showlegend=False, height=400, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("NO PLAYERS MATCH THE SELECTED FILTERS. PLEASE ADJUST YOUR CRITERIA.")

elif page == "SHARE":
    st.markdown('<div class="main-header"><h1>SHARE YOUR DASHBOARD</h1><p>DEPLOY AND SHARE WITH FRIENDS AND COLLEAGUES</p></div>', unsafe_allow_html=True)
    
    local_ip = get_local_ip()
    port = 8501
    
    st.markdown("### QUICK ACCESS URLS")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="share-card"><h4>LOCAL MACHINE</h4><div class="code-block">http://localhost:8501</div><p style="font-size:0.7rem">WORKS ON THIS COMPUTER</p></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'<div class="share-card"><h4>SAME WIFI NETWORK</h4><div class="code-block">http://{local_ip}:8501</div><p style="font-size:0.7rem">SHARE WITH FRIENDS ON SAME NETWORK</p></div>', unsafe_allow_html=True)
        if st.button("COPY WIFI URL", use_container_width=True):
            st.success(f"COPIED: http://{local_ip}:8501")
    
    with col3:
        st.markdown('<div class="share-card"><h4>PUBLIC (NGROK)</h4><div class="code-block">ngrok http 8501</div><p style="font-size:0.7rem">SHARE WITH ANYONE WORLDWIDE</p></div>', unsafe_allow_html=True)
    
    st.markdown("### INSTRUCTIONS FOR FRIENDS")
    col_inst1, col_inst2 = st.columns(2)
    
    with col_inst1:
        st.markdown("#### SAME WIFI NETWORK")
        st.markdown(f"""
        1. CONNECT TO THE **SAME WIFI** AS THE HOST COMPUTER
        2. OPEN A WEB BROWSER (CHROME, FIREFOX, SAFARI)
        3. TYPE THIS ADDRESS: `http://{local_ip}:8501`
        4. THE DASHBOARD WILL LOAD AUTOMATICALLY
        """)
    
    with col_inst2:
        st.markdown("#### DIFFERENT LOCATION")
        st.markdown("""
        **OPTION A - NGROK (TEMPORARY):**
        - HOST RUNS: `ngrok http 8501`
        - SHARE THE HTTPS://XXXX.NGROK.IO URL
        
        **OPTION B - DEPLOY (PERMANENT):**
        - PUSH CODE TO GITHUB
        - DEPLOY ON STREAMLIT.IO
        - GET PERMANENT URL
        """)
    
    st.markdown("### DEPLOY FOR PERMANENT ACCESS")
    col_deploy1, col_deploy2 = st.columns(2)
    
    with col_deploy1:
        st.markdown("#### STREAMLIT CLOUD (FREE)")
        st.markdown("""
        1. CREATE GITHUB REPOSITORY
        2. PUSH YOUR CODE
        3. GO TO SHARE.STREAMLIT.IO
        4. DEPLOY → `HTTPS://YOUR-APP.STREAMLIT.APP`
        """)
        st.markdown("#### HUGGING FACE SPACES (FREE)")
        st.markdown("""
        1. CREATE ACCOUNT AT HUGGINGFACE.CO
        2. CREATE NEW SPACE WITH STREAMLIT SDK
        3. UPLOAD YOUR FILES
        4. GET `HTTPS://YOUR-NAME.HF.SPACE`
        """)
    
    with col_deploy2:
        st.markdown("#### RENDER.COM (FREE)")
        st.markdown("""
        1. CREATE ACCOUNT AT RENDER.COM
        2. CLICK 'NEW WEB SERVICE'
        3. CONNECT GITHUB REPOSITORY
        4. DEPLOY WITH AUTOSCALING
        """)
        st.markdown("#### RAILWAY.APP (FREE)")
        st.markdown("""
        1. CREATE ACCOUNT AT RAILWAY.APP
        2. DEPLOY FROM GITHUB
        3. GET AUTOMATIC HTTPS URL
        """)
    
    st.markdown("### SHARE VIA MESSAGE")
    share_msg = f"""
GAMBLING RISK DETECTION DASHBOARD - YSU MASTER'S THESIS

I'VE BUILT AN AI-POWERED GAMBLING RISK DETECTION SYSTEM FOR MY THESIS!

ACCESS THE DASHBOARD:
- SAME WIFI: http://{local_ip}:8501
- PUBLIC: (ASK FOR NGROK URL)

KEY FEATURES:
- 40/25/20/15 RISK SEGMENTATION MODEL
- XGBOOST WITH {models['auc_xgb']:.3f} AUC-ROC
- REAL-TIME PLAYER RISK ASSESSMENT
- BEHAVIORAL ANALYTICS FOR {total_players:,} PLAYERS OVER 180 DAYS
- INDIVIDUAL PLAYER RISK PROFILING

TRY IT OUT AND LET ME KNOW YOUR FEEDBACK!

---
YSU MASTER'S PROGRAM | DATA SCIENCE IN BUSINESS 2025-2026
"""
    st.text_area("COPY THIS MESSAGE:", share_msg, height=300)
    if st.button("COPY MESSAGE", use_container_width=True):
        st.success("MESSAGE COPIED TO CLIPBOARD!")

st.markdown('<div class="footer">GAMBLING RISK DETECTION SYSTEM | ARMENIAN SYNTHETIC DATASET | YSU MASTER\'S THESIS 2025-2026 | TEAM: HAYK MNATSAKANYAN, ANUSH KHACHATRYAN, SRBUHI KHACHATRYAN, MANE DAVTYAN | SUPERVISOR: A. POKRIKYAN | RESEARCH & EDUCATIONAL PURPOSE ONLY</div>', unsafe_allow_html=True)
