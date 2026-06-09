# -*- coding: utf-8 -*-
"""
Armenian Synthetic Gambling Behavior Dataset - Complete Dashboard
EPՀ Master's Project | Data Science in Business | 2025–2026

Save this file as: gambling_dashboard.py (NOT dash.py)
Run: python gambling_dashboard.py
"""

import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback, ctx
import dash_bootstrap_components as dbc

warnings.filterwarnings("ignore")

# ==================== DATA LOADING ====================

# Update this path to your data directory
DATA_DIR = Path(r"C:\Users\User\Desktop\Mag1\Tntes\Final")  # ← UPDATE THIS PATH

PLAYERS_CSV = DATA_DIR / "players.csv"
FEATURES_CSV = DATA_DIR / "features.csv"
TRANSACTIONS_CSV = DATA_DIR / "transactions_sample_50k.csv"
METADATA_JSON = DATA_DIR / "metadata.json"

# Check if files exist
if not PLAYERS_CSV.exists():
    print(f"ERROR: Cannot find {PLAYERS_CSV}")
    print("Please update DATA_DIR to the correct path")
    exit(1)

players = pd.read_csv(PLAYERS_CSV)
features = pd.read_csv(FEATURES_CSV)
trans = pd.read_csv(TRANSACTIONS_CSV, parse_dates=["timestamp"])

with open(METADATA_JSON, "r", encoding="utf-8") as f:
    meta = json.load(f)

# ==================== GLOBAL CONSTANTS ====================

RISK_ORDER = ["recreational", "regular", "at_risk", "problem"]
RISK_COLORS = {
    "recreational": "#2ecc71",  # Emerald green
    "regular": "#3498db",       # Blue
    "at_risk": "#f39c12",       # Orange
    "problem": "#e74c3c"        # Red
}

FEATURE_COLS = [
    "bet_frequency",
    "stake_variance",
    "chasing_ratio",
    "session_duration_avg",
    "loss_streak_response",
    "deposit_frequency",
    "time_of_day_skew",
    "withdrawal_ratio",
    "self_excl_attempts"
]

FEATURE_LABELS = {
    "bet_frequency": "Bet Frequency (bets/week)",
    "stake_variance": "Stake Variance (AMD)",
    "chasing_ratio": "Chasing Ratio",
    "session_duration_avg": "Session Duration (minutes)",
    "loss_streak_response": "Loss Streak Response",
    "deposit_frequency": "Deposit Frequency (deposits/month)",
    "time_of_day_skew": "Night-time Bet Share",
    "withdrawal_ratio": "Withdrawal Ratio",
    "self_excl_attempts": "Self-Exclusion Attempts"
}

# Model performance thresholds (from proposal)
THRESHOLDS = {
    "auc_roc": 0.85,
    "f1": 0.78,
    "precision_k": 0.75,
    "recall": 0.80,
    "brier": 0.15
}

# Armenian market context
MARKET_CONTEXT = {
    "ggr_2025": "205M USD",
    "avg_income": "270,595 AMD",
    "national_avg_income": "113,163 AMD",
    "yerevan_share": 72.3,
    "problem_gambling_prevalence": "2.3-5.3%"
}

# ==================== DATA PROCESSING ====================

# Merge data for analysis
trans_risk = trans.merge(players[["player_id", "risk_level", "risk_label"]], on="player_id")
bets = trans_risk[trans_risk["type"] == "bet"].copy()
bets["hour"] = bets["timestamp"].dt.hour
bets["date"] = bets["timestamp"].dt.date

# Calculate summary statistics
stats = meta["stats"]
sim_period = stats["simulation_period"]
total_players = stats["total_observations"]["players_total"]
total_transactions = stats["total_observations"]["transactions_total"]
avg_trans_per_player = stats["total_observations"]["avg_transactions_per_player"]
risk_dist = stats["risk_class_distribution"]
positive_class_share = risk_dist["positive_class_share_percent"]

# ==================== CHART CREATION FUNCTIONS ====================

def create_overview_kpi_cards():
    """Create KPI cards for overview section"""
    cards = [
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(f"{total_players:,}", className="card-title text-center display-4 text-primary"),
                    html.P("Total Players", className="card-text text-center text-muted"),
                    html.Small(f"Armenia's regulated market", className="text-muted d-block text-center")
                ])
            ], className="shadow-sm h-100 border-0")
        ], md=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(f"{total_transactions:,}", className="card-title text-center display-4 text-success"),
                    html.P("Total Transactions", className="card-text text-center text-muted"),
                    html.Small(f"Avg {avg_trans_per_player:.0f}/player", className="text-muted d-block text-center")
                ])
            ], className="shadow-sm h-100 border-0")
        ], md=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(f"{positive_class_share:.1f}%", className="card-title text-center display-4 text-warning"),
                    html.P("High-Risk Players", className="card-text text-center text-muted"),
                    html.Small("at_risk + problem gamblers", className="text-muted d-block text-center")
                ])
            ], className="shadow-sm h-100 border-0")
        ], md=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(MARKET_CONTEXT["ggr_2025"], className="card-title text-center display-4 text-info"),
                    html.P("Market GGR (2025)", className="card-text text-center text-muted"),
                    html.Small("SiGMA Armenia Report", className="text-muted d-block text-center")
                ])
            ], className="shadow-sm h-100 border-0")
        ], md=3)
    ]
    return dbc.Row(cards, className="g-3 mb-4")

def create_risk_distribution_chart():
    """Create risk distribution pie chart with donut style"""
    risk_counts = players["risk_level"].value_counts().reindex(RISK_ORDER)
    
    fig = go.Figure(data=[
        go.Pie(
            labels=RISK_ORDER,
            values=risk_counts.values,
            marker_colors=[RISK_COLORS[r] for r in RISK_ORDER],
            hole=0.5,
            textinfo="label+percent",
            textposition="auto",
            textfont_size=14,
            hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent:.1f}%<extra></extra>",
            pull=[0.05 if r in ["at_risk", "problem"] else 0 for r in RISK_ORDER]
        )
    ])
    
    fig.update_layout(
        title=dict(text="Risk Class Distribution", font_size=18, x=0.5),
        height=450,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
        annotations=[
            dict(
                text=f"Total<br>{total_players:,}",
                x=0.5, y=0.5, font_size=20, showarrow=False,
                align="center", font_family="Arial Black"
            )
        ]
    )
    return fig

def create_feature_comparison_chart():
    """Create grouped bar chart comparing feature medians across risk classes"""
    df_plot = features.merge(players[["player_id", "risk_level"]], on="player_id")
    
    data = []
    for risk in RISK_ORDER:
        subset = df_plot[df_plot["risk_level"] == risk]
        medians = [subset[f].median() for f in FEATURE_COLS]
        data.append(medians)
    
    fig = go.Figure()
    
    for i, risk in enumerate(RISK_ORDER):
        fig.add_trace(go.Bar(
            name=risk,
            x=list(FEATURE_LABELS.values()),
            y=data[i],
            marker_color=RISK_COLORS[risk],
            text=[f"{v:.1f}" if v < 1000 else f"{v/1000:.0f}K" for v in data[i]],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>" + risk + ": %{y:,.1f}<extra></extra>"
        ))
    
    fig.update_layout(
        title=dict(text="Behavioral Indicators - Median Values by Risk Level", font_size=18, x=0.5),
        xaxis_title="Behavioral Indicator",
        yaxis_title="Median Value",
        barmode="group",
        height=500,
        legend_title="Risk Level",
        xaxis_tickangle=-45,
        yaxis_type="log"
    )
    
    return fig

def create_bet_volume_chart():
    """Create chart comparing bet volumes across risk classes"""
    bet_by_risk = features.merge(players[["player_id", "risk_level"]], on="player_id")
    
    fig = go.Figure()
    
    for risk in RISK_ORDER:
        subset = bet_by_risk[bet_by_risk["risk_level"] == risk]["total_bet_amd"]
        
        fig.add_trace(go.Box(
            y=subset,
            name=risk,
            marker_color=RISK_COLORS[risk],
            boxmean="sd",
            jitter=0.3,
            pointpos=-1.8,
            hovertemplate="<b>%{x}</b><br>Total Bet: %{y:,.0f} AMD<extra></extra>"
        ))
    
    fig.update_layout(
        title=dict(text="Total Bet Volume by Risk Level (180 days)", font_size=18, x=0.5),
        xaxis_title="Risk Level",
        yaxis_title="Total Bet (AMD)",
        height=500,
        yaxis_type="log",
        yaxis_tickformat=",.0f"
    )
    
    return fig

def create_gender_risk_chart():
    """Create grouped bar chart for gender distribution by risk level"""
    gender_risk = players.groupby(["risk_level", "gender"]).size().reset_index(name="count")
    totals = gender_risk.groupby("risk_level")["count"].transform("sum")
    gender_risk["pct"] = gender_risk["count"] / totals * 100
    
    fig = go.Figure()
    
    for gender, color in [("M", "#3498db"), ("F", "#e91e8c")]:
        subset = gender_risk[gender_risk["gender"] == gender]
        pcts = []
        for risk in RISK_ORDER:
            val = subset[subset["risk_level"] == risk]["pct"].values
            pcts.append(val[0] if len(val) > 0 else 0)
        
        fig.add_trace(go.Bar(
            name="Male" if gender == "M" else "Female",
            x=RISK_ORDER,
            y=pcts,
            marker_color=color,
            text=[f"{v:.1f}%" for v in pcts],
            textposition="inside",
            hovertemplate="<b>%{x}</b><br>Gender: " + ("Male" if gender == "M" else "Female") + "<br>Share: %{y:.1f}%<extra></extra>"
        ))
    
    fig.update_layout(
        title=dict(text="Gender Distribution by Risk Level", font_size=18, x=0.5),
        xaxis_title="Risk Level",
        yaxis_title="% within Risk Class",
        barmode="group",
        height=450,
        legend_title="Gender"
    )
    
    return fig

def create_preferred_game_chart():
    """Create bar chart for preferred game distribution"""
    game_counts = players["preferred_game"].value_counts().head(10)
    
    fig = go.Figure(data=[
        go.Bar(
            x=game_counts.values,
            y=game_counts.index,
            orientation="h",
            marker_color="#8e44ad",
            text=game_counts.values,
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Players: %{x}<extra></extra>"
        )
    ])
    
    fig.update_layout(
        title=dict(text="Top 10 Preferred Game Types", font_size=18, x=0.5),
        xaxis_title="Number of Players",
        yaxis_title="Game Type",
        height=500,
        yaxis=dict(categoryorder="total ascending")
    )
    
    return fig

def create_region_chart():
    """Create horizontal bar chart for region distribution"""
    region_counts = players["region"].value_counts()
    
    fig = go.Figure(data=[
        go.Bar(
            x=region_counts.values,
            y=region_counts.index,
            orientation="h",
            marker_color="#16a085",
            text=region_counts.values,
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Players: %{x}<extra></extra>"
        )
    ])
    
    fig.update_layout(
        title=dict(text="Players by Region", font_size=18, x=0.5),
        xaxis_title="Number of Players",
        yaxis_title="Region",
        height=500,
        yaxis=dict(categoryorder="total ascending")
    )
    
    return fig

def create_model_threshold_chart():
    """Create progress bars for model performance thresholds"""
    # Current performance (placeholder - replace with actual model results)
    current_performance = {
        "auc_roc": 0.87,
        "f1": 0.81,
        "precision_k": 0.79,
        "recall": 0.84,
        "brier": 0.12
    }
    
    fig = make_subplots(
        rows=5, cols=1,
        subplot_titles=("AUC-ROC", "F1 Score", "Precision@K", "Recall", "Brier Score"),
        vertical_spacing=0.12
    )
    
    metrics = ["auc_roc", "f1", "precision_k", "recall", "brier"]
    titles = ["AUC-ROC", "F1 Score", "Precision@K", "Recall", "Brier Score"]
    is_higher_better = [True, True, True, True, False]
    
    for i, (metric, title, higher_better) in enumerate(zip(metrics, titles, is_higher_better), 1):
        current = current_performance[metric]
        target = THRESHOLDS[metric]
        
        if higher_better:
            bar_color = "#27ae60" if current >= target else "#c0392b"
        else:
            bar_color = "#27ae60" if current <= target else "#c0392b"
        
        fig.add_trace(
            go.Bar(
                x=[current],
                y=["Current"],
                orientation="h",
                marker_color=bar_color,
                text=[f"{current:.2f}"],
                textposition="outside",
                showlegend=False,
                hovertemplate=f"{title}<br>Current: {{x:.3f}}<br>Target: {target}<extra></extra>"
            ),
            row=i, col=1
        )
        
        fig.add_vline(x=target, line_dash="dash", line_color="white", 
                      annotation_text=f"Target: {target}", row=i, col=1)
        
        fig.update_xaxes(range=[0, 1], row=i, col=1)
        fig.update_yaxes(showticklabels=False, row=i, col=1)
    
    fig.update_layout(
        title=dict(text="Model Performance vs. Proposal Targets", font_size=18, x=0.5),
        height=700,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig

def create_income_vs_bet_chart():
    """Create scatter plot of income vs betting behavior"""
    df = players.merge(features[["player_id", "bet_frequency", "total_bet_amd"]], on="player_id")
    df = df.dropna(subset=["monthly_income_amd", "total_bet_amd"])
    
    fig = go.Figure()
    
    for risk in RISK_ORDER:
        subset = df[df["risk_level"] == risk]
        fig.add_trace(go.Scatter(
            x=subset["monthly_income_amd"],
            y=subset["total_bet_amd"],
            mode="markers",
            name=risk,
            marker=dict(
                color=RISK_COLORS[risk],
                size=8,
                opacity=0.6,
                line=dict(width=1, color="white")
            ),
            hovertemplate="<b>" + risk + "</b><br>Income: %{x:,.0f} AMD<br>Total Bet: %{y:,.0f} AMD<extra></extra>"
        ))
    
    fig.update_layout(
        title=dict(text="Monthly Income vs Total Bet Volume", font_size=18, x=0.5),
        xaxis_title="Monthly Income (AMD)",
        yaxis_title="Total Bet (AMD)",
        height=500,
        xaxis_type="log",
        yaxis_type="log",
        xaxis_tickformat=",.0f",
        yaxis_tickformat=",.0f",
        legend_title="Risk Level"
    )
    
    fig.add_hline(y=df["total_bet_amd"].median(), line_dash="dash", line_color="gray",
                  annotation_text=f"Median Bet: {df['total_bet_amd'].median():,.0f} AMD")
    fig.add_vline(x=df["monthly_income_amd"].median(), line_dash="dash", line_color="gray",
                  annotation_text=f"Median Income: {df['monthly_income_amd'].median():,.0f} AMD")
    
    return fig

def create_temporal_trend_chart():
    """Create time series of betting activity over simulation period"""
    bets_daily = bets.copy()
    bets_daily["date"] = pd.to_datetime(bets_daily["timestamp"]).dt.date
    
    daily_counts = bets_daily.groupby(["date", "risk_level"]).size().reset_index(name="count")
    
    fig = go.Figure()
    
    for risk in RISK_ORDER:
        subset = daily_counts[daily_counts["risk_level"] == risk]
        fig.add_trace(go.Scatter(
            x=subset["date"],
            y=subset["count"],
            mode="lines",
            name=risk,
            line=dict(color=RISK_COLORS[risk], width=2),
            stackgroup="one",
            groupnorm="percent",
            hovertemplate="<b>Date: %{{x|%Y-%m-%d}}</b><br>" + risk + ": %{{y}} bets<extra></extra>"
        ))
    
    fig.update_layout(
        title=dict(text="Daily Betting Activity by Risk Level (Stacked %)", font_size=18, x=0.5),
        xaxis_title="Date",
        yaxis_title="% of Daily Bets",
        height=450,
        legend_title="Risk Level"
    )
    
    return fig

def create_game_risk_heatmap():
    """Create heatmap of risk concentration by game type"""
    game_risk = (
        players.groupby(["preferred_game", "risk_level"])
        .size()
        .reset_index(name="count")
    )
    
    totals = game_risk.groupby("preferred_game")["count"].transform("sum")
    game_risk["pct"] = game_risk["count"] / totals * 100
    
    pivot = game_risk.pivot(index="preferred_game", columns="risk_level", values="pct").fillna(0)
    pivot = pivot.reindex(columns=RISK_ORDER)
    pivot["high_risk"] = pivot["at_risk"] + pivot["problem"]
    pivot = pivot.sort_values("high_risk", ascending=False)
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot[RISK_ORDER].values,
        x=RISK_ORDER,
        y=pivot.index,
        colorscale=[
            [0, "#2ecc71"],
            [0.33, "#3498db"],
            [0.66, "#f39c12"],
            [1, "#e74c3c"]
        ],
        text=pivot[RISK_ORDER].values.round(1),
        texttemplate="%{text:.1f}%",
        textfont={"size": 10},
        hovertemplate="<b>%{y}</b><br>Risk: %{x}<br>Share: %{z:.1f}%<extra></extra>"
    ))
    
    fig.update_layout(
        title=dict(text="Risk Concentration by Game Type (%)", font_size=18, x=0.5),
        xaxis_title="Risk Level",
        yaxis_title="Game Type",
        height=600,
        yaxis=dict(categoryorder="array", categoryarray=pivot.index.tolist())
    )
    
    return fig

# ==================== DASH APP ====================

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    title="Gambling Risk Detection Dashboard"
)

server = app.server

# ==================== LAYOUT ====================

app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("🎰 Armenian Synthetic Gambling Behavior Dataset", 
                    className="text-center mt-4 mb-2 text-primary"),
            html.H4("Risk Detection & Behavioral Analysis", 
                    className="text-center text-muted mb-3"),
            html.P([
                "Team: Hayk Mnatsakanyan, Anush Khachatryan, Srbuhi Khachatryan, Mane Davtyan | ",
                "Supervisor: A. Pokrikyan | EPՀ Master's Program 2025–2026"
            ], className="text-center text-muted small mb-4"),
            html.Hr()
        ])
    ]),
    
    # Overview Section
    dbc.Row([
        dbc.Col([
            html.H2("📊 Overview", className="mt-3 mb-3 text-primary"),
            html.P([
                "This dashboard presents analysis of the ",
                html.Strong("Armenian Synthetic Gambling Behavior Dataset"),
                ", calibrated using real market data from ",
                html.Strong("SiGMA, ArmStat, and the State Revenue Committee (ՊԵԿ)"),
                ". The dataset simulates gambling behavior of 2,000 players over 180 days, "
                "with 1.65 million transactions and 9 behavioral indicators grounded in DSM-5 criteria."
            ], className="mb-4")
        ])
    ]),
    
    # KPI Cards
    create_overview_kpi_cards(),
    
    # Simulation period info
    dbc.Row([
        dbc.Col([
            dbc.Alert([
                html.I(className="fas fa-calendar-alt me-2"),
                f" Simulation Period: {sim_period['start_date']} → {sim_period['end_date']} ({sim_period['days']} days)",
                html.Br(),
                html.I(className="fas fa-chart-line me-2"),
                f" Behavioral Features: 9 | Transaction Types: bet (99.0%), deposit (0.7%), withdrawal (0.3%)"
            ], color="info", className="mb-4")
        ])
    ]),
    
    # Section 1: Risk Distribution
    dbc.Row([
        dbc.Col([
            html.H2("📈 Risk Class Distribution", className="mt-4 mb-3 text-primary"),
            html.P([
                "Players are classified into four risk levels based on DSM-5 criteria. ",
                f"High-risk players (at_risk + problem) constitute {positive_class_share:.1f}% of the population, "
                "aligning with the WHO global prevalence range of 2–5% for problem gambling."
            ], className="mb-3")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=create_risk_distribution_chart())
                ])
            ], className="shadow-sm mb-4")
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=create_bet_volume_chart())
                ])
            ], className="shadow-sm mb-4")
        ], md=6)
    ]),
    
    # Section 2: Behavioral Indicators
    dbc.Row([
        dbc.Col([
            html.H2("📐 Behavioral Indicators", className="mt-4 mb-3 text-primary"),
            html.P([
                "The 9 behavioral indicators are derived from transaction logs and ground-truth risk labels. "
                "They show strong monotonic progression across risk levels, with the most discriminative features "
                "being bet frequency, session duration, and total bet volume."
            ], className="mb-3")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=create_feature_comparison_chart())
                ])
            ], className="shadow-sm mb-4")
        ], md=12)
    ]),
    
    # Section 3: Demographics
    dbc.Row([
        dbc.Col([
            html.H2("👥 Demographics", className="mt-4 mb-3 text-primary"),
            html.P([
                "Demographic analysis reveals that Yerevan accounts for 72% of players, reflecting Armenia's "
                "urban concentration. Sports football is the most popular game type (31.8%), followed by "
                "casino slots (18.7%). Male dominance increases with risk level, from 77% in recreational "
                "to 93% in problem gamblers."
            ], className="mb-3")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=create_preferred_game_chart())
                ])
            ], className="shadow-sm mb-4")
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=create_region_chart())
                ])
            ], className="shadow-sm mb-4")
        ], md=6)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=create_gender_risk_chart())
                ])
            ], className="shadow-sm mb-4")
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=create_game_risk_heatmap())
                ])
            ], className="shadow-sm mb-4")
        ], md=6)
    ]),
    
    # Section 4: Monetary Analysis
    dbc.Row([
        dbc.Col([
            html.H2("💰 Monetary Analysis", className="mt-4 mb-3 text-primary"),
            html.P([
                f"The mean total bet per player ({stats['monetary_summary_amd']['total_bet_per_player']['mean']:,} AMD) "
                f"far exceeds the median ({stats['monetary_summary_amd']['total_bet_per_player']['median']:,} AMD), "
                "confirming extreme right skew driven by problem gamblers. The dataset mean income "
                f"({stats['monetary_summary_amd']['monthly_income']['mean']:,} AMD) exceeds the national average "
                f"({MARKET_CONTEXT['national_avg_income']:,} AMD), indicating selection bias typical of online gambling populations."
            ], className="mb-3")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=create_income_vs_bet_chart())
                ])
            ], className="shadow-sm mb-4")
        ], md=8),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Key Monetary Insights", className="mb-0")),
                dbc.CardBody([
                    html.Ul([
                        html.Li([
                            html.Strong("Problem Gamblers:"), 
                            " Mean bet 53.4M AMD, 472x recreational players"
                        ]),
                        html.Li([
                            html.Strong("Income Gradient:"), 
                            " Higher risk correlates with higher income"
                        ]),
                        html.Li([
                            html.Strong("Withdrawal Behavior:"), 
                            " High-risk players rarely withdraw funds"
                        ]),
                        html.Li([
                            html.Strong("Session Duration:"), 
                            " Problem players average 130 minutes (8.6x recreational)"
                        ])
                    ], className="mt-2")
                ])
            ], className="shadow-sm mb-4")
        ], md=4)
    ]),
    
    # Section 5: Model Performance
    dbc.Row([
        dbc.Col([
            html.H2("🎯 Model Performance Targets", className="mt-4 mb-3 text-primary"),
            html.P([
                "Based on the project proposal, the model must achieve the following performance thresholds. "
                "These metrics will be evaluated using 5-fold stratified cross-validation with SMOTE oversampling "
                "to address class imbalance (7.8:1 ratio)."
            ], className="mb-3")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=create_model_threshold_chart())
                ])
            ], className="shadow-sm mb-4")
        ], md=8),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Model Specifications", className="mb-0")),
                dbc.CardBody([
                    html.Ul([
                        html.Li("Algorithms: Logistic Regression, Random Forest, XGBoost"),
                        html.Li("Evaluation: 5-fold Stratified CV"),
                        html.Li("Sampling: SMOTE oversampling"),
                        html.Li("Features: 9 behavioral indicators"),
                        html.Li("Target: at_risk + problem (binary classification)"),
                        html.Li("Imbalance Ratio: 7.8:1 (negative:positive)")
                    ], className="mt-2")
                ])
            ], className="shadow-sm mb-4")
        ], md=4)
    ]),
    
    # Section 6: Temporal Patterns
    dbc.Row([
        dbc.Col([
            html.H2("⏰ Temporal Patterns", className="mt-4 mb-3 text-primary"),
            html.P([
                "Analysis of hourly betting patterns reveals that high-risk players are significantly "
                "more active during night-time hours (00:00–06:00), with problem gamblers showing "
                "the highest proportion of late-night activity."
            ], className="mb-3")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=create_temporal_trend_chart())
                ])
            ], className="shadow-sm mb-4")
        ], md=12)
    ]),
    
    # Footer
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P([
                "Data source: Synthetic Armenian Gambling Behavior Dataset v1.0 | ",
                "Calibrated using real market data | ",
                "Generated: 2026-06-05"
            ], className="text-center text-muted small"),
            html.P([
                "📧 Contact: ",
                html.A("hayk.mnatsakanyan@example.com", href="#"),
                " | ",
                html.A("anush.khachatryan@example.com", href="#")
            ], className="text-center text-muted small")
        ])
    ], className="mb-4")
], fluid=True, style={"backgroundColor": "#f8f9fa"})


# ==================== RUN APP ====================

if __name__ == "__main__":
    print("=" * 60)
    print("🎲 Gambling Risk Detection Dashboard")
    print("=" * 60)
    print(f"Dataset: {stats['dataset_name']} v{stats['version']}")
    print(f"Players: {total_players:,}")
    print(f"Transactions: {total_transactions:,}")
    print(f"Simulation: {sim_period['days']} days")
    print(f"High-risk share: {positive_class_share:.1f}%")
    print("=" * 60)
    print("Starting dashboard server...")
    print("Open http://localhost:8050 in your browser")
    print("=" * 60)
    
    app.run_server(debug=True, port=8050)