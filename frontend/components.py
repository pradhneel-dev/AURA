"""Small reusable visual components for the AURA Streamlit application."""
from html import escape
import streamlit as st


PALETTE = {"navy":"#14263d", "blue":"#2563eb", "ink":"#172033", "muted":"#64748b",
           "line":"#e2e8f0", "paper":"#ffffff", "canvas":"#f5f7fb", "green":"#15803d",
           "amber":"#a16207", "red":"#b91c1c"}


def apply_theme():
    st.markdown("""
    <style>
      .stApp { background: #f5f7fb; color: #172033; }
      .block-container { max-width: 1500px; padding-top: 1.5rem; padding-bottom: 3rem; }
      section[data-testid="stSidebar"] { background: #14263d; }
      section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] label,
      section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color: #e7edf5; }
      section[data-testid="stSidebar"] button { background: #203650; color: #f8fafc; border: 1px solid #405a76; }
      h1, h2, h3 { color: #172033; letter-spacing: -0.02em; }
      [data-testid="stMetric"] { background: #fff; border: 1px solid #e2e8f0; padding: 1rem 1.1rem; border-radius: .8rem; }
      [data-testid="stMetricLabel"] { color: #64748b; font-size: .82rem; line-height: 1.2; }
      [data-testid="stMetricValue"] { font-size: clamp(1.25rem, 2.3vw, 2rem); }
      div.stButton > button, div.stFormSubmitButton > button { border-radius: .55rem; min-height: 2.5rem; }
      div[data-testid="stDataFrame"] { border: 1px solid #e2e8f0; border-radius: .65rem; overflow: hidden; }
      .aura-eyebrow { color: #2563eb; text-transform: uppercase; letter-spacing: .11em; font-size: .72rem; font-weight: 700; margin-bottom: .25rem; }
      .aura-subtitle { color: #64748b; font-size: .98rem; margin-top: -.6rem; margin-bottom: 1.35rem; }
      .aura-brand { font-size: 1.35rem; font-weight: 750; letter-spacing: .025em; }
      .aura-brand-sub { font-size: .72rem; color: #bac8d8; line-height: 1.35; }
      .aura-status { display: inline-block; border-radius: 999px; padding: .2rem .62rem; font-size: .75rem; font-weight: 650; background: #e2e8f0; color: #334155; }
      .aura-status.good { background: #dcfce7; color: #166534; }
      .aura-status.pending { background: #fef3c7; color: #92400e; }
      .aura-status.bad { background: #fee2e2; color: #991b1b; }
      .aura-card { background: white; border: 1px solid #e2e8f0; border-radius: .8rem; padding: 1rem 1.1rem; margin: .25rem 0 .7rem 0; }
      .aura-muted { color: #64748b; font-size: .86rem; }
      @media (max-width: 800px) { .block-container { padding-left: 1rem; padding-right: 1rem; } }
    </style>
    """, unsafe_allow_html=True)


def page_header(title: str, description: str, eyebrow: str = "AURA WORKSPACE"):
    st.markdown(f'<div class="aura-eyebrow">{escape(eyebrow)}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="aura-subtitle">{escape(description)}</div>', unsafe_allow_html=True)


def status_badge(value: str):
    text = (value or "unknown").replace("_", " ").title()
    normalized = value.lower() if value else ""
    css = "good" if normalized in {"completed","approved","executed","success","low"} else (
        "bad" if normalized in {"failed","rejected","critical","high","execution_failed"} else "pending")
    st.markdown(f'<span class="aura-status {css}">{escape(text)}</span>', unsafe_allow_html=True)


def empty_state(message: str, action=None):
    st.info(message)
    if action:
        action()


def chart_layout(fig, height=330):
    fig.update_layout(height=height, margin={"l":12,"r":12,"t":35,"b":12},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color":"#334155"}, legend={"orientation":"h","yanchor":"bottom","y":1.02,"xanchor":"left","x":0})
    return fig
