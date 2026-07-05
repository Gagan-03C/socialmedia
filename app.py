# =============================================================================
# Social Media Engagement Analysis Dashboard
# A professional Streamlit application for Decision Tree ML project
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
import os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Social Media Engagement Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Dark Mode Professional Dashboard
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Import Google Fonts ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── Global Theme Overrides ───────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
    font-family: 'Inter', sans-serif;
}

body {
    background-color: #07120B !important;
}

.stApp {
    background-color: #07120B !important;
    background-image: 
        radial-gradient(circle at 50% 0%, rgba(58, 242, 143, 0.07) 0%, transparent 60%),
        radial-gradient(circle at 100% 100%, rgba(58, 242, 143, 0.03) 0%, transparent 40%) !important;
    color: #F5F7F9 !important;
}

/* Hide default Streamlit elements for cleaner look */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {
    background-color: transparent !important;
}
.stDeployButton {
    display: none !important;
}
[data-testid="stHeaderDecoration"] {
    display: none !important;
}

/* --- Main container spacing --- */
[data-testid="block-container"] {
    padding-top: 1.5rem !important;
    padding-bottom: 4rem !important;
    padding-left: 4rem !important;
    padding-right: 4rem !important;
    max-width: 1400px !important;
}

@media (max-width: 768px) {
    [data-testid="block-container"] {
        padding-left: 1.2rem !important;
        padding-right: 1.2rem !important;
    }
}

/* ── Sticky Top Navbar ─────────────────────────────────────────────── */
.top-navbar {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    padding: 14px 28px !important;
    background: rgba(13, 24, 18, 0.75) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(58, 242, 143, 0.15) !important;
    border-radius: 20px !important;
    margin-bottom: 30px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
}

.navbar-logo {
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
}

.navbar-logo .logo-icon {
    font-size: 1.8rem !important;
}

.navbar-logo .logo-text {
    font-family: 'Poppins', sans-serif !important;
    font-size: 1.3rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #3AF28F 0%, #5BFFB1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}

.navbar-actions {
    display: flex !important;
    align-items: center !important;
    gap: 20px !important;
}

.search-bar {
    position: relative !important;
    display: flex !important;
    align-items: center !important;
}

.search-bar .search-icon {
    position: absolute !important;
    left: 12px !important;
    color: #9FB0A7 !important;
    font-size: 0.9rem !important;
}

.search-bar input {
    background-color: #07120B !important;
    border: 1px solid rgba(58, 242, 143, 0.15) !important;
    border-radius: 20px !important;
    padding: 8px 16px 8px 36px !important;
    color: #F5F7F9 !important;
    font-size: 0.85rem !important;
    width: 180px !important;
    transition: all 0.3s ease !important;
}

.search-bar input:focus {
    width: 240px !important;
    border-color: #3AF28F !important;
    box-shadow: 0 0 10px rgba(58, 242, 143, 0.2) !important;
    outline: none !important;
}

.notification-btn {
    position: relative !important;
    cursor: pointer !important;
    width: 38px !important;
    height: 38px !important;
    border-radius: 50% !important;
    background: rgba(19, 33, 27, 0.6) !important;
    border: 1px solid rgba(58, 242, 143, 0.15) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: all 0.3s ease !important;
}

.notification-btn:hover {
    border-color: #3AF28F !important;
    box-shadow: 0 0 10px rgba(58, 242, 143, 0.2) !important;
}

.notification-icon {
    font-size: 1.1rem !important;
    color: #9FB0A7 !important;
}

.notification-badge {
    position: absolute !important;
    top: 2px !important;
    right: 2px !important;
    width: 8px !important;
    height: 8px !important;
    background-color: #3AF28F !important;
    border-radius: 50% !important;
    box-shadow: 0 0 6px #3AF28F !important;
}

.user-avatar {
    width: 38px !important;
    height: 38px !important;
    border-radius: 50% !important;
    overflow: hidden !important;
    border: 2px solid rgba(58, 242, 143, 0.3) !important;
}

.user-avatar img {
    width: 100% !important;
    height: 100% !important;
    object-fit: cover !important;
}

/* ── Gradient Header Banner ──────────────────────────────────────────── */
.gradient-header {
    background: linear-gradient(135deg, #0D1812 0%, #13211B 100%) !important;
    border: 1px solid rgba(58, 242, 143, 0.18) !important;
    padding: 32px 40px !important;
    border-radius: 20px !important;
    margin-bottom: 30px !important;
    position: relative !important;
    overflow: hidden !important;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4) !important;
}

.gradient-header::before {
    content: '' !important;
    position: absolute !important;
    top: -50% !important;
    left: -50% !important;
    width: 200% !important;
    height: 200% !important;
    background: radial-gradient(circle, rgba(58, 242, 143, 0.08) 0%, transparent 60%) !important;
    pointer-events: none !important;
}

.gradient-header h1 {
    font-family: 'Poppins', sans-serif !important;
    color: #3AF28F !important;
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    margin: 0 0 10px 0 !important;
    text-shadow: 0 0 20px rgba(58, 242, 143, 0.2) !important;
    letter-spacing: -0.5px !important;
}

.gradient-header p {
    font-family: 'Inter', sans-serif !important;
    color: #9FB0A7 !important;
    font-size: 1.05rem !important;
    margin: 0 !important;
    font-weight: 400 !important;
}

/* ── Sidebar Redesign ────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background-color: #07120B !important;
    border-right: 1px solid rgba(58, 242, 143, 0.15) !important;
}

section[data-testid="stSidebar"] > div {
    background-color: #0B1610 !important;
}

section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    padding: 24px 16px !important;
}

/* Custom Navigation Radio Buttons */
div[data-testid="stRadio"] > div[role="radiogroup"] {
    gap: 10px !important;
}

div[data-testid="stRadio"] > div[role="radiogroup"] > label {
    background-color: rgba(19, 33, 27, 0.4) !important;
    border: 1px solid rgba(58, 242, 143, 0.08) !important;
    border-radius: 12px !important;
    padding: 10px 14px !important;
    color: #9FB0A7 !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
    margin: 0 !important;
    width: 100% !important;
}

div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {
    display: none !important;
}

div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
    color: #5BFFB1 !important;
    background-color: rgba(58, 242, 143, 0.08) !important;
    border-color: rgba(58, 242, 143, 0.3) !important;
    box-shadow: 0 0 15px rgba(58, 242, 143, 0.1) !important;
    transform: translateX(4px) !important;
}

div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked),
div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"] {
    background: linear-gradient(135deg, rgba(58, 242, 143, 0.15) 0%, rgba(13, 24, 18, 0.6) 100%) !important;
    border: 1px solid rgba(58, 242, 143, 0.5) !important;
    color: #3AF28F !important;
    font-weight: 600 !important;
    box-shadow: 0 0 20px rgba(58, 242, 143, 0.15) !important;
    position: relative !important;
    padding-left: 20px !important;
}

div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked)::before,
div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"]::before {
    content: '' !important;
    position: absolute !important;
    left: 8px !important;
    width: 4px !important;
    height: 16px !important;
    background-color: #3AF28F !important;
    border-radius: 2px !important;
    box-shadow: 0 0 8px #3AF28F !important;
}

/* ── Metric / KPI Cards ──────────────────────────────────────────────── */
.metric-card {
    background: #13211B !important;
    border: 1px solid rgba(58, 242, 143, 0.15) !important;
    border-radius: 20px !important;
    padding: 20px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.25) !important;
    position: relative !important;
    overflow: hidden !important;
    margin-bottom: 16px !important;
}

.metric-card::before {
    content: '' !important;
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    width: 100% !important;
    height: 100% !important;
    background: radial-gradient(circle at 100% 0%, rgba(58, 242, 143, 0.05) 0%, transparent 60%) !important;
    pointer-events: none !important;
}

.metric-card:hover {
    transform: translateY(-4px) !important;
    border-color: rgba(58, 242, 143, 0.4) !important;
    box-shadow: 0 12px 40px rgba(58, 242, 143, 0.2) !important;
}

.metric-card-inner {
    display: flex !important;
    align-items: center !important;
    gap: 16px !important;
}

.metric-icon-wrapper {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 48px !important;
    height: 48px !important;
    background: rgba(58, 242, 143, 0.1) !important;
    border: 1px solid rgba(58, 242, 143, 0.2) !important;
    border-radius: 12px !important;
    font-size: 1.5rem !important;
    color: #3AF28F !important;
}

.metric-content {
    flex-grow: 1 !important;
}

.metric-label {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: #9FB0A7 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    margin-bottom: 4px !important;
}

.metric-value-container {
    display: flex !important;
    align-items: baseline !important;
    gap: 10px !important;
}

.metric-value {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: #F5F7F9 !important;
    letter-spacing: -0.5px !important;
}

.trend-badge {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    padding: 2px 8px !important;
    border-radius: 20px !important;
    display: inline-flex !important;
    align-items: center !important;
    white-space: nowrap !important;
}

.trend-badge.success {
    background-color: rgba(58, 242, 143, 0.15) !important;
    color: #3AF28F !important;
    border: 1px solid rgba(58, 242, 143, 0.3) !important;
}

.trend-badge.warning {
    background-color: rgba(255, 200, 87, 0.15) !important;
    color: #FFC857 !important;
    border: 1px solid rgba(255, 200, 87, 0.3) !important;
}

.trend-badge.danger {
    background-color: rgba(255, 92, 92, 0.15) !important;
    color: #FF5C5C !important;
    border: 1px solid rgba(255, 92, 92, 0.3) !important;
}

.trend-badge.info {
    background-color: rgba(0, 229, 255, 0.15) !important;
    color: #00E5FF !important;
    border: 1px solid rgba(0, 229, 255, 0.3) !important;
}

/* Native Metric Widgets Styling */
div[data-testid="stMetric"] {
    background: #13211B !important;
    border: 1px solid rgba(58, 242, 143, 0.15) !important;
    border-radius: 20px !important;
    padding: 18px !important;
    transition: transform 0.3s ease !important;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-4px) !important;
    box-shadow: 0 10px 30px rgba(58, 242, 143, 0.15) !important;
}

div[data-testid="stMetric"] label {
    color: #9FB0A7 !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    font-size: 0.78rem !important;
    letter-spacing: 1px;
}

div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #F5F7F9 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
}

/* ── Glass Cards ─────────────────────────────────────────────────────── */
.glass-card {
    background: rgba(19, 33, 27, 0.6) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(58, 242, 143, 0.15) !important;
    border-radius: 20px !important;
    padding: 28px !important;
    margin-bottom: 20px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
}

.glass-card:hover {
    border-color: rgba(58, 242, 143, 0.35) !important;
    box-shadow: 0 12px 40px rgba(58, 242, 143, 0.15) !important;
}

.glass-card h3 {
    font-family: 'Poppins', sans-serif !important;
    color: #3AF28F !important;
    font-weight: 600 !important;
    margin-bottom: 16px !important;
    font-size: 1.2rem !important;
}

/* ── Section Headers ─────────────────────────────────────────────────── */
.section-header {
    color: #3AF28F !important;
    font-family: 'Poppins', sans-serif !important;
    font-size: 1.35rem !important;
    font-weight: 600 !important;
    margin: 32px 0 18px 0 !important;
    padding-bottom: 8px !important;
    border-bottom: 1px solid rgba(58, 242, 143, 0.15) !important;
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
}

/* ── Styled Tables ───────────────────────────────────────────────────── */
.styled-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    background: #13211B !important;
    border-radius: 16px !important;
    overflow: hidden !important;
    font-size: 0.9rem !important;
    border: 1px solid rgba(58, 242, 143, 0.15) !important;
}

.styled-table th {
    background: rgba(58, 242, 143, 0.08) !important;
    color: #3AF28F !important;
    padding: 14px 18px !important;
    text-align: left !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    border-bottom: 1px solid rgba(58, 242, 143, 0.15) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

.styled-table td {
    padding: 12px 18px !important;
    border-bottom: 1px solid rgba(58, 242, 143, 0.08) !important;
    color: #F5F7F9 !important;
}

.styled-table tr:hover td {
    background: rgba(58, 242, 143, 0.04) !important;
}

/* ── Status Badges ───────────────────────────────────────────────────── */
.badge {
    display: inline-block !important;
    padding: 4px 12px !important;
    border-radius: 20px !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
}

.badge-success {
    background: rgba(58, 242, 143, 0.15) !important;
    color: #3AF28F !important;
    border: 1px solid rgba(58, 242, 143, 0.3) !important;
}

.badge-info {
    background: rgba(0, 229, 255, 0.15) !important;
    color: #00E5FF !important;
    border: 1px solid rgba(0, 229, 255, 0.3) !important;
}

.badge-warning {
    background: rgba(255, 200, 87, 0.15) !important;
    color: #FFC857 !important;
    border: 1px solid rgba(255, 200, 87, 0.3) !important;
}

/* ── Info Box ────────────────────────────────────────────────────────── */
.info-box {
    background: rgba(58, 242, 143, 0.05) !important;
    border-left: 4px solid #3AF28F !important;
    border-radius: 0 16px 16px 0 !important;
    padding: 18px 22px !important;
    margin: 20px 0 !important;
    color: #F5F7F9 !important;
    font-size: 0.92rem !important;
    line-height: 1.6 !important;
}

.info-box strong {
    color: #3AF28F !important;
}

/* ── Technology Chips ────────────────────────────────────────────────── */
.tech-chip {
    display: inline-block !important;
    background: rgba(58, 242, 143, 0.08) !important;
    border: 1px solid rgba(58, 242, 143, 0.2) !important;
    color: #3AF28F !important;
    padding: 6px 14px !important;
    border-radius: 20px !important;
    margin: 4px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
}

.tech-chip:hover {
    background: rgba(58, 242, 143, 0.15) !important;
    border-color: #3AF28F !important;
    box-shadow: 0 0 12px rgba(58, 242, 143, 0.25) !important;
    transform: translateY(-2px) !important;
}

/* ── Animated Pulse Dot ──────────────────────────────────────────────── */
.pulse-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #3AF28F;
    display: inline-block;
    margin-right: 8px;
    animation: pulse 2s ease-in-out infinite;
    box-shadow: 0 0 8px #3AF28F;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.8); }
}

/* ── Input overrides ─────────────────────────────────────────────────── */
.stSelectbox label, .stMultiSelect label, .stSlider label,
.stNumberInput label, .stTextInput label, .stFileUploader label {
    color: #9FB0A7 !important;
    font-weight: 500 !important;
}

/* Selectbox wrapper */
div[data-baseweb="select"] > div {
    background-color: #0D1812 !important;
    border: 1px solid rgba(58, 242, 143, 0.15) !important;
    border-radius: 12px !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.3s ease !important;
}

/* Ensure the selected option text in selectbox is visible and not truncated */
div[data-baseweb="select"] [data-testid="stSelectboxSelectedValue"],
div[data-baseweb="select"] > div > div > div,
div[data-baseweb="select"] div[aria-selected="true"],
div[data-baseweb="select"] span {
    color: #F5F7F9 !important;
    max-width: none !important;
    white-space: nowrap !important;
    overflow: visible !important;
    text-overflow: clip !important;
}

/* Text input, number input wrappers */
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input,
div[data-testid="stTextArea"] textarea {
    background-color: #0D1812 !important;
    color: #F5F7F9 !important;
    border: 1px solid rgba(58, 242, 143, 0.15) !important;
    border-radius: 12px !important;
    padding: 8px 12px !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.3s ease !important;
}



div[data-baseweb="select"] > div:focus-within,
div[data-testid="stTextInput"] input:focus,
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stTextArea"] textarea:focus {
    border-color: #3AF28F !important;
    box-shadow: 0 0 10px rgba(58, 242, 143, 0.25) !important;
    outline: none !important;
}

/* Dropdown list popup */
ul[role="listbox"] {
    background-color: #0D1812 !important;
    border: 1px solid rgba(58, 242, 143, 0.2) !important;
    border-radius: 12px !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5) !important;
}

ul[role="listbox"] li {
    color: #9FB0A7 !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.2s ease !important;
}

ul[role="listbox"] li:hover,
ul[role="listbox"] li[aria-selected="true"] {
    background-color: rgba(58, 242, 143, 0.1) !important;
    color: #3AF28F !important;
}

/* Slider styling */
div[data-testid="stSlider"] [role="slider"] {
    background-color: #3AF28F !important;
    border: 2px solid #07120B !important;
    box-shadow: 0 0 10px rgba(58, 242, 143, 0.5) !important;
    width: 20px !important;
    height: 20px !important;
}

div[data-testid="stSlider"] [data-presentation="true"] {
    background-color: rgba(58, 242, 143, 0.15) !important;
    height: 6px !important;
}

div[data-testid="stSlider"] div[role="none"] {
    background-color: #3AF28F !important;
}

/* File uploader container */
div[data-testid="stFileUploader"] {
    background: rgba(19, 33, 27, 0.5) !important;
    border: 2px dashed rgba(58, 242, 143, 0.25) !important;
    border-radius: 20px !important;
    padding: 30px 20px !important;
    transition: all 0.3s ease !important;
    text-align: center !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2) !important;
}

div[data-testid="stFileUploader"]:hover {
    border-color: #3AF28F !important;
    box-shadow: 0 0 25px rgba(58, 242, 143, 0.2) !important;
    background: rgba(19, 33, 27, 0.8) !important;
}

div[data-testid="stFileUploader"] section {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

div[data-testid="stFileUploader"] label {
    font-family: 'Poppins', sans-serif !important;
    font-size: 1.25rem !important;
    color: #F5F7F9 !important;
    margin-bottom: 12px !important;
}

div[data-testid="stFileUploader"] [data-testid="stFileUploaderIcon"] {
    color: #3AF28F !important;
    font-size: 3rem !important;
    margin-bottom: 16px !important;
    transform: scale(1.2);
}

div[data-testid="stFileUploader"] [data-testid="stUploadedFile"] {
    background-color: rgba(13, 24, 18, 0.8) !important;
    border: 1px solid rgba(58, 242, 143, 0.3) !important;
    border-radius: 12px !important;
    color: #F5F7F9 !important;
}

/* ── Buttons overrides ───────────────────────────────────────────────── */
div.stButton > button, .stDownloadButton > button, button[data-testid="baseButton-primary"], button[data-testid="baseButton-secondary"] {
    background: linear-gradient(135deg, #3AF28F 0%, #13211B 100%) !important;
    color: #07120B !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    border: 1px solid rgba(58, 242, 143, 0.3) !important;
    border-radius: 12px !important;
    padding: 10px 24px !important;
    font-size: 0.95rem !important;
    cursor: pointer !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 12px rgba(58, 242, 143, 0.1) !important;
    position: relative !important;
    overflow: hidden !important;
}

div.stButton > button:hover, .stDownloadButton > button:hover {
    background: linear-gradient(135deg, #5BFFB1 0%, #13211B 100%) !important;
    border-color: #5BFFB1 !important;
    box-shadow: 0 0 20px rgba(91, 255, 177, 0.4) !important;
    transform: translateY(-2px) !important;
    color: #07120B !important;
}

button[data-testid="baseButton-secondary"] {
    background: rgba(19, 33, 27, 0.6) !important;
    color: #F5F7F9 !important;
    border: 1px solid rgba(58, 242, 143, 0.15) !important;
}
button[data-testid="baseButton-secondary"]:hover {
    background: rgba(58, 242, 143, 0.1) !important;
    color: #5BFFB1 !important;
    border-color: rgba(58, 242, 143, 0.4) !important;
    box-shadow: 0 0 15px rgba(58, 242, 143, 0.15) !important;
}

/* ── Tabs & Expanders ────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px !important;
    background-color: #0D1812 !important;
    border-radius: 14px !important;
    padding: 6px !important;
    border: 1px solid rgba(58, 242, 143, 0.1) !important;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    color: #9FB0A7 !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    padding: 8px 16px !important;
    transition: all 0.3s ease !important;
    border: none !important;
}

.stTabs [aria-selected="true"] {
    background: #3AF28F !important;
    color: #07120B !important;
    box-shadow: 0 4px 12px rgba(58, 242, 143, 0.25) !important;
}

.streamlit-expanderHeader {
    background-color: #13211B !important;
    border: 1px solid rgba(58, 242, 143, 0.15) !important;
    border-radius: 12px !important;
    color: #F5F7F9 !important;
    font-family: 'Poppins', sans-serif !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}

.streamlit-expanderHeader:hover {
    border-color: rgba(58, 242, 143, 0.4) !important;
}

.stDataFrame {
    border-radius: 16px !important;
    overflow: hidden !important;
    border: 1px solid rgba(58, 242, 143, 0.15) !important;
}

/* ── About Cards ─────────────────────────────────────────────────────── */
.about-card {
    background: #13211B !important;
    border: 1px solid rgba(58, 242, 143, 0.15) !important;
    border-radius: 20px !important;
    padding: 24px !important;
    text-align: center !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.about-card:hover {
    transform: translateY(-4px) !important;
    border-color: rgba(58, 242, 143, 0.4) !important;
    box-shadow: 0 12px 30px rgba(58, 242, 143, 0.15) !important;
}

.about-card .about-icon {
    font-size: 2.5rem !important;
    margin-bottom: 12px !important;
    color: #3AF28F !important;
}

.about-card h4 {
    font-family: 'Poppins', sans-serif !important;
    color: #F5F7F9 !important;
    font-size: 1.1rem !important;
    margin: 8px 0 !important;
}

.about-card p {
    color: #9FB0A7 !important;
    font-size: 0.85rem !important;
    line-height: 1.6 !important;
}

/* ── Custom prediction card ─────────────────────────────────────────── */
.prediction-glow-card {
    background: linear-gradient(135deg, rgba(19, 33, 27, 0.9) 0%, rgba(13, 24, 18, 0.9) 100%) !important;
    border: 1px solid rgba(58, 242, 143, 0.3) !important;
    border-radius: 20px !important;
    padding: 24px !important;
    box-shadow: 0 0 30px rgba(58, 242, 143, 0.15) !important;
    position: relative !important;
    overflow: hidden !important;
    margin-bottom: 24px !important;
}

.prediction-card-header {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    border-bottom: 1px solid rgba(58, 242, 143, 0.1) !important;
    padding-bottom: 12px !important;
    margin-bottom: 20px !important;
}

.prediction-card-header h3 {
    margin: 0 !important;
    font-size: 1.2rem !important;
    color: #3AF28F !important;
}

.prediction-card-body {
    display: flex !important;
    align-items: center !important;
    gap: 30px !important;
    flex-wrap: wrap !important;
}

.circular-progress-wrapper {
    position: relative !important;
    width: 120px !important;
    height: 120px !important;
}

.circular-progress {
    width: 100% !important;
    height: 100% !important;
    transform: rotate(-90deg) !important;
}

.circular-progress circle {
    fill: none !important;
    stroke-width: 8 !important;
}

.circular-progress .bg-circle {
    stroke: rgba(58, 242, 143, 0.1) !important;
}

.circular-progress .fg-circle {
    stroke: #3AF28F !important;
    stroke-linecap: round !important;
    stroke-dasharray: 251.2 !important;
    transition: stroke-dashoffset 1s ease-in-out !important;
}

.progress-text {
    position: absolute !important;
    top: 50% !important;
    left: 50% !important;
    transform: translate(-50%, -50%) !important;
    text-align: center !important;
    width: 100% !important;
}

.progress-text .percentage {
    display: block !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: #F5F7F9 !important;
}

.progress-text .label {
    font-size: 0.65rem !important;
    color: #9FB0A7 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

.prediction-details {
    flex: 1 !important;
    min-width: 200px !important;
}

.prediction-metric {
    margin-bottom: 12px !important;
}

.prediction-metric .metric-label {
    display: block !important;
    font-size: 0.75rem !important;
    color: #9FB0A7 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    margin-bottom: 4px !important;
}

.prediction-metric .metric-val {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    color: #3AF28F !important;
}

.recommendation-text {
    margin: 0 !important;
    font-size: 0.9rem !important;
    color: #F5F7F9 !important;
    line-height: 1.5 !important;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Plotly chart dark theme
# ─────────────────────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#9FB0A7"),
    margin=dict(l=40, r=40, t=50, b=40),
    title_font=dict(size=18, color="#F5F7F9", family="Poppins, sans-serif"),
    legend=dict(
        bgcolor="rgba(19, 33, 27, 0.8)",
        bordercolor="rgba(58, 242, 143, 0.15)",
        borderwidth=1,
        font=dict(size=12, color="#F5F7F9"),
    ),
)

COLOR_PALETTE = [
    "#3AF28F", # Neon Green
    "#5BFFB1", # Light Green Hover Accent
    "#00E5FF", # Neon Cyan
    "#FFC857", # Gold Warning
    "#FF5C5C", # Danger Red
    "#8A2BE2", # Purple
    "#20C997", # Teal
    "#9FB0A7", # Secondary Text
]

def apply_chart_style(fig, height=450):
    """Apply consistent dark-mode styling to a Plotly figure."""
    fig.update_layout(**PLOTLY_LAYOUT, height=height)
    fig.update_xaxes(gridcolor="rgba(58, 242, 143, 0.08)", zeroline=False, tickfont=dict(color="#9FB0A7"))
    fig.update_yaxes(gridcolor="rgba(58, 242, 143, 0.08)", zeroline=False, tickfont=dict(color="#9FB0A7"))
    # Make line charts smoother
    for trace in fig.data:
        if trace.type == "scatter" and hasattr(trace, "line") and trace.line:
            trace.line.shape = "spline"
            trace.line.smoothing = 1.3
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Render metric card HTML
# ─────────────────────────────────────────────────────────────────────────────

def metric_card(icon, value, label, show_trend=True):
    """Return HTML for a single animated premium KPI card."""
    trend_html = ""
    lbl_upper = label.upper()
    
    if show_trend:
        if "ROWS" in lbl_upper:
            trend_html = '<span class="trend-badge success">↗ 12%</span>'
        elif "COLUMNS" in lbl_upper:
            trend_html = '<span class="trend-badge success">Stable</span>'
        elif "MISSING" in lbl_upper:
            if str(value) == "0":
                trend_html = '<span class="trend-badge success">Perfect</span>'
            else:
                trend_html = f'<span class="trend-badge warning">{value} Missing</span>'
        elif "DUPLICATE" in lbl_upper:
            if str(value) == "0":
                trend_html = '<span class="trend-badge success">Clean</span>'
            else:
                trend_html = f'<span class="trend-badge danger">{value} Dups</span>'
        elif "ACCURACY" in lbl_upper:
            trend_html = '<span class="trend-badge success">High</span>'
        elif "HIGH ENGAGEMENT" in lbl_upper:
            trend_html = '<span class="trend-badge success">Yes</span>'
        elif "LOW ENGAGEMENT" in lbl_upper:
            trend_html = '<span class="trend-badge danger">No</span>'
        elif "PLATFORM" in lbl_upper:
            trend_html = '<span class="trend-badge info">Active</span>'
        elif "SENTIMENT" in lbl_upper:
            trend_html = '<span class="trend-badge info">Positive</span>'
        elif "LIKES" in lbl_upper:
            trend_html = '<span class="trend-badge success">↗ 8%</span>'
        elif "SHARES" in lbl_upper:
            trend_html = '<span class="trend-badge success">↗ 5%</span>'
        elif "COMMENTS" in lbl_upper:
            trend_html = '<span class="trend-badge success">↗ 15%</span>'
        elif "IMPORTANCE" in lbl_upper or "FEATURE" in lbl_upper:
            trend_html = '<span class="trend-badge info">Weight</span>'
        
    html_content = (
        f'<div class="metric-card">'
        f'<div class="metric-card-inner">'
        f'<div class="metric-icon-wrapper">{icon}</div>'
        f'<div class="metric-content">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value-container">'
        f'<span class="metric-value">{value}</span>'
        f'{trend_html}'
        f'</div>'
        f'</div>'
        f'</div>'
        f'</div>'
    )
    return html_content


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL & ENCODERS (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model_and_encoders():
    """Load the trained Decision Tree model and label encoders from disk."""
    base = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(base, "model.pkl"))
    encoders = joblib.load(os.path.join(base, "encoders.pkl"))
    return model, encoders


try:
    model, encoders = load_model_and_encoders()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    st.error(f"⚠️ Could not load model files: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DEFAULT DATASET (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_default_dataset():
    """Attempt to load the bundled CSV dataset for analytics pages."""
    base = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base, "Social Media Engagement Dataset .csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None


default_df = load_default_dataset()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px 0;">
        <div style="font-size:2.5rem;">📊</div>
        <h2 style="color:#e2e8f0; font-size:1.15rem; margin:8px 0 2px 0;">
            Social Media<br>Engagement Dashboard
        </h2>
        <p style="color:#8b8fa3; font-size:0.78rem; margin:0;">
            Decision Tree ML Project
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        [
            "🏠  Home",
            "🔍  Dataset Explorer",
            "📊  Exploratory Data Analysis",
            "🤖  ML Prediction",
            "🌳  Feature Importance",
            "📈  Dashboard Insights",
            "🎨  Visual Analytics",
            "ℹ️  About",
        ],
        label_visibility="collapsed",
    )

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # Status indicator
    status = "Online" if MODEL_LOADED else "Model Error"
    color = "#34d399" if MODEL_LOADED else "#f87171"
    st.markdown(
        f'<div style="text-align:center; padding:8px 0;">'
        f'<span class="pulse-dot" style="background:{color};"></span>'
        f'<span style="color:{color}; font-size:0.82rem; font-weight:600;">'
        f"System {status}</span></div>",
        unsafe_allow_html=True,
    )

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═════════════════════════════════════════════════════════════════════════════

if page == "🏠  Home":
    # ── Header
    st.markdown("""
    <div class="gradient-header">
        <h1>📊 Social Media Engagement Dashboard</h1>
        <p>Predicting post engagement using Decision Tree Classification &mdash;
        A Machine Learning powered analytics platform</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Project description
    st.markdown("""
    <div class="glass-card">
        <h3>🚀 Project Overview</h3>
        <p style="color:#c9d1d9; line-height:1.7; font-size:0.95rem;">
        This application analyses social media engagement data and predicts whether a
        post will achieve <strong style="color:#818cf8;">High Engagement</strong> or
        <strong style="color:#f472b6;">Low Engagement</strong> using a trained
        <strong>Decision Tree Classifier</strong>. Posts with an engagement rate ≥ 0.10
        are classified as <em>High Engagement (Yes)</em>, otherwise
        <em>Low Engagement (No)</em>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Algorithm card
    st.markdown("""
    <div class="glass-card">
        <h3>🌳 Algorithm — Decision Tree Classifier</h3>
        <p style="color:#c9d1d9; line-height:1.7; font-size:0.95rem;">
        Decision Trees work by recursively splitting data based on feature thresholds
        that maximise information gain (Gini impurity). The resulting tree structure
        makes the model highly interpretable — ideal for understanding which social
        media features drive engagement the most.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Dataset overview metrics
    st.markdown('<div class="section-header">📋 Dataset Overview</div>', unsafe_allow_html=True)

    if default_df is not None:
        total_rows = default_df.shape[0]
        total_cols = default_df.shape[1]
        num_features = default_df.select_dtypes(include=np.number).shape[1]
        cat_features = default_df.select_dtypes(include="object").shape[1]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(metric_card("📑", f"{total_rows:,}", "Total Rows"), unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card("📐", total_cols, "Total Columns"), unsafe_allow_html=True)
        with c3:
            st.markdown(metric_card("🔢", num_features, "Numerical Features"), unsafe_allow_html=True)
        with c4:
            st.markdown(metric_card("🏷️", cat_features, "Categorical Features"), unsafe_allow_html=True)
    else:
        st.error("⚠️ Default dataset not found. Please ensure the 'Social Media Engagement Dataset .csv' file is in the project folder.")

    # ── Quick stats
    if default_df is not None:
        st.markdown('<div class="section-header">⚡ Quick Statistics</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(metric_card("🌐", default_df["platform"].nunique() if "platform" in default_df.columns else "—", "Platforms"), unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card("🗣️", default_df["language"].nunique() if "language" in default_df.columns else "—", "Languages"), unsafe_allow_html=True)
        with c3:
            st.markdown(metric_card("👤", default_df["user_id"].nunique() if "user_id" in default_df.columns else "—", "Unique Users"), unsafe_allow_html=True)
        with c4:
            st.markdown(metric_card("🎯", default_df["campaign_name"].nunique() if "campaign_name" in default_df.columns else "—", "Campaigns"), unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: DATASET EXPLORER
# ═════════════════════════════════════════════════════════════════════════════

elif page == "🔍  Dataset Explorer":
    st.markdown("""
    <div class="gradient-header">
        <h1>🔍 Dataset Explorer</h1>
        <p>Preview and profile your dataset — understand every column at a glance</p>
    </div>
    """, unsafe_allow_html=True)

    df = None
    if default_df is not None:
        df = default_df.copy()
    else:
        st.error("⚠️ Default dataset not found. Please ensure the 'Social Media Engagement Dataset .csv' file is in the project folder.")

    if df is not None:
        # ── Preview
        st.markdown('<div class="section-header">👁️ Data Preview</div>', unsafe_allow_html=True)
        preview_n = st.slider("Rows to preview", 5, 100, 10)
        st.dataframe(df.head(preview_n), use_container_width=True, height=400)

        # ── Column Names
        st.markdown('<div class="section-header">🏷️ Column Names</div>', unsafe_allow_html=True)
        cols_html = " ".join(f'<span class="tech-chip">{c}</span>' for c in df.columns)
        st.markdown(f'<div style="margin-bottom:15px;">{cols_html}</div>', unsafe_allow_html=True)

        # ── Summary cards
        st.markdown('<div class="section-header">📊 Dataset Summary</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(metric_card("📑", f"{df.shape[0]:,}", "Total Rows"), unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card("📐", df.shape[1], "Total Columns"), unsafe_allow_html=True)
        with c3:
            missing_total = int(df.isnull().sum().sum())
            st.markdown(metric_card("⚠️", f"{missing_total:,}", "Missing Values"), unsafe_allow_html=True)
        with c4:
            dup = int(df.duplicated().sum())
            st.markdown(metric_card("🔄", f"{dup:,}", "Duplicate Rows"), unsafe_allow_html=True)

        # ── Data Types
        st.markdown('<div class="section-header">🧬 Data Types</div>', unsafe_allow_html=True)
        dtype_df = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Non-Null Count": df.notnull().sum().values,
            "Missing": df.isnull().sum().values,
            "Unique": df.nunique().values,
        })
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)

        # ── Missing values visualisation
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            st.markdown('<div class="section-header">🕳️ Missing Values</div>', unsafe_allow_html=True)
            fig = px.bar(
                x=missing.values,
                y=missing.index,
                orientation="h",
                labels={"x": "Count", "y": "Column"},
                color_discrete_sequence=["#f472b6"],
            )
            fig.update_layout(title="Missing Values per Column")
            st.plotly_chart(apply_chart_style(fig, 350), use_container_width=True)

        # ── Descriptive statistics
        st.markdown('<div class="section-header">📈 Descriptive Statistics</div>', unsafe_allow_html=True)
        st.dataframe(df.describe().T.round(3), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: EXPLORATORY DATA ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

elif page == "📊  Exploratory Data Analysis":
    st.markdown("""
    <div class="gradient-header">
        <h1>📊 Exploratory Data Analysis</h1>
        <p>Interactive charts to uncover patterns, distributions and relationships in the data</p>
    </div>
    """, unsafe_allow_html=True)

    df = default_df

    if df is None:
        st.error("⚠️ Default dataset not found. Please ensure the 'Social Media Engagement Dataset .csv' file is in the project folder.")
        st.stop()

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    # ── Chart type tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Histogram", "📦 Box Plot", "🔵 Scatter", "🔥 Heatmap",
        "🥧 Pie Chart", "📊 Bar Chart", "🔢 Count Plot", "📈 Line Chart"
    ])

    # ── HISTOGRAM
    with tab1:
        st.markdown('<div class="section-header">📊 Histogram</div>', unsafe_allow_html=True)
        col = st.selectbox("Select column", num_cols, key="hist_col")
        bins = st.slider("Number of bins", 10, 100, 30, key="hist_bins")
        fig = px.histogram(
            df, x=col, nbins=bins, color_discrete_sequence=["#818cf8"],
            marginal="box", title=f"Distribution of {col}",
        )
        st.plotly_chart(apply_chart_style(fig), use_container_width=True)

    # ── BOX PLOT
    with tab2:
        st.markdown('<div class="section-header">📦 Box Plot</div>', unsafe_allow_html=True)
        col = st.selectbox("Select numerical column", num_cols, key="box_col")
        group = st.selectbox("Group by (optional)", ["None"] + cat_cols, key="box_grp")
        if group == "None":
            fig = px.box(df, y=col, color_discrete_sequence=["#c084fc"], title=f"Box Plot — {col}")
        else:
            fig = px.box(df, x=group, y=col, color=group, color_discrete_sequence=COLOR_PALETTE, title=f"{col} by {group}")
        st.plotly_chart(apply_chart_style(fig), use_container_width=True)

    # ── SCATTER PLOT
    with tab3:
        st.markdown('<div class="section-header">🔵 Scatter Plot</div>', unsafe_allow_html=True)
        sc1, sc2 = st.columns(2)
        with sc1:
            x_col = st.selectbox("X axis", num_cols, key="scat_x")
        with sc2:
            y_col = st.selectbox("Y axis", num_cols, index=min(1, len(num_cols)-1), key="scat_y")
        color_col = st.selectbox("Color by (optional)", ["None"] + cat_cols, key="scat_clr")
        fig = px.scatter(
            df, x=x_col, y=y_col,
            color=None if color_col == "None" else color_col,
            color_discrete_sequence=COLOR_PALETTE,
            opacity=0.6, title=f"{y_col} vs {x_col}",
        )
        st.plotly_chart(apply_chart_style(fig, 500), use_container_width=True)

    # ── CORRELATION HEATMAP
    with tab4:
        st.markdown('<div class="section-header">🔥 Correlation Heatmap</div>', unsafe_allow_html=True)
        corr = df[num_cols].corr().round(2)
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="Viridis",
            text=corr.values,
            texttemplate="%{text}",
            textfont=dict(size=10),
            hoverongaps=False,
        ))
        fig.update_layout(title="Feature Correlation Matrix")
        st.plotly_chart(apply_chart_style(fig, 600), use_container_width=True)

    # ── PIE CHART
    with tab5:
        st.markdown('<div class="section-header">🥧 Pie Chart</div>', unsafe_allow_html=True)
        pie_col = st.selectbox("Select categorical column", cat_cols, key="pie_col")
        counts = df[pie_col].value_counts().head(12)
        fig = px.pie(
            names=counts.index, values=counts.values,
            color_discrete_sequence=COLOR_PALETTE,
            title=f"Distribution of {pie_col}", hole=0.4,
        )
        st.plotly_chart(apply_chart_style(fig, 480), use_container_width=True)

    # ── BAR CHART
    with tab6:
        st.markdown('<div class="section-header">📊 Bar Chart</div>', unsafe_allow_html=True)
        bar_cat = st.selectbox("Categorical column", cat_cols, key="bar_cat")
        bar_num = st.selectbox("Numerical column (mean)", num_cols, key="bar_num")
        bar_data = df.groupby(bar_cat)[bar_num].mean().sort_values(ascending=False).head(15).reset_index()
        fig = px.bar(
            bar_data, x=bar_cat, y=bar_num,
            color=bar_num, color_continuous_scale="Viridis",
            title=f"Average {bar_num} by {bar_cat}",
        )
        st.plotly_chart(apply_chart_style(fig), use_container_width=True)

    # ── COUNT PLOT
    with tab7:
        st.markdown('<div class="section-header">🔢 Count Plot</div>', unsafe_allow_html=True)
        cnt_col = st.selectbox("Select column", cat_cols, key="cnt_col")
        counts = df[cnt_col].value_counts().head(15).reset_index()
        counts.columns = [cnt_col, "count"]
        fig = px.bar(
            counts, x=cnt_col, y="count",
            color_discrete_sequence=["#f472b6"],
            title=f"Count of {cnt_col}",
        )
        st.plotly_chart(apply_chart_style(fig), use_container_width=True)

    # ── LINE CHART
    with tab8:
        st.markdown('<div class="section-header">📈 Line Chart</div>', unsafe_allow_html=True)
        line_col = st.selectbox("Select numerical column", num_cols, key="line_col")
        sample_size = st.slider("Sample size", 100, min(2000, len(df)), min(500, len(df)), key="line_n")
        sample_df = df[line_col].dropna().head(sample_size).reset_index(drop=True)
        fig = px.line(
            sample_df, title=f"Trend — {line_col}",
            color_discrete_sequence=["#60a5fa"],
        )
        fig.update_traces(line=dict(width=1.5))
        st.plotly_chart(apply_chart_style(fig), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: ML PREDICTION
# ═════════════════════════════════════════════════════════════════════════════

elif page == "🤖  ML Prediction":
    st.markdown("""
    <div class="gradient-header">
        <h1>🤖 Machine Learning Prediction</h1>
        <p>Upload a CSV, run the Decision Tree model and download predictions</p>
    </div>
    """, unsafe_allow_html=True)

    if not MODEL_LOADED:
        st.error("Model files could not be loaded. Please ensure `model.pkl` and `encoders.pkl` are present.")
        st.stop()

    # ── CSV Schema Guide
    with st.expander("📋 CSV File Requirements — Click to expand before uploading", expanded=False):
        st.markdown("""
        <div style="padding: 4px 0;">
            <p style="color:#9FB0A7; font-size:0.93rem; margin-bottom:16px;">
                Your CSV must contain the columns listed below. Unknown category values are handled gracefully
                (encoded as <code>-1</code>). Missing columns are filled with <code>0</code> automatically.
                <br><br>
                <span style="color:#3AF28F; font-weight:600;">💡 Tip:</span>
                The bundled <code>Social Media Engagement Dataset .csv</code> already has all the correct columns
                and can be used directly for testing.
            </p>
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("""
            <div style="background:rgba(19,33,27,0.7); border:1px solid rgba(58,242,143,0.15);
                        border-radius:14px; padding:18px;">
                <div style="color:#3AF28F; font-weight:700; font-size:0.9rem;
                            text-transform:uppercase; letter-spacing:1px; margin-bottom:12px;">
                    📝 Text / Categorical Columns
                </div>
                <table style="width:100%; font-size:0.83rem; border-collapse:collapse;">
                    <tr style="border-bottom:1px solid rgba(58,242,143,0.1);">
                        <td style="padding:6px 8px; color:#3AF28F; font-weight:600;">post_id</td>
                        <td style="padding:6px 8px; color:#9FB0A7;">Unique post identifier</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(58,242,143,0.1);">
                        <td style="padding:6px 8px; color:#3AF28F; font-weight:600;">timestamp</td>
                        <td style="padding:6px 8px; color:#9FB0A7;">e.g. 2024-12-09 11:26:15</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(58,242,143,0.1);">
                        <td style="padding:6px 8px; color:#3AF28F; font-weight:600;">day_of_week</td>
                        <td style="padding:6px 8px; color:#9FB0A7;">Monday … Sunday</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(58,242,143,0.1);">
                        <td style="padding:6px 8px; color:#3AF28F; font-weight:600;">platform</td>
                        <td style="padding:6px 8px; color:#9FB0A7;">Instagram / Twitter / YouTube / Reddit</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(58,242,143,0.1);">
                        <td style="padding:6px 8px; color:#3AF28F; font-weight:600;">user_id</td>
                        <td style="padding:6px 8px; color:#9FB0A7;">Unique user identifier</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(58,242,143,0.1);">
                        <td style="padding:6px 8px; color:#3AF28F; font-weight:600;">location</td>
                        <td style="padding:6px 8px; color:#9FB0A7;">e.g. Melbourne, Australia</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(58,242,143,0.1);">
                        <td style="padding:6px 8px; color:#3AF28F; font-weight:600;">language</td>
                        <td style="padding:6px 8px; color:#9FB0A7;">ISO code: en, pt, ru, hi …</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(58,242,143,0.1);">
                        <td style="padding:6px 8px; color:#3AF28F; font-weight:600;">text_content</td>
                        <td style="padding:6px 8px; color:#9FB0A7;">Full post text</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(58,242,143,0.1);">
                        <td style="padding:6px 8px; color:#3AF28F; font-weight:600;">hashtags</td>
                        <td style="padding:6px 8px; color:#9FB0A7;">#nike #review …</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(58,242,143,0.1);">
                        <td style="padding:6px 8px; color:#3AF28F; font-weight:600;">keywords</td>
                        <td style="padding:6px 8px; color:#9FB0A7;">nike, review …</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(58,242,143,0.1);">
                        <td style="padding:6px 8px; color:#3AF28F; font-weight:600;">topic_category</td>
                        <td style="padding:6px 8px; color:#9FB0A7;">Technology, Sports …</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(58,242,143,0.1);">
                        <td style="padding:6px 8px; color:#3AF28F; font-weight:600;">sentiment_label</td>
                        <td style="padding:6px 8px; color:#9FB0A7;">Positive / Negative / Neutral</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(58,242,143,0.1);">
                        <td style="padding:6px 8px; color:#3AF28F; font-weight:600;">emotion_type</td>
                        <td style="padding:6px 8px; color:#9FB0A7;">Joy, Anger, Sadness …</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(58,242,143,0.1);">
                        <td style="padding:6px 8px; color:#3AF28F; font-weight:600;">brand_name</td>
                        <td style="padding:6px 8px; color:#9FB0A7;">Nike, Pepsi …</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(58,242,143,0.1);">
                        <td style="padding:6px 8px; color:#3AF28F; font-weight:600;">product_name</td>
                        <td style="padding:6px 8px; color:#9FB0A7;">Air Max, Coke Zero …</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(58,242,143,0.1);">
                        <td style="padding:6px 8px; color:#3AF28F; font-weight:600;">campaign_name</td>
                        <td style="padding:6px 8px; color:#9FB0A7;">Summer Sale …</td>
                    </tr>
                    <tr>
                        <td style="padding:6px 8px; color:#3AF28F; font-weight:600;">campaign_phase</td>
                        <td style="padding:6px 8px; color:#9FB0A7;">Launch, Growth …</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            st.markdown("""
            <div style="background:rgba(19,33,27,0.7); border:1px solid rgba(58,242,143,0.15);
                        border-radius:14px; padding:18px; margin-bottom:16px;">
                <div style="color:#00E5FF; font-weight:700; font-size:0.9rem;
                            text-transform:uppercase; letter-spacing:1px; margin-bottom:12px;">
                    🔢 Numerical Columns
                </div>
                <table style="width:100%; font-size:0.83rem; border-collapse:collapse;">
                    <tr style="border-bottom:1px solid rgba(0,229,255,0.1);">
                        <td style="padding:6px 8px; color:#00E5FF; font-weight:600;">sentiment_score</td>
                        <td style="padding:6px 8px; color:#9FB0A7;">Float, e.g. 0.75</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(0,229,255,0.1);">
                        <td style="padding:6px 8px; color:#00E5FF; font-weight:600;">toxicity_score</td>
                        <td style="padding:6px 8px; color:#9FB0A7;">Float, e.g. 0.12</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(0,229,255,0.1);">
                        <td style="padding:6px 8px; color:#00E5FF; font-weight:600;">likes_count</td>
                        <td style="padding:6px 8px; color:#9FB0A7;">Integer, e.g. 542</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(0,229,255,0.1);">
                        <td style="padding:6px 8px; color:#00E5FF; font-weight:600;">shares_count</td>
                        <td style="padding:6px 8px; color:#9FB0A7;">Integer, e.g. 88</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(0,229,255,0.1);">
                        <td style="padding:6px 8px; color:#00E5FF; font-weight:600;">comments_count</td>
                        <td style="padding:6px 8px; color:#9FB0A7;">Integer, e.g. 34</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(0,229,255,0.1);">
                        <td style="padding:6px 8px; color:#00E5FF; font-weight:600;">impressions</td>
                        <td style="padding:6px 8px; color:#9FB0A7;">Integer, e.g. 12000</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(0,229,255,0.1);">
                        <td style="padding:6px 8px; color:#00E5FF; font-weight:600;">user_past_sentiment_avg</td>
                        <td style="padding:6px 8px; color:#9FB0A7;">Float, e.g. 0.65</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(0,229,255,0.1);">
                        <td style="padding:6px 8px; color:#00E5FF; font-weight:600;">user_engagement_growth</td>
                        <td style="padding:6px 8px; color:#9FB0A7;">Float, e.g. 0.03</td>
                    </tr>
                    <tr>
                        <td style="padding:6px 8px; color:#00E5FF; font-weight:600;">buzz_change_rate</td>
                        <td style="padding:6px 8px; color:#9FB0A7;">Float, e.g. 1.25</td>
                    </tr>
                </table>
            </div>

            <div style="background:rgba(19,33,27,0.7); border:1px solid rgba(255,200,87,0.25);
                        border-radius:14px; padding:18px;">
                <div style="color:#FFC857; font-weight:700; font-size:0.9rem;
                            text-transform:uppercase; letter-spacing:1px; margin-bottom:12px;">
                    ⚙️ Optional / Auto-handled Columns
                </div>
                <table style="width:100%; font-size:0.83rem; border-collapse:collapse;">
                    <tr style="border-bottom:1px solid rgba(255,200,87,0.1);">
                        <td style="padding:6px 8px; color:#FFC857; font-weight:600;">engagement_rate</td>
                        <td style="padding:6px 8px; color:#9FB0A7;">
                            Optional — if present, the app evaluates model accuracy against actual values
                        </td>
                    </tr>
                    <tr>
                        <td style="padding:6px 8px; color:#FFC857; font-weight:600;">mentions</td>
                        <td style="padding:6px 8px; color:#9FB0A7;">
                            Automatically dropped (not used during training)
                        </td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

    uploaded = st.file_uploader("📁 Upload CSV for prediction", type=["csv"], key="pred_upload")

    if uploaded:
        raw_df = pd.read_csv(uploaded)
        st.markdown('<div class="section-header">📋 Uploaded Data Preview</div>', unsafe_allow_html=True)
        st.dataframe(raw_df.head(10), use_container_width=True)

        st.markdown('<div class="section-header">⚙️ Preprocessing & Prediction</div>', unsafe_allow_html=True)

        try:
            proc_df = raw_df.copy()

            # ── Store actual engagement_rate if present (for evaluation)
            has_target = "engagement_rate" in proc_df.columns
            if has_target:
                actual_engagement = proc_df["engagement_rate"].copy()
                # Binarise the same way as training
                actual_labels = np.where(actual_engagement >= 0.1, "Yes", "No")
                proc_df.drop(columns=["engagement_rate"], inplace=True)

            # ── Drop 'mentions' column (was dropped during training)
            if "mentions" in proc_df.columns:
                proc_df.drop(columns=["mentions"], inplace=True)

            # ── Get expected feature names from the model
            expected_features = list(model.feature_names_in_)

            # ── Drop columns not needed by the model (saves encoding time)
            cols_to_keep = [c for c in proc_df.columns if c in expected_features or c in encoders]
            proc_df = proc_df[[c for c in cols_to_keep if c in proc_df.columns]]

            # ── Encode categorical columns — fast vectorized map (50-100x faster than apply)
            for col in list(proc_df.columns):
                if pd.api.types.is_string_dtype(proc_df[col]) and col in encoders:
                    le = encoders[col]
                    # Build a lookup dict: known label → encoded int, unknown → -1
                    lookup = {cls: idx for idx, cls in enumerate(le.classes_)}
                    proc_df[col] = proc_df[col].map(lookup).fillna(-1).astype(int)

            # ── Ensure all expected columns exist, fill missing with 0
            for feat in expected_features:
                if feat not in proc_df.columns:
                    proc_df[feat] = 0

            # ── Reorder to match training feature order
            proc_df = proc_df[expected_features]

            # ── Nuclear coercion: force everything to numeric so sklearn never sees strings
            proc_df = proc_df.apply(pd.to_numeric, errors="coerce").fillna(-1)

            # ── Predict
            predictions = model.predict(proc_df)
            probabilities = model.predict_proba(proc_df)

            st.success("✅ Prediction completed successfully!")

            # Calculate predictions details
            yes_count = int((predictions == "Yes").sum())
            no_count = int((predictions == "No").sum())
            total = len(predictions)
            percentage = int(round((yes_count / total) * 100)) if total > 0 else 0
            stroke_offset = 251.2 - (251.2 * percentage / 100)
            confidence_avg = int(round(probabilities.max(axis=1).mean() * 100))
            
            if percentage >= 70:
                recommendation = "Excellent optimization! The majority of your posts have High Engagement potential. Continue with your current content strategies, and focus on high-performing sentiment categories."
            elif percentage >= 40:
                recommendation = "Moderate engagement potential. A significant portion of posts are flagged as Low Engagement. Consider adjusting topics, increasing positive sentiment, or targeting optimal platforms."
            else:
                recommendation = "Optimization required. Most posts are predicted to receive Low Engagement. Review toxicity levels, emotional tone, and campaign alignment to boost performance."

            # Glowing result card with SVG circular indicator
            st.markdown(f"""
            <div class="prediction-glow-card">
                <div class="prediction-card-header">
                    <h3>🔮 Machine Learning Predictive Analysis</h3>
                    <div style="display: flex; align-items: center;">
                        <span class="pulse-dot"></span>
                        <span style="color: #3AF28F; font-size: 0.82rem; font-weight: 600;">DT Classifier Active</span>
                    </div>
                </div>
                <div class="prediction-card-body">
                    <div class="circular-progress-wrapper">
                        <svg class="circular-progress" viewBox="0 0 100 100">
                            <circle class="bg-circle" cx="50" cy="50" r="40"></circle>
                            <circle class="fg-circle" cx="50" cy="50" r="40" style="stroke-dashoffset: {stroke_offset};"></circle>
                        </svg>
                        <div class="progress-text">
                            <span class="percentage">{percentage}%</span>
                            <span class="label">High Engage</span>
                        </div>
                    </div>
                    <div class="prediction-details">
                        <div class="prediction-metric">
                            <span class="metric-label">Model Confidence (Avg)</span>
                            <span class="metric-val">{confidence_avg}%</span>
                        </div>
                        <div class="prediction-metric">
                            <span class="metric-label">Recommendation</span>
                            <p class="recommendation-text">{recommendation}</p>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Results summary cards
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(metric_card("📊", total, "Total Predictions"), unsafe_allow_html=True)
            with c2:
                st.markdown(metric_card("✅", yes_count, "High Engagement"), unsafe_allow_html=True)
            with c3:
                st.markdown(metric_card("❌", no_count, "Low Engagement"), unsafe_allow_html=True)

            # ── Prediction distribution pie
            fig = px.pie(
                names=["High Engagement (Yes)", "Low Engagement (No)"],
                values=[yes_count, no_count],
                color_discrete_sequence=["#3AF28F", "#FF5C5C"],
                hole=0.45,
                title="Prediction Distribution",
            )
            st.plotly_chart(apply_chart_style(fig, 380), use_container_width=True)

            # ── Prediction table
            st.markdown('<div class="section-header">📋 Prediction Results</div>', unsafe_allow_html=True)
            result_df = raw_df.copy()
            result_df["Predicted_Engagement"] = predictions
            result_df["Confidence_Yes"] = probabilities[:, 1].round(4)
            result_df["Confidence_No"] = probabilities[:, 0].round(4)
            st.dataframe(result_df, use_container_width=True, height=400)

            # ── Download button
            csv_out = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Predictions as CSV",
                csv_out,
                "predictions.csv",
                "text/csv",
            )

            # ══════════════════════════════════════════════════════
            # Model evaluation (if actual target exists)
            # ══════════════════════════════════════════════════════
            if has_target:
                st.markdown('<div class="section-header">📏 Model Evaluation</div>', unsafe_allow_html=True)

                # Filter out NaN actuals
                mask = ~pd.isna(actual_engagement)
                eval_preds = predictions[mask]
                eval_actuals = actual_labels[mask]

                acc = accuracy_score(eval_actuals, eval_preds)
                st.markdown(metric_card("🎯", f"{acc*100:.2f}%", "Accuracy"), unsafe_allow_html=True)

                # ── Confusion Matrix
                st.markdown('<div class="section-header">🔢 Confusion Matrix</div>', unsafe_allow_html=True)
                cm = confusion_matrix(eval_actuals, eval_preds, labels=["No", "Yes"])
                fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=["Predicted No", "Predicted Yes"],
                    y=["Actual No", "Actual Yes"],
                    colorscale=[[0, "#13211B"], [1, "#3AF28F"]],
                    text=cm,
                    texttemplate="%{text}",
                    textfont=dict(size=18, color="white"),
                    showscale=False,
                ))
                fig.update_layout(title="Confusion Matrix")
                st.plotly_chart(apply_chart_style(fig, 380), use_container_width=True)

                # ── Classification Report
                st.markdown('<div class="section-header">📄 Classification Report</div>', unsafe_allow_html=True)
                report = classification_report(eval_actuals, eval_preds, output_dict=True)
                report_df = pd.DataFrame(report).T.round(4)
                st.dataframe(report_df, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.info("Ensure your CSV columns match the training data schema.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: FEATURE IMPORTANCE
# ═════════════════════════════════════════════════════════════════════════════

elif page == "🌳  Feature Importance":
    st.markdown("""
    <div class="gradient-header">
        <h1>🌳 Feature Importance</h1>
        <p>Understand which features the Decision Tree considers most influential</p>
    </div>
    """, unsafe_allow_html=True)

    if not MODEL_LOADED:
        st.error("Model not loaded.")
        st.stop()

    feature_names = list(model.feature_names_in_)
    importances = model.feature_importances_

    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=True)

    # ── Horizontal bar chart
    fig = px.bar(
        fi_df, x="Importance", y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Viridis",
        title="Decision Tree Feature Importance",
    )
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(apply_chart_style(fig, 650), use_container_width=True)

    # ── Top features highlight
    st.markdown('<div class="section-header">🏆 Top 5 Most Important Features</div>', unsafe_allow_html=True)
    top5 = fi_df.tail(5).iloc[::-1]
    cols = st.columns(5)
    medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
    for i, (_, row) in enumerate(top5.iterrows()):
        with cols[i]:
            st.markdown(
                metric_card(medals[i], f"{row['Importance']:.4f}", row["Feature"], show_trend=False),
                unsafe_allow_html=True,
            )

    # ── Feature importance table
    st.markdown('<div class="section-header">📋 All Features</div>', unsafe_allow_html=True)
    display_df = fi_df.iloc[::-1].reset_index(drop=True)
    display_df.index = display_df.index + 1
    display_df.index.name = "Rank"
    display_df["Importance (%)"] = (display_df["Importance"] * 100).round(2)
    st.dataframe(display_df, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD INSIGHTS
# ═════════════════════════════════════════════════════════════════════════════

elif page == "📈  Dashboard Insights":
    st.markdown("""
    <div class="gradient-header">
        <h1>📈 Dashboard Insights</h1>
        <p>Key performance indicators and aggregate statistics from the dataset</p>
    </div>
    """, unsafe_allow_html=True)

    df = default_df
    if df is None:
        st.warning("Default dataset not found. Please ensure the CSV is in the project folder.")
        st.stop()

    # ── Row 1 KPIs
    st.markdown('<div class="section-header">🏷️ Categorical Insights</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        val = df["platform"].mode()[0] if "platform" in df.columns else "—"
        st.markdown(metric_card("🌐", val, "Most Common Platform"), unsafe_allow_html=True)
    with c2:
        val = df["language"].mode()[0] if "language" in df.columns else "—"
        st.markdown(metric_card("🗣️", val, "Most Common Language"), unsafe_allow_html=True)
    with c3:
        val = df["sentiment_label"].mode()[0] if "sentiment_label" in df.columns else "—"
        st.markdown(metric_card("💬", val, "Most Common Sentiment"), unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ── Row 2 KPIs — Averages
    st.markdown('<div class="section-header">📊 Engagement Averages</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        val = f"{df['likes_count'].mean():,.0f}" if "likes_count" in df.columns else "—"
        st.markdown(metric_card("❤️", val, "Average Likes"), unsafe_allow_html=True)
    with c2:
        val = f"{df['shares_count'].mean():,.0f}" if "shares_count" in df.columns else "—"
        st.markdown(metric_card("🔁", val, "Average Shares"), unsafe_allow_html=True)
    with c3:
        val = f"{df['comments_count'].mean():,.0f}" if "comments_count" in df.columns else "—"
        st.markdown(metric_card("💬", val, "Average Comments"), unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ── Row 3 KPIs — Scores
    st.markdown('<div class="section-header">🔬 Score Averages</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        val = f"{df['sentiment_score'].mean():.3f}" if "sentiment_score" in df.columns else "—"
        st.markdown(metric_card("📈", val, "Avg Sentiment Score"), unsafe_allow_html=True)
    with c2:
        val = f"{df['toxicity_score'].mean():.3f}" if "toxicity_score" in df.columns else "—"
        st.markdown(metric_card("☠️", val, "Avg Toxicity Score"), unsafe_allow_html=True)
    with c3:
        val = f"{df['engagement_rate'].mean():.4f}" if "engagement_rate" in df.columns else "—"
        st.markdown(metric_card("🎯", val, "Avg Engagement Rate"), unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ── Additional metrics
    st.markdown('<div class="section-header">🔢 Additional Statistics</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        val = f"{df['impressions'].mean():,.0f}" if "impressions" in df.columns else "—"
        st.markdown(metric_card("👁️", val, "Avg Impressions"), unsafe_allow_html=True)
    with c2:
        val = f"{df['user_engagement_growth'].mean():.3f}" if "user_engagement_growth" in df.columns else "—"
        st.markdown(metric_card("📈", val, "Avg Engagement Growth"), unsafe_allow_html=True)
    with c3:
        val = f"{df['buzz_change_rate'].mean():.2f}" if "buzz_change_rate" in df.columns else "—"
        st.markdown(metric_card("🔥", val, "Avg Buzz Change"), unsafe_allow_html=True)
    with c4:
        val = f"{df['user_past_sentiment_avg'].mean():.3f}" if "user_past_sentiment_avg" in df.columns else "—"
        st.markdown(metric_card("📊", val, "Avg Past Sentiment"), unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: VISUAL ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════

elif page == "🎨  Visual Analytics":
    st.markdown("""
    <div class="gradient-header">
        <h1>🎨 Visual Analytics</h1>
        <p>Rich visualisations across every dimension of the dataset</p>
    </div>
    """, unsafe_allow_html=True)

    df = default_df
    if df is None:
        st.warning("Default dataset not found.")
        st.stop()

    def safe_hist(column, title, color="#818cf8"):
        """Render a histogram if the column exists."""
        if column in df.columns:
            fig = px.histogram(
                df, x=column, nbins=40,
                color_discrete_sequence=[color],
                title=title,
                marginal="violin",
            )
            st.plotly_chart(apply_chart_style(fig, 400), use_container_width=True)

    def safe_pie(column, title, top_n=10):
        """Render a pie/donut chart for a categorical column."""
        if column in df.columns:
            counts = df[column].value_counts().head(top_n)
            fig = px.pie(
                names=counts.index, values=counts.values,
                color_discrete_sequence=COLOR_PALETTE,
                title=title, hole=0.4,
            )
            st.plotly_chart(apply_chart_style(fig, 420), use_container_width=True)

    # ── Engagement Metrics
    st.markdown('<div class="section-header">💎 Engagement Metrics</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        safe_hist("likes_count", "❤️ Likes Distribution", "#f472b6")
    with c2:
        safe_hist("comments_count", "💬 Comments Distribution", "#818cf8")
    with c3:
        safe_hist("shares_count", "🔁 Shares Distribution", "#34d399")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ── Platform & Language
    st.markdown('<div class="section-header">🌐 Platform & Language</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        safe_pie("platform", "🌐 Platform Distribution")
    with c2:
        safe_pie("language", "🗣️ Language Distribution")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ── Sentiment & Emotion
    st.markdown('<div class="section-header">💭 Sentiment & Emotion</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        safe_pie("sentiment_label", "😊 Sentiment Distribution")
    with c2:
        safe_pie("emotion_type", "🎭 Emotion Distribution")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ── Topic & Campaign
    st.markdown('<div class="section-header">📂 Topic & Campaign</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        safe_pie("topic_category", "📂 Topic Distribution", 12)
    with c2:
        safe_pie("campaign_name", "🎯 Campaign Distribution", 10)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ── Engagement Rate
    st.markdown('<div class="section-header">📈 Engagement Rate</div>', unsafe_allow_html=True)
    if "engagement_rate" in df.columns:
        fig = px.histogram(
            df, x="engagement_rate", nbins=50,
            color_discrete_sequence=["#a78bfa"],
            title="📈 Engagement Rate Distribution",
            marginal="box",
        )
        st.plotly_chart(apply_chart_style(fig), use_container_width=True)

        # Engagement by platform
        if "platform" in df.columns:
            fig = px.box(
                df, x="platform", y="engagement_rate",
                color="platform",
                color_discrete_sequence=COLOR_PALETTE,
                title="Engagement Rate by Platform",
            )
            st.plotly_chart(apply_chart_style(fig), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ═════════════════════════════════════════════════════════════════════════════

elif page == "ℹ️  About":
    st.markdown("""
    <div class="gradient-header">
        <h1>ℹ️ About This Project</h1>
        <p>Social Media Engagement Prediction — Decision Tree Machine Learning Project</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Project Description
    st.markdown("""
    <div class="glass-card">
        <h3>📖 Project Description</h3>
        <p style="color:#c9d1d9; line-height:1.8; font-size:0.95rem;">
        This project applies <strong style="color:#818cf8;">Machine Learning</strong> to predict
        social media post engagement levels. Using a comprehensive dataset of 7,000+ social media
        posts across multiple platforms, the <strong>Decision Tree Classifier</strong> learns
        patterns in post metadata, sentiment, toxicity, user history and campaign details to
        classify whether a post will achieve <em>high</em> or <em>low</em> engagement.<br><br>
        The interactive dashboard provides full data exploration, visual analytics,
        real-time predictions, feature importance analysis and key performance
        indicators — all wrapped in a modern, dark-mode interface.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Technologies Used
    st.markdown('<div class="section-header">🛠️ Technologies Used</div>', unsafe_allow_html=True)

    techs = [
        ("🐍", "Python", "Core programming language"),
        ("🌐", "Streamlit", "Web application framework"),
        ("🐼", "Pandas", "Data manipulation & analysis"),
        ("🔢", "NumPy", "Numerical computing"),
        ("🤖", "Scikit-learn", "Machine Learning library"),
        ("🌳", "Decision Tree", "Classification algorithm"),
        ("📊", "Plotly", "Interactive visualisations"),
        ("📉", "Matplotlib", "Static plotting"),
        ("🎨", "Seaborn", "Statistical visualisations"),
        ("💾", "Joblib", "Model serialisation"),
    ]

    cols = st.columns(5)
    for i, (icon, name, desc) in enumerate(techs):
        with cols[i % 5]:
            st.markdown(f"""
            <div class="about-card" style="margin-bottom:15px;">
                <div class="about-icon">{icon}</div>
                <h4>{name}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ── Developer Information
    st.markdown('<div class="section-header">👨‍💻 Developer Information</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card" style="text-align:center;">
        <div style="font-size:4rem; margin-bottom:10px;">👨‍💻</div>
        <h3 style="margin:5px 0;">Developer Name</h3>
        <p style="color:#8b8fa3; margin:5px 0;">Final Year Machine Learning Project</p>
        <div style="margin-top:18px;">
            <a href="https://github.com/" target="_blank" style="text-decoration:none;">
                <span class="tech-chip">🔗 GitHub</span>
            </a>
            <a href="https://linkedin.com/" target="_blank" style="text-decoration:none;">
                <span class="tech-chip">💼 LinkedIn</span>
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ── Project Info
    st.markdown("""
    <div class="info-box">
        <strong>📌 Project Details</strong><br>
        <strong>Algorithm:</strong> Decision Tree Classifier<br>
        <strong>Target Variable:</strong> engagement_rate (binarised: ≥ 0.1 → High, &lt; 0.1 → Low)<br>
        <strong>Dataset Size:</strong> 7,315 rows × 28 columns<br>
        <strong>Features Used:</strong> 26 (after dropping <code>mentions</code> and target)<br>
        <strong>Classes:</strong> Yes (High Engagement) / No (Low Engagement)
    </div>
    """, unsafe_allow_html=True)