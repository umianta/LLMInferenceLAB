import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import streamlit as st
import torch
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*scaled_dot_product_attention.*")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from transformers import (
    DistilBertTokenizer,
    DistilBertModel,
    GPT2Tokenizer,
    GPT2LMHeadModel
)
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
import time

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="LLM Inference Learning Lab",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- MODERN STYLING -----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700;800&family=Outfit:wght@300;400;500;600;700;800&family=Hubot+Sans:wght@400;500;600;700;800&family=DM+Sans:wght@400;500;600;700&display=swap');
    
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    html, body, [class*="css"] {
        font-family: 'Sora', 'Outfit', 'DM Sans', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #1a1f3a 0%, #1e293b 50%, #1d2a3a 100%);
        padding: 2rem;
        min-height: 100vh;
    }
    
    @keyframes backgroundPulse {
        0%, 100% { 
            background: linear-gradient(135deg, #1a1f3a 0%, #1e293b 50%, #1d2a3a 100%);
        }
        50% { 
            background: linear-gradient(135deg, #1e293b 0%, #2d3548 50%, #1f2937 100%);
        }
    }
    
    .block-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #eef2f7 100%);
        border-radius: 24px;
        padding: 2rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.5);
        max-width: 1400px;
        backdrop-filter: blur(10px);
        animation: slideInContainer 0.8s ease-out;
        border: 1px solid rgba(103, 126, 234, 0.2);
    }
    
    @keyframes slideInContainer {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    h1 {
        color: #f8fafc;
        font-weight: 900;
        font-size: 3rem !important;
        margin-bottom: 0.5rem !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.03em;
        font-family: 'Sora', 'Outfit', 'Hubot Sans', sans-serif;
        animation: gradientFlow 6s ease infinite;
        text-transform: capitalize;
    }
    
    @keyframes gradientFlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    h2 {
        color: #f1f5f9;
        font-weight: 800;
        font-size: 1.85rem !important;
        margin-top: 2rem !important;
        margin-bottom: 1.25rem !important;
        padding-bottom: 0.75rem;
        border-bottom: 4px solid;
        border-image: linear-gradient(90deg, #667eea, #764ba2) 1;
        display: inline-block;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-family: 'Outfit', 'Sora', 'Hubot Sans', sans-serif;
        animation: colorShift 5s ease infinite;
        letter-spacing: -0.01em;
    }
    
    @keyframes colorShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    h3 {
        color: #1a1a2e;
        font-weight: 700;
        font-size: 1.4rem !important;
        margin-bottom: 0.5rem !important;
        font-family: 'Outfit', 'Sora', 'Hubot Sans', sans-serif;
        transition: all 0.3s ease;
        letter-spacing: -0.005em;
    }
    
    h3:hover {
        color: #667eea;
        transform: translateX(6px);
    }
    
    h4 {
        color: #2d3748;
        font-weight: 700;
        font-size: 1.1rem !important;
        font-family: 'Outfit', 'Sora', sans-serif;
        transition: color 0.3s ease;
        letter-spacing: -0.005em;
    }
    
    h4:hover {
        color: #667eea;
    }
    
    p {
        line-height: 1.8;
        color: #2d3748;
        font-family: 'DM Sans', 'Sora', sans-serif;
        font-size: 0.95rem;
        font-weight: 400;
        letter-spacing: 0.3px;
    }
    
    strong, b {
        color: #1a1a2e;
        font-weight: 700;
    }
    
    em, i {
        color: #4a5568;
        font-style: italic;
    }
    
    code {
        background: #2d3548;
        color: #a5b4fc;
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    a {
        color: #667eea;
        text-decoration: none;
        font-weight: 600;
        transition: all 0.3s ease;
        border-bottom: 2px solid transparent;
    }
    
    a:hover {
        color: #a5b4fc;
        border-bottom-color: #667eea;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-size: 200% 200%;
        color: white;
        border: none;
        border-radius: 14px;
        padding: 0.85rem 2.2rem;
        font-weight: 700;
        font-size: 1rem;
        font-family: 'Sora', 'Outfit', sans-serif;
        transition: all 0.4s cubic-bezier(0.23, 1, 0.320, 1);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        letter-spacing: 0.02em;
        text-transform: uppercase;
        position: relative;
        overflow: hidden;
    }
    
    .stButton button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .stButton button:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #f093fb 100%);
        animation: buttonPulse 0.6s ease;
    }
    
    @keyframes buttonPulse {
        0%, 100% { box-shadow: 0 12px 35px rgba(102, 126, 234, 0.6); }
        50% { box-shadow: 0 15px 45px rgba(118, 75, 162, 0.8); }
    }
    
    .stButton button:active {
        transform: translateY(-1px);
    }
    
    .stTextArea textarea {
        border-radius: 14px;
        border: 2px solid #e2e8f0;
        padding: 1.25rem;
        font-size: 1rem;
        font-family: 'DM Sans', 'Inter', monospace;
        transition: all 0.3s cubic-bezier(0.23, 1, 0.320, 1);
        background: #2d3548;
        color: #e2e8f0;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        background: #1e2332;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.15), 0 0 20px rgba(102, 126, 234, 0.3);
        outline: none;
        animation: textareaPulse 1.5s ease infinite;
    }
    
    @keyframes textareaPulse {
        0%, 100% { box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.15), 0 0 20px rgba(102, 126, 234, 0.3); }
        50% { box-shadow: 0 0 0 8px rgba(102, 126, 234, 0.25), 0 0 30px rgba(102, 126, 234, 0.5); }
    }
    
    .stTextArea textarea::placeholder {
        color: #94a3b8;
        font-style: italic;
    }
    
    .stSlider {
        padding: 1.5rem 0;
    }
    
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 0.95rem;
        font-weight: 600;
        color: #2d3748;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    
    .stAlert {
        border-radius: 14px;
        border: none;
        padding: 1.5rem;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.08) 0%, rgba(118, 75, 162, 0.08) 100%);
        border-left: 5px solid #667eea;
        backdrop-filter: blur(10px);
        animation: alertSlideIn 0.5s ease-out;
        font-family: 'Sora', 'Outfit', sans-serif;
    }
    
    @keyframes alertSlideIn {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .stAlert:hover {
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.15);
    }
    
    .stSuccess {
        background: linear-gradient(135deg, rgba(72, 187, 120, 0.08) 0%, rgba(56, 161, 105, 0.08) 100%) !important;
        border-left-color: #48bb78 !important;
        animation: successGlow 2s ease infinite;
    }
    
    @keyframes successGlow {
        0%, 100% { border-left-color: #48bb78; }
        50% { border-left-color: #38a169; }
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.08) 0%, rgba(217, 119, 6, 0.08) 100%) !important;
        border-left-color: #f59e0b !important;
        animation: warningGlow 2s ease infinite;
    }
    
    @keyframes warningGlow {
        0%, 100% { border-left-color: #f59e0b; }
        50% { border-left-color: #d97706; }
    }
    
    .stDataFrame {
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
        animation: dataFrameFade 0.6s ease-out;
        font-family: 'DM Sans', 'Sora', sans-serif;
    }
    
    @keyframes dataFrameFade {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .stDataFrame:hover {
        border-color: #667eea;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15);
    }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f3a 0%, #0f172a 100%) !important;
        animation: sidebarSlide 0.6s ease-out;
    }
    
    @keyframes sidebarSlide {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    section[data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #1a1f3a 0%, #0f172a 100%) !important;
        padding: 2rem 1.25rem;
        border-right: 3px solid #667eea;
        transition: all 0.3s ease;
    }
    
    section[data-testid="stSidebar"] label {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        font-family: 'Sora', 'Outfit', sans-serif;
        transition: all 0.3s ease;
    }
    
    section[data-testid="stSidebar"] label:hover {
        color: #667eea !important;
        transform: translateX(4px);
    }
    
    section[data-testid="stSidebar"] h2 {
        color: #667eea !important;
        border-bottom-color: #667eea !important;
        animation: sidebarTitle 0.6s ease-out;
    }
    
    @keyframes sidebarTitle {
        from {
            opacity: 0;
            transform: translateY(-10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #cbd5e0;
        font-family: 'DM Sans', 'Sora', sans-serif;
    }

    /* Make buttons in the sidebar visible and prominent */
    section[data-testid="stSidebar"] div.stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.65rem 0.9rem !important;
        width: 100% !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.35) !important;
        text-align: center !important;
        font-weight: 700 !important;
        font-family: 'Sora', 'Outfit', sans-serif !important;
        text-transform: uppercase !important;
    }

    section[data-testid="stSidebar"] div.stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.55) !important;
    }

    /* Broad fallback: make any buttons inside the sidebar visible */
    section[data-testid="stSidebar"] button,
    section[data-testid="stSidebar"] .stButton button,
    section[data-testid="stSidebar"] .stButton {
        display: block !important;
        opacity: 1 !important;
        visibility: visible !important;
        color: white !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        z-index: 999 !important;
    }

    /* Ensure icon/text inside the button is visible */
    section[data-testid="stSidebar"] button * {
        color: inherit !important;
        fill: white !important;
    }
    
    section[data-testid="stSidebar"] hr {
        border-color: #2d3548;
        margin: 1.5rem 0;
        animation: hrFade 0.6s ease-out;
    }
    
    @keyframes hrFade {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .stCheckbox {
        padding: 0.75rem 0;
    }
    
    .stCheckbox label {
        font-size: 0.95rem;
        cursor: pointer;
    }
    
    .stRadio {
        padding: 0.75rem 0;
    }
    
    .stRadio label {
        font-size: 0.95rem;
        cursor: pointer;
        font-weight: 600;
        color: #e2e8f0 !important;
    }

    /* Ensure sidebar option labels (radio/checkbox/select) are bright and visible */
    section[data-testid="stSidebar"] .stRadio,
    section[data-testid="stSidebar"] .stCheckbox,
    section[data-testid="stSidebar"] .stMultiSelect,
    section[data-testid="stSidebar"] .stSelectbox,
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] .stCheckbox label,
    section[data-testid="stSidebar"] .stMultiSelect label,
    section[data-testid="stSidebar"] .stSelectbox label {
        color: #ffffff !important;
        opacity: 1 !important;
        font-weight: 600 !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.6) !important;
    }

    /* Make small helper text slightly dimmer but readable */
    section[data-testid="stSidebar"] small,
    section[data-testid="stSidebar"] .help-text,
    section[data-testid="stSidebar"] .stMarkdown small {
        color: #cbd5e0 !important;
        opacity: 0.95 !important;
    }

    /* Ensure icons inside options use visible fill */
    section[data-testid="stSidebar"] .stRadio label svg,
    section[data-testid="stSidebar"] .stCheckbox label svg {
        fill: #e2e8f0 !important;
        color: #e2e8f0 !important;
        opacity: 1 !important;
    }

    /* Broadly target any text-bearing element inside sidebar controls */
    section[data-testid="stSidebar"] .stRadio *,
    section[data-testid="stSidebar"] .stCheckbox *,
    section[data-testid="stSidebar"] .stMultiSelect *,
    section[data-testid="stSidebar"] .stSelectbox * {
        color: #ffffff !important;
        opacity: 1 !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.45) !important;
    }
    
    .stMultiSelect {
        margin: 1.5rem 0;
    }
    
    div.stDownloadButton > button {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        background-size: 200% 200%;
        box-shadow: 0 8px 25px rgba(72, 187, 120, 0.4);
        border-radius: 14px;
        padding: 0.85rem 2rem !important;
        font-weight: 700 !important;
        font-family: 'Sora', 'Outfit', sans-serif;
        text-transform: uppercase !important;
        letter-spacing: 0.02em !important;
        transition: all 0.4s ease;
        animation: downloadButtonEntry 0.6s ease-out;
    }
    
    @keyframes downloadButtonEntry {
        from {
            opacity: 0;
            transform: scale(0.9);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    div.stDownloadButton > button:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 35px rgba(72, 187, 120, 0.6);
        animation: downloadButtonHover 0.4s ease;
    }
    
    @keyframes downloadButtonHover {
        0% { box-shadow: 0 12px 35px rgba(72, 187, 120, 0.6); }
        50% { box-shadow: 0 15px 45px rgba(72, 187, 120, 0.8); }
        100% { box-shadow: 0 12px 35px rgba(72, 187, 120, 0.6); }
    }
    
    .element-container {
        margin-bottom: 1.5rem;
        animation: elementFade 0.5s ease-out;
    }
    
    @keyframes elementFade {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Tab styling */
    div[data-baseweb="tab-list"] {
        gap: 12px;
        background: transparent;
        border-bottom: 3px solid #e2e8f0;
        padding-bottom: 0.5rem;
    }
    
    div[data-baseweb="tab"] {
        border-radius: 12px;
        background-color: #f0f4f8;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-family: 'Sora', 'Outfit', sans-serif;
        border: 2px solid transparent;
        transition: all 0.3s cubic-bezier(0.23, 1, 0.320, 1);
        cursor: pointer;
        animation: tabFade 0.5s ease-out;
    }
    
    @keyframes tabFade {
        from {
            opacity: 0;
            transform: translateY(-10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    div[data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, #e8f4f8 0%, #e0f2fe 100%);
        border-color: #cbd5e0;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.1);
    }
    
    div[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-size: 200% 200%;
        color: white !important;
        border-color: #667eea;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
        animation: tabActive 0.4s ease;
    }
    
    @keyframes tabActive {
        0% {
            transform: scale(0.95);
            box-shadow: 0 2px 10px rgba(102, 126, 234, 0.2);
        }
        100% {
            transform: scale(1);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
        }
    }
    
    /* Professional Cards */
    .card {
        background: white;
        color: #2d3748;
        border-radius: 16px;
        padding: 1.75rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        transition: all 0.4s cubic-bezier(0.23, 1, 0.320, 1);
        animation: cardFadeIn 0.6s ease-out;
    }

    /* Ensure readable text on inline white / white-gradient backgrounds
       (target common inline-style usage in the app). Exclude <code>
       so inline code snippets retain their explicit colors. */
    [style*="background: white"] p,
    [style*="background: white"] span,
    [style*="background: white"] h1,
    [style*="background: white"] h2,
    [style*="background: white"] h3,
    [style*="background: white"] h4,
    [style*="background: white"] li,
    [style*="background: white"] a,
    [style*="background: white"] em,
    [style*="background: white"] i {
        color: #2d3748 !important;
    }

    /* Also catch linear gradients that begin with white (e.g. modal overlay) */
    [style*="linear-gradient(135deg, white"] p,
    [style*="linear-gradient(135deg, white"] span,
    [style*="linear-gradient(135deg, white"] h1,
    [style*="linear-gradient(135deg, white"] h2,
    [style*="linear-gradient(135deg, white"] h3,
    [style*="linear-gradient(135deg, white"] h4,
    [style*="linear-gradient(135deg, white"] li {
        color: #2d3748 !important;
    }
    
    @keyframes cardFadeIn {
        from {
            opacity: 0;
            transform: translateY(15px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .card:hover {
        transform: translateY(-6px);
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.15);
        border: 2px solid #667eea;
        animation: cardHover 0.4s ease;
    }
    
    @keyframes cardHover {
        0% { border-color: #e2e8f0; }
        100% { border-color: #667eea; }
    }
    
    .gradient-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-size: 200% 200%;
        color: white;
        border: none;
        animation: gradientCard 6s ease infinite;
    }
    
    @keyframes gradientCard {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .gradient-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
    }
    
    .concept-card {
        background: linear-gradient(135deg, #e8f4f8 0%, #e0f2fe 100%);
        border: 2px solid #667eea;
        border-left: 5px solid #667eea;
        animation: conceptCard 0.6s ease-out;
    }
    
    @keyframes conceptCard {
        from {
            opacity: 0;
            transform: translateX(-10px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .concept-card:hover {
        border-left-color: #764ba2;
        border-color: #764ba2;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.2);
    }
    
    .learning-tip {
        background: linear-gradient(135deg, #fef5e7 0%, #fef9e7 100%);
        border-left: 5px solid #f39c12;
        border-radius: 12px;
        animation: tipSlideIn 0.6s ease-out;
    }
    
    @keyframes tipSlideIn {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .learning-tip:hover {
        border-left-color: #e8860b;
        box-shadow: 0 6px 20px rgba(243, 156, 18, 0.2);
    }

    /* Animation utilities */
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    @keyframes glow {
        0%, 100% {
            box-shadow: 0 0 20px rgba(102, 126, 234, 0.4);
        }
        50% {
            box-shadow: 0 0 30px rgba(102, 126, 234, 0.7);
        }
    }

    .animated-gradient {
        background: linear-gradient(270deg, #667eea, #764ba2, #f093fb, #48bb78);
        background-size: 400% 400%;
        animation: gradientShift 8s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .pulse-card {
        animation: pulse 1.6s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        box-shadow: 0 8px 30px rgba(118,75,162,0.15) !important;
        border-color: #764ba2 !important;
    }

    @keyframes pulse {
        0%, 100% { 
            transform: translateY(0);
            box-shadow: 0 8px 30px rgba(118,75,162,0.15);
        }
        50% { 
            transform: translateY(-6px);
            box-shadow: 0 12px 40px rgba(118,75,162,0.25);
        }
    }

    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .fade-in {
        animation: fadeInUp 0.6s ease;
    }

    .glow-effect {
        animation: glow 2s ease-in-out infinite;
    }

    /* Modal overlay */
    .overlay-backdrop {
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(10,11,13,0.55);
        backdrop-filter: blur(8px);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 9999;
    }

    .overlay-card {
        background: linear-gradient(135deg, white 0%, #f7fafc 100%);
        padding: 2.5rem;
        border-radius: 20px;
        max-width: 800px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        transform: translateY(-10px);
        animation: floatIn 0.6s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
        border: 1px solid rgba(255,255,255,0.8);
    }

    @keyframes floatIn {
        from { 
            opacity: 0; 
            transform: translateY(20px) scale(0.95); 
        }
        to { 
            opacity: 1; 
            transform: translateY(0) scale(1); 
        }
    }

    .intrusive-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 50px;
        background: linear-gradient(90deg, #ff7a7a, #ffb27a);
        color: white;
        font-weight: 800;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        box-shadow: 0 8px 20px rgba(255,122,122,0.35);
        animation: badgePulse 1.8s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }

    @keyframes badgePulse {
        0%, 100% { 
            transform: scale(1);
            box-shadow: 0 8px 20px rgba(255,122,122,0.35);
        }
        50% { 
            transform: scale(1.08);
            box-shadow: 0 12px 30px rgba(255,122,122,0.5);
        }
    }

    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
    }

    /* Metric container styling */
    .metric-card {
        background: linear-gradient(135deg, #f0f4f8 0%, #f7fafc 100%);
        padding: 1.5rem;
        border-radius: 16px;
        border: 2px solid #e2e8f0;
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        border-color: #667eea;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.15);
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f0f4f8 0%, #f7fafc 100%) !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        font-family: 'Sora', 'Outfit', sans-serif;
        transition: all 0.3s ease;
        animation: expanderEntry 0.5s ease-out;
    }
    
    @keyframes expanderEntry {
        from {
            opacity: 0;
            transform: translateX(-10px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, #e2e8f0 0%, #edf2f7 100%) !important;
        border-color: #667eea !important;
        transform: translateX(4px);
    }
    
    /* Metric styling */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        border-radius: 14px;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
        animation: metricBounce 0.6s ease-out;
    }
    
    @keyframes metricBounce {
        from {
            opacity: 0;
            transform: scale(0.95) translateY(10px);
        }
        to {
            opacity: 1;
            transform: scale(1) translateY(0);
        }
    }
    
    /* Additional color animations for text */
    .text-gradient {
        background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: textGradientShift 6s ease infinite;
    }
    
    @keyframes textGradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Focus animations */
    input:focus, select:focus, textarea:focus {
        animation: focusGlow 0.3s ease-out !important;
    }
    
    @keyframes focusGlow {
        from { box-shadow: none; }
        to { box-shadow: 0 0 20px rgba(102, 126, 234, 0.3); }
    }
    
    /* Smooth transitions for all interactive elements */
    * {
        transition: all 0.2s ease;
    }
    
    button, [role="button"], input, select, textarea {
        transition: all 0.3s cubic-bezier(0.23, 1, 0.320, 1);
    }
</style>
""", unsafe_allow_html=True)

# ----------------- HEADER -----------------
st.markdown("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            border-radius: 24px; padding: 3rem; margin-bottom: 2.5rem; box-shadow: 0 15px 50px rgba(102,126,234,0.25);'>
    <div style='text-align: center;'>
        <h1 style='margin: 0; color: blue; font-size: 3.5rem; font-weight: 800; letter-spacing: -0.02em; text-shadow: 0 2px 10px rgba(0,0,0,0.1);'>
            🎓LLM Inference Learning Lab
        </h1>
        <p style='font-size: 1.4rem; color: rgba(255,255,255,0.95); margin: 1rem 0 0 0; font-weight: 600;'>
            Master Large Language Models through Interactive Exploration
        </p>
        <p style='font-size: 1.05rem; color: rgba(255,255,255,0.85); margin-top: 1rem; line-height: 1.7; max-width: 800px; margin-left: auto; margin-right: auto;'>
            ✨ Explore tokenization, embeddings, attention mechanisms, and predictions in real-time  
            <br>🚀 Learn by doing with hands-on experiments  
            <br>💡 Understand how transformers really work from the inside out
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- SAFE DEFAULTS -----------------
top_tokens = []
top_probs = np.array([0.0])

# ----------------- LOAD MODELS -----------------
@st.cache_resource
def load_models():
    bert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased", output_attentions=True)
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    return bert_tokenizer, bert_model, gpt2_tokenizer, gpt2_model

with st.spinner("🚀 Loading models..."):
    bert_tokenizer, bert_model, gpt2_tokenizer, gpt2_model = load_models()

# ----------------- PROMPT INPUT -----------------
st.markdown("### 💬 Enter Your Prompt")
# Prompt input: single-line submits on Enter via form, plus optional multi-line box
default_prompt = "Explain why ice floats on water."
prompt = default_prompt

with st.form(key='prompt_form'):
    prompt_input = st.text_input("Enter prompt (press Enter to submit)", value=default_prompt, placeholder="Type prompt here...", key='prompt_input')
    submit_btn = st.form_submit_button("Submit")

# Multi-line option for longer prompts (submit with button)
with st.expander("Need multi-line prompt?", expanded=False):
    long_prompt = st.text_area("Multi-line prompt", value=default_prompt, height=150, placeholder="Write a longer prompt here...")
    long_submit = st.button("Submit Multi-line Prompt")

if submit_btn:
    prompt = prompt_input
elif long_submit:
    prompt = long_prompt
else:
    # preserve existing prompt if user hasn't interacted
    prompt = st.session_state.get('prompt', default_prompt)

# store current prompt in session state
    st.session_state['prompt'] = prompt

# Welcome/Quick Start Card
st.markdown("""
<div style='background: linear-gradient(135deg, #f0f4f8 0%, #e0f2fe 100%);
            border-left: 6px solid #667eea;
            border-radius: 18px; 
            padding: 2.5rem; 
            margin: 2.5rem 0;
            box-shadow: 0 8px 25px rgba(102,126,234,0.15);
            border: 2px solid rgba(102,126,234,0.2);
            backdrop-filter: blur(10px);'>
    <h3 style='margin-top: 0; font-size: 1.6rem; color: #667eea; font-weight: 800; display: flex; align-items: center; gap: 0.75rem;'>
        🚀 Quick Start Guide
    </h3>
    <div style='display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1.5rem; margin-top: 1.5rem;'>
        <div style='background: white; padding: 1.5rem; border-radius: 14px; border-left: 4px solid #667eea; box-shadow: 0 2px 8px rgba(0,0,0,0.05);'>
            <p style='font-weight: 700; font-size: 1.1rem; color: #667eea; margin: 0 0 0.5rem 0;'>1️⃣ Enter Prompt</p>
            <p style='color: #2d3748; font-size: 0.95rem; margin: 0; line-height: 1.6;'>Type your text below and watch the model analyze it in real-time.</p>
        </div>
        <div style='background: white; padding: 1.5rem; border-radius: 14px; border-left: 4px solid #764ba2; box-shadow: 0 2px 8px rgba(0,0,0,0.05);'>
            <p style='font-weight: 700; font-size: 1.1rem; color: #764ba2; margin: 0 0 0.5rem 0;'>2️⃣ Explore</p>
            <p style='color: #2d3748; font-size: 0.95rem; margin: 0; line-height: 1.6;'>Discover tokenization, embeddings, and attention patterns.</p>
        </div>
        <div style='background: white; padding: 1.5rem; border-radius: 14px; border-left: 4px solid #48bb78; box-shadow: 0 2px 8px rgba(0,0,0,0.05);'>
            <p style='font-weight: 700; font-size: 1.1rem; color: #48bb78; margin: 0 0 0.5rem 0;'>3️⃣ Experiment</p>
            <p style='color: #2d3748; font-size: 0.95rem; margin: 0; line-height: 1.6;'>Change prompts and see predictions update instantly.</p>
        </div>
    </div>
    <p style='color: #667eea; font-size: 0.95rem; margin-top: 1.5rem; margin-bottom: 0; font-weight: 600;'>
        💡 <em style='color: #4a5568;'>Pro Tip: Select your learning level in the sidebar to customize which concepts are shown.</em>
    </p>
</div>
""", unsafe_allow_html=True)

inputs = bert_tokenizer(prompt, return_tensors="pt")
tokens = bert_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
with torch.no_grad():
    bert_outputs = bert_model(**inputs)
    embeddings = bert_outputs.last_hidden_state[0]
    attentions = bert_outputs.attentions

# Learning path & preferences
st.sidebar.markdown("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 16px;
            margin-bottom: 1.5rem;
            box-shadow: 0 8px 20px rgba(102,126,234,0.25);'>
    <h3 style='margin-top: 0; font-size: 1.3rem; font-weight: 800; display: flex; align-items: center; gap: 0.5rem;'>
        📚 Your Learning Path
    </h3>
    <p style='color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 0.9rem; line-height: 1.6;'>
        Select your expertise level to customize what concepts are shown.
    </p>
</div>
""", unsafe_allow_html=True)

learning_path = st.sidebar.radio(
    "Select your level:",
    ("🌱 Beginner", "🌿 Intermediate", "🌳 Advanced"),
    help="Beginner: Focus on basics. Intermediate: Explore internals. Advanced: Deep dive into all features."
)

show_tips = st.sidebar.checkbox("💡 Show Learning Tips", True, help="Enable inline explanations and best practices.")
show_equations = st.sidebar.checkbox("📐 Show Mathematics", False, help="Display mathematical formulas and equations.")

# Intrusive / Animated Mode toggle
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='background: linear-gradient(135deg, rgba(255,122,122,0.1) 0%, rgba(255,178,122,0.1) 100%);
            border: 2px solid #ff7a7a;
            border-radius: 12px;
            padding: 1rem;
            margin: 1rem 0;'>
    <p style='color: #d63031; font-weight: 700; margin: 0 0 0.5rem 0; font-size: 0.95rem;'>✨ Enhanced Experience</p>
    <p style='color: #7d4949; font-size: 0.85rem; margin: 0;'>Enable animations and visual highlights on key insights.</p>
</div>
""", unsafe_allow_html=True)

intrusive_mode = st.sidebar.checkbox("🔔 Intrusive Mode (animated)", False)

# Learning sections based on level
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Learning Modules")

# Define sections based on learning path
if learning_path == "🌱 Beginner":
    sections = {
        "Prompt & Models": True,
        "Tokenization": True,
        "Embeddings": True,
        "Attention": True,
        "Attention Comparison": False,
        "Token Similarity": False,
        "Next Token": True,
        "Text Generation": False,
        "Debug Summary": True
    }
    st.sidebar.info("📖 Beginner: Learn the fundamentals of how LLMs process text.")
elif learning_path == "🌿 Intermediate":
    sections = {
        "Prompt & Models": True,
        "Tokenization": True,
        "Embeddings": True,
        "Attention": True,
        "Attention Comparison": True,
        "Token Similarity": True,
        "Next Token": True,
        "Text Generation": True,
        "Debug Summary": True
    }
    st.sidebar.info("🔬 Intermediate: Explore model internals and attention mechanisms.")
else:  # Advanced
    sections = {
        "Prompt & Models": True,
        "Tokenization": True,
        "Embeddings": True,
        "Attention": True,
        "Attention Comparison": True,
        "Token Similarity": True,
        "Next Token": True,
        "Text Generation": True,
        "Debug Summary": True
    }
    st.sidebar.info("🧠 Advanced: Full access to all features and deep diagnostics.")

# Custom section selection for power users
st.sidebar.markdown("---")
if st.sidebar.checkbox("🎯 Customize Sections", False):
    sections = {
        "Prompt & Models": st.sidebar.checkbox("📝 Prompt & Models", sections["Prompt & Models"]),
        "Tokenization": st.sidebar.checkbox("🔤 Tokenization", sections["Tokenization"]),
        "Embeddings": st.sidebar.checkbox("📊 Embeddings Visualization", sections["Embeddings"]),
        "Attention": st.sidebar.checkbox("🧠 Attention Analysis", sections["Attention"]),
        "Attention Comparison": st.sidebar.checkbox("🔄 Attention Comparison", sections["Attention Comparison"]),
        "Token Similarity": st.sidebar.checkbox("💡 Token Similarity", sections["Token Similarity"]),
        "Next Token": st.sidebar.checkbox("➡️ Next Token Prediction", sections["Next Token"]),
        "Text Generation": st.sidebar.checkbox("✍️ Text Generation", sections["Text Generation"]),
        "Debug Summary": st.sidebar.checkbox("📋 Debug Summary", sections["Debug Summary"])
    }

st.sidebar.markdown("---")
st.sidebar.markdown("""
<p style='color: #4a5568; font-size: 0.85rem; text-align: center;'>
    🎓 LLM Learning Lab v1.0<br>
    Built with Streamlit & Transformers
</p>
""", unsafe_allow_html=True)

# session state for modal
if "show_intrusive_modal" not in st.session_state:
    st.session_state.show_intrusive_modal = False

# When intrusive mode is turned ON, show the modal
if intrusive_mode and not st.session_state.show_intrusive_modal:
    st.session_state.show_intrusive_modal = True

# When intrusive mode is turned OFF, hide the modal
if not intrusive_mode:
    st.session_state.show_intrusive_modal = False

# Display modal only if intrusive mode is ON and modal hasn't been dismissed
if st.session_state.show_intrusive_modal:
    st.warning("⚠️ **Intrusive Mode Enabled** - Stronger visual cues and animations active", icon="🎯")
    
    with st.expander("📋 Intrusive Mode Details", expanded=True):
        st.markdown("""
        This mode surfaces stronger visual cues and animations to make debugging more immediate and attention-grabbing.
        
        **Features:**
        - ✨ Animated highlights on key predictions
        - 💓 Pulse effects on important metrics
        - 💡 Overlay hints for quick tips
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✨ Got it, understood!", use_container_width=True, key="modal_got_it"):
                st.session_state.show_intrusive_modal = False
                st.rerun()
        with col2:
            if st.button("Close", use_container_width=True, key="modal_dismiss"):
                st.session_state.show_intrusive_modal = False
                st.rerun()

# Learning helper functions
def learning_tip(title, content):
    """Display a learning tip card if show_tips is enabled."""
    if show_tips:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #fef5e7 0%, #fef9e7 100%);
                    border-left: 5px solid #f39c12;
                    border-radius: 12px;
                    padding: 1.25rem;
                    margin: 1.5rem 0;
                    box-shadow: 0 4px 12px rgba(243,156,18,0.15);
                    border: 1px solid rgba(243,156,18,0.2);
                    animation: slideInRight 0.5s ease;'>
            <p style='color: #d68910; font-weight: 800; margin: 0; font-size: 1rem;'>💡 {title}</p>
            <p style='color: #7d6608; margin: 0.75rem 0 0 0; line-height: 1.7; font-size: 0.95rem;'>{content}</p>
        </div>
        """, unsafe_allow_html=True)

def concept_card(title, description, examples=None):
    """Display a learning concept card."""
    html = f"""
    <div style='background: linear-gradient(135deg, #e8f4f8 0%, #dbeafe 100%);
                border: 2px solid #667eea;
                border-left: 6px solid #667eea;
                border-radius: 16px;
                padding: 1.75rem;
                margin: 1.5rem 0;
                box-shadow: 0 4px 15px rgba(102,126,234,0.12);
                animation: slideInRight 0.5s ease;'>
        <h4 style='color: #667eea; margin: 0 0 0.75rem 0; font-weight: 800; font-size: 1.15rem;'>📚 {title}</h4>
        <p style='color: #2d3748; margin: 0 0 0.75rem 0; line-height: 1.7; font-size: 0.95rem;'>{description}</p>
    """
    if examples:
        html += "<p style='color: #2d3748; margin: 1rem 0 0.5rem 0; font-weight: 700; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em;'>Examples:</p>"
        html += "<ul style='color: #2d3748; margin: 0.5rem 0; padding-left: 1.5rem;'>"
        for ex in examples:
            html += f"<li style='margin: 0.5rem 0; font-family: monospace; background: rgba(102, 126, 234, 0.1); padding: 0.5rem 0.75rem; border-radius: 8px; border-left: 3px solid #667eea;'><code>{ex}</code></li>"
        html += "</ul>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

# ----------------- PROMPT & MODELS -----------------
if sections["Prompt & Models"]:
    st.markdown("## 📝 Prompt & Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); 
                    padding: 1.5rem; border-radius: 12px; border-left: 4px solid #667eea;'>
            <h4 style='color: #667eea; margin-top: 0;'>🤖 DistilBERT (Encoder)</h4>
            <ul style='color: #2d3748; line-height: 1.8;'>
                <li><strong style='color: #1a1a2e;'>Layers:</strong> 6</li>
                <li><strong style='color: #1a1a2e;'>Heads:</strong> 12 per layer</li>
                <li><strong style='color: #1a1a2e;'>Embedding Size:</strong> 768</li>
                <li><strong style='color: #1a1a2e;'>Use Cases:</strong> Embeddings, Attention, Similarity</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(118, 75, 162, 0.1) 0%, rgba(102, 126, 234, 0.1) 100%); 
                    padding: 1.5rem; border-radius: 12px; border-left: 4px solid #764ba2;'>
            <h4 style='color: #764ba2; margin-top: 0;'>🔮 GPT-2 (Decoder)</h4>
            <ul style='color: #2d3748; line-height: 1.8;'>
                <li><strong style='color: #1a1a2e;'>Layers:</strong> 12</li>
                <li><strong style='color: #1a1a2e;'>Heads:</strong> 12 per layer</li>
                <li><strong style='color: #1a1a2e;'>Embedding Size:</strong> 768</li>
                <li><strong style='color: #1a1a2e;'>Use Cases:</strong> Token Prediction, Generation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style='background: #2d3548; padding: 1rem; border-radius: 8px; margin-top: 1rem; border: 1px solid #667eea;'>
        <strong style='color: #f1f5f9;'>Current Prompt:</strong> <code style='background: #1e2332; color: #a5b4fc; padding: 0.25rem 0.5rem; border-radius: 4px;'>{prompt}</code>
    </div>
    """, unsafe_allow_html=True)

# Intrusive modal now handled above with st.warning/st.expander

# ----------------- TOKENIZATION -----------------
if sections["Tokenization"]:
    st.markdown("## 🔤 Tokenization Analysis")
    
    concept_card(
        "What is Tokenization?",
        "Tokenization breaks your text into <strong>subword units</strong> that the model can process. Each token is mapped to a numerical ID and an embedding vector.",
        examples=["'hello' → ['hello']", "'unhappy' → ['un', '##happy']", "'world123' → ['world', '##123']"]
    )
    
    learning_tip(
        "Why Tokenization Matters",
        "The way text is split into tokens affects how the model understands it. Rare words may split into many tokens, which can impact performance."
    )
    
    token_df = pd.DataFrame({
        "Index": range(len(tokens)), 
        "Token": tokens,
        "Length": [len(t) for t in tokens]
    })
    
    st.dataframe(
        token_df,
        use_container_width=True,
        height=min(400, len(tokens) * 35 + 38)
    )
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Tokens", len(tokens))
    col2.metric("Avg Token Length", f"{token_df['Length'].mean():.1f}")
    col3.metric("Max Token Length", token_df['Length'].max())
    
    # Beginner-level insight
    if learning_path == "🌱 Beginner":
        st.info("✅ **Key Takeaway**: Shorter prompts are easier to tokenize. Long or unusual words may split into multiple tokens.")

# ----------------- EMBEDDINGS -----------------
if sections["Embeddings"]:
    st.markdown("## 📊 Token Embeddings Visualization")
    
    concept_card(
        "What are Embeddings?",
        "Embeddings are <strong>dense vectors</strong> (lists of numbers) that represent the meaning of each token. Similar tokens have similar embeddings."
    )
    
    learning_tip(
        "Reading the Visualization",
        "The chart shows 2D projections of 768-dimensional embeddings using <strong>PCA</strong>. Tokens closer together share semantic meaning."
    )
    
    if show_equations:
        st.latex(r"embedding \, vector \in \mathbb{R}^{768}")
    
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(embeddings.numpy())
    emb_df = pd.DataFrame({"Token": tokens, "X": emb_2d[:,0], "Y": emb_2d[:,1]})
    
    scatter_placeholder = st.empty()
    
    # Animate embedding visualization
    num_frames = min(len(tokens), 15) if intrusive_mode else 1
    for i in range(1, num_frames + 1):
        fig = px.scatter(
            emb_df.iloc[:i], x="X", y="Y", text="Token", color="Y",
            color_continuous_scale="Viridis",
            range_x=[emb_2d[:,0].min()-0.5, emb_2d[:,0].max()+0.5],
            range_y=[emb_2d[:,1].min()-0.5, emb_2d[:,1].max()+0.5],
        )
        fig.update_traces(
            textposition="top center",
            marker=dict(size=14, line=dict(width=2, color='white'), opacity=0.8),
            textfont=dict(size=11, color='#2d3748', family='Inter')
        )
        fig.update_layout(
            title=f"2D PCA Projection of Token Embeddings ({i}/{len(tokens)} tokens)",
            height=550,
            plot_bgcolor='rgba(247,250,252,0.5)',
            paper_bgcolor='white',
            font=dict(family='Inter', color='#2d3748')
        )
        scatter_placeholder.plotly_chart(fig, use_container_width=True, key=f"embed_frame{i}")
        if intrusive_mode:
            time.sleep(0.08)
    
    # Final full visualization
    fig = px.scatter(
        emb_df, x="X", y="Y", text="Token", color="Y",
        color_continuous_scale="Viridis",
        range_x=[emb_2d[:,0].min()-0.5, emb_2d[:,0].max()+0.5],
        range_y=[emb_2d[:,1].min()-0.5, emb_2d[:,1].max()+0.5],
    )
    fig.update_traces(
        textposition="top center",
        marker=dict(size=14, line=dict(width=2, color='white'), opacity=0.8),
        textfont=dict(size=11, color='#2d3748', family='Inter')
    )
    fig.update_layout(
        title="2D PCA Projection of All Token Embeddings",
        height=550,
        plot_bgcolor='rgba(247,250,252,0.5)',
        paper_bgcolor='white',
        font=dict(family='Inter', color='#2d3748')
    )
    scatter_placeholder.plotly_chart(fig, use_container_width=True)

# ----------------- ATTENTION -----------------
if sections["Attention"]:
    st.markdown("## 🧠 Attention Pattern Analysis")
    
    concept_card(
        "What is Attention?",
        "Attention shows which tokens the model focuses on when processing each token. It's the core mechanism that makes transformers work."
    )
    
    learning_tip(
        "Reading Attention Heatmaps",
        "Darker colors = stronger attention. Rows = query tokens (what's attending). Columns = key tokens (what's being attended to)."
    )
    # Add short explainer and examples for users
    with st.expander("Examples & How to probe attention (quick)"):
        st.markdown("""
        - Try a sentence with a pronoun (e.g., "Alice went to the store. She bought milk.") and look for which token ‘She’ attends to — this often reveals coreference links.
        - Use a sentence with a long-range dependency (e.g., "The book that the students read was fascinating") and see how verbs attend to their subjects across distance.
        - Observe punctuation and special tokens — early layers often attend locally (nearby tokens), later layers may spread attention to thematic words.
        """)
    
    if show_equations:
        st.latex(r"Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("### ⚙️ Settings")
        layer_idx = st.slider("Layer", 0, 5, 5)
        head_idx = st.slider("Head", 0, 11, 0)
        # Additional visualization options
        thresh_enable = st.checkbox("🔍 Threshold small weights", value=False)
        thresh_val = st.slider("Min weight", 0.0, 0.2, 0.01, 0.005) if thresh_enable else 0.0
        show_hover = st.checkbox("Show hover values", value=True)
    colorscale = st.selectbox("Color scale", ["Viridis", "Blues", "Purples", "RdYlGn_r", "Inferno", "Cividis"], index=0)
    normalize_mode = st.radio("Normalize", ["None", "Row-normalize", "Col-normalize"], index=0)
    
    with col1:
        attn = attentions[layer_idx][0, head_idx].numpy()
        fig_attn = go.Figure()
        fig_attn.add_trace(go.Heatmap(
            z=np.zeros_like(attn),
            x=tokens, y=tokens, 
            colorscale=colorscale,
            colorbar=dict(title="Attention<br>Weight"),
            hovertemplate=('Query: %{y}<br>Key: %{x}<br>Attention: %{z:.4f}<extra></extra>') if show_hover else None
        ))
        fig_attn.update_layout(
            title=f"Attention Heatmap - Layer {layer_idx}, Head {head_idx}",
            yaxis_autorange="reversed",
            height=550,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Inter', color='#2d3748')
        )
        heatmap_placeholder = st.empty()
        
        # Animate heatmap
        num_steps = 15 if intrusive_mode else 1
        for alpha in np.linspace(0, 1, num_steps):
            z = attn * alpha
            if thresh_enable and thresh_val > 0:
                z = np.where(z >= thresh_val, z, 0.0)
            # apply normalization if requested
            if normalize_mode == "Row-normalize":
                row_sums = z.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1.0
                z = z / row_sums
            elif normalize_mode == "Col-normalize":
                col_sums = z.sum(axis=0, keepdims=True)
                col_sums[col_sums == 0] = 1.0
                z = z / col_sums
            fig_attn.data[0].z = z
            heatmap_placeholder.plotly_chart(fig_attn, use_container_width=True, key=f"attn_{layer_idx}_{head_idx}_frame{int(alpha*100)}")
            if intrusive_mode:
                time.sleep(0.04)

        # Provide downloads for the current attention matrix (applies threshold/normalize as selected)
        try:
            z_display = attn.copy()
            if thresh_enable and thresh_val > 0:
                z_display = np.where(z_display >= thresh_val, z_display, 0.0)
            if normalize_mode == "Row-normalize":
                row_sums = z_display.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1.0
                z_display = z_display / row_sums
            elif normalize_mode == "Col-normalize":
                col_sums = z_display.sum(axis=0, keepdims=True)
                col_sums[col_sums == 0] = 1.0
                z_display = z_display / col_sums

            df = pd.DataFrame(z_display, index=tokens, columns=tokens)
            csv = df.to_csv(index=True)
            json_str = df.to_json(orient='split')
            st.download_button(label="Download attention CSV", data=csv, file_name=f"attention_layer{layer_idx}_head{head_idx}.csv", mime='text/csv')
            st.download_button(label="Download attention JSON", data=json_str, file_name=f"attention_layer{layer_idx}_head{head_idx}.json", mime='application/json')
            try:
                img = fig_attn.to_image(format='png')
                st.download_button(label="Download heatmap PNG", data=img, file_name=f"attention_layer{layer_idx}_head{head_idx}.png", mime='image/png')
            except Exception:
                # kaleido may not be available; skip PNG
                pass
        except Exception:
            pass

    # Persist threshold setting so comparison view can use it
    if thresh_enable:
        st.session_state.attn_thresh = thresh_val
    else:
        st.session_state.attn_thresh = 0.0

    # Beginner insight
    if learning_path == "🌱 Beginner":
        st.success("✅ **Insight**: Early layers (0-2) capture syntax; later layers (4-5) capture semantics. Try comparing them!")

# ----------------- ATTENTION COMPARISON (SIDE-BY-SIDE) -----------------
if sections["Attention Comparison"]:
    st.markdown("## 🔄 Side-by-Side Attention Comparison")
    
    st.info("""
    **Purpose:** Compare attention patterns across layers and heads simultaneously.
    
    **Insights:** Observe how token focus shifts - early layers capture syntax, later layers capture semantics.
    """)
    
    # Controls
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🔵 Left Heatmap")
        left_layer = st.slider("Layer (Left)", 0, 5, 0, key="left_layer")
        left_head = st.slider("Head (Left)", 0, 11, 0, key="left_head")
    
    with col2:
        st.markdown("#### 🟣 Right Heatmap")
        right_layer = st.slider("Layer (Right)", 0, 5, 5, key="right_layer")
        right_head = st.slider("Head (Right)", 0, 11, 0, key="right_head")
    
    # Side-by-side heatmaps
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Layer {left_layer}, Head {left_head}**")
        attn_left = attentions[left_layer][0, left_head].numpy()
        # optionally threshold small weights (for clarity)
        if 'attn_thresh' in st.session_state and st.session_state.attn_thresh > 0:
            attn_left_viz = np.where(attn_left >= st.session_state.attn_thresh, attn_left, 0.0)
        else:
            attn_left_viz = attn_left

        fig_left = go.Figure(go.Heatmap(
            z=attn_left_viz,
            x=tokens,
            y=tokens,
            colorscale="Blues",
            hovertemplate='Query: %{y}<br>Key: %{x}<br>Attention: %{z:.4f}<extra></extra>',
            hoverongaps=False,
            colorbar=dict(title="Attention", x=0.45)
        ))
        fig_left.update_layout(
            yaxis_autorange="reversed",
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Inter', color='#2d3748', size=10),
            margin=dict(l=50, r=50, t=30, b=50)
        )
        st.plotly_chart(fig_left, use_container_width=True)
        # Download buttons for left heatmap
        try:
            df_left = pd.DataFrame(attn_left_viz, index=tokens, columns=tokens)
            st.download_button(label="Download left CSV", data=df_left.to_csv(index=True), file_name=f"left_layer{left_layer}_head{left_head}.csv", mime='text/csv')
            st.download_button(label="Download left JSON", data=df_left.to_json(orient='split'), file_name=f"left_layer{left_layer}_head{left_head}.json", mime='application/json')
            try:
                img_left = fig_left.to_image(format='png')
                st.download_button(label="Download left PNG", data=img_left, file_name=f"left_layer{left_layer}_head{left_head}.png", mime='image/png')
            except Exception:
                pass
        except Exception:
            pass
    
    with col2:
        st.markdown(f"**Layer {right_layer}, Head {right_head}**")
        attn_right = attentions[right_layer][0, right_head].numpy()
        if 'attn_thresh' in st.session_state and st.session_state.attn_thresh > 0:
            attn_right_viz = np.where(attn_right >= st.session_state.attn_thresh, attn_right, 0.0)
        else:
            attn_right_viz = attn_right

        fig_right = go.Figure(go.Heatmap(
            z=attn_right_viz,
            x=tokens,
            y=tokens,
            colorscale="Purples",
            hovertemplate='Query: %{y}<br>Key: %{x}<br>Attention: %{z:.4f}<extra></extra>',
            hoverongaps=False,
            colorbar=dict(title="Attention", x=1.0)
        ))
        fig_right.update_layout(
            yaxis_autorange="reversed",
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Inter', color='#2d3748', size=10),
            margin=dict(l=50, r=50, t=30, b=50)
        )
        st.plotly_chart(fig_right, use_container_width=True)
        # Download buttons for right heatmap
        try:
            df_right = pd.DataFrame(attn_right_viz, index=tokens, columns=tokens)
            st.download_button(label="Download right CSV", data=df_right.to_csv(index=True), file_name=f"right_layer{right_layer}_head{right_head}.csv", mime='text/csv')
            st.download_button(label="Download right JSON", data=df_right.to_json(orient='split'), file_name=f"right_layer{right_layer}_head{right_head}.json", mime='application/json')
            try:
                img_right = fig_right.to_image(format='png')
                st.download_button(label="Download right PNG", data=img_right, file_name=f"right_layer{right_layer}_head{right_head}.png", mime='image/png')
            except Exception:
                pass
        except Exception:
            pass
    
    # Difference view
    st.markdown("### 📊 Attention Difference")
    st.info("Shows the absolute difference between the two attention patterns above.")
    
    diff = np.abs(attn_left - attn_right)
    fig_diff = go.Figure(go.Heatmap(
        z=diff,
        x=tokens,
        y=tokens,
        colorscale="RdYlGn_r",
        hovertemplate='Query: %{y}<br>Key: %{x}<br>Diff: %{z:.4f}<extra></extra>',
        hoverongaps=False,
        colorbar=dict(title="Absolute<br>Difference")
    ))
    fig_diff.update_layout(
        title="Attention Pattern Difference",
        yaxis_autorange="reversed",
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Inter', color='#2d3748')
    )
    st.plotly_chart(fig_diff, use_container_width=True)

    # Expanded explanation for attention heatmaps
    with st.expander("How to read these attention heatmaps — detailed guide", expanded=False):
        st.markdown("""
        **What's shown:** Each heatmap cell represents how much the token on the Y axis (the "query") attends to the token on the X axis (the "key"). Values come from the transformer's softmax and typically sum to ~1 across a row.

        **Reading tips:**
        - Darker/brighter colors show stronger attention.
        - Diagonal patterns often mean tokens attend to themselves or nearby tokens (useful for syntax).
        - Off-diagonal high values indicate long-range dependencies (e.g., coreference, topic words).
        - Use the *Attention Comparison* mode to spot how attention shifts between layers/heads.

        **Try this:**
        1. Pick an informative token (e.g., a verb or named entity) and look across its row to see which tokens influence it.
        2. Compare early vs. late layers — early layers focus locally (syntax), later layers capture semantics (topic/meaning).
        3. Toggle thresholding to remove noise and highlight strong interactions.

        **Limitations:**
        - Attention weights aren't the whole story — they are one interpretation of model behavior, not definitive proof of causality.
        - Small numerical differences may not be meaningful; use thresholding and layer comparisons for robustness.
        """)

    # Add a short, visible helper directly above the Token Similarity section
    st.markdown(
        """
        <div style='margin-top: 1rem; padding: 0.75rem; border-radius: 8px; background: linear-gradient(90deg, #eef2ff, #f8fafc); color: #1a1a2e;'>
            <strong>Tip:</strong> Hover any heatmap cell to see exact values. Use thresholding to hide small weights and reveal strong token interactions.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------- TOKEN SIMILARITY -----------------
if sections["Token Similarity"]:
    st.markdown("## 💡 Token Similarity Matrix")
    
    st.info("""
    **What is Similarity?** Cosine similarity between token embeddings, measuring semantic relatedness.
    
    **How to Read:** Brighter colors indicate higher similarity (0-1 scale).
    """)
    
    similarity = cosine_similarity(embeddings.numpy())
    fig_sim = go.Figure(go.Heatmap(
        z=similarity, x=tokens, y=tokens, 
        colorscale="Viridis",
        hoverongaps=False,
        colorbar=dict(title="Cosine<br>Similarity")
    ))
    fig_sim.update_layout(
        title="Token-to-Token Similarity Heatmap",
        yaxis_autorange="reversed",
        height=550,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Inter', color='#2d3748')
    )
    st.plotly_chart(fig_sim, use_container_width=True)

# ----------------- NEXT TOKEN -----------------
if sections["Next Token"]:
    st.markdown("## ➡️ Next Token Prediction")
    
    concept_card(
        "What is Token Prediction?",
        "The model's output layer computes a probability distribution over all possible next tokens. The highest probability token is the top prediction."
    )
    
    learning_tip(
        "Understanding Confidence",
        "Low probabilities (spread distribution) indicate uncertainty. High confidence (peaked distribution) shows the model is confident in its prediction."
    )
    
    if show_equations:
        st.latex(r"P(next\_token | context) = softmax(output\_logits)")
    
    gpt_inputs = gpt2_tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = gpt2_model(**gpt_inputs)
        logits = out.logits[0, -1]
        probs = torch.softmax(logits, dim=0)
    topk = torch.topk(probs, 10)
    top_tokens = [gpt2_tokenizer.decode([i]) for i in topk.indices]
    top_probs = topk.values.numpy()
    
    # apply a pulsing card style if intrusive mode is enabled
    card_cls = 'pulse-card' if intrusive_mode else ''
    st.markdown(f"""
    <div class='{card_cls}' style='background: linear-gradient(135deg, rgba(72, 187, 120, 0.1) 0%, rgba(56, 161, 105, 0.1) 100%); 
                padding: 1.5rem; border-radius: 12px; border-left: 4px solid #48bb78; margin-bottom: 1rem;'>
        <h4 style='color: #48bb78; margin-top: 0;'>🎯 Top Prediction</h4>
        <p style='font-size: 1.5rem; font-weight: 600; color: #1a1a2e; margin: 0;'>
            "<code style='background: white; color: #667eea; padding: 0.25rem 0.5rem; border-radius: 4px;'>{top_tokens[0]}</code>" 
            <span style='color: #48bb78;'>({top_probs[0]*100:.1f}%)</span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    fig_probs = go.Figure(go.Bar(
        x=top_tokens, 
        y=np.zeros_like(top_probs),
        marker=dict(
            color=top_probs,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Probability")
        ),
        text=[f"{p*100:.1f}%" for p in top_probs],
        textposition='outside'
    ))
    fig_probs.update_layout(
        title="Top 10 Predicted Tokens",
        xaxis_title="Token",
        yaxis_title="Probability",
        height=450,
        plot_bgcolor='rgba(247,250,252,0.5)',
        paper_bgcolor='white',
        font=dict(family='Inter', color='#2d3748')
    )
    
    bar_placeholder = st.empty()
    num_steps = 20 if intrusive_mode else 1
    for alpha in np.linspace(0, 1, num_steps):
        fig_probs.data[0].y = top_probs * alpha
        bar_placeholder.plotly_chart(fig_probs, use_container_width=True, key=f"probs_frame{int(alpha*100)}")
        if intrusive_mode:
            time.sleep(0.04)
    
    # Interpretability
    if learning_path == "🌿 Intermediate" or learning_path == "🌳 Advanced":
        with st.expander("📊 Prediction Details"):
            st.write(f"**Model**: GPT-2 (12 layers, 768-dim)")
            st.write(f"**Output Layer Size**: ~50K tokens in vocabulary")
            st.write(f"**Top 3 Predictions**:")
            for i, (tok, prob) in enumerate(zip(top_tokens[:3], top_probs[:3])):
                st.write(f"  {i+1}. `{tok}` — {prob*100:.2f}%")

# ----------------- TEXT GENERATION -----------------
if sections["Text Generation"]:
    st.markdown("## ✍️ Token-by-Token Text Generation")
    
    concept_card(
        "How Does Text Generation Work?",
        "The model iteratively predicts the next token based on all previous tokens. Each new token is added to the context for the next prediction."
    )
    
    learning_tip(
        "Autoregressive Decoding",
        "This is <strong>autoregressive generation</strong>: prediction depends on prior outputs. That's why early mistakes compound over time."
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        max_steps = st.slider("Tokens to generate", 5, 30, 15)
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        generate_btn = st.button("▶ Generate Text", use_container_width=True)
    
    if generate_btn:
        ids = gpt2_tokenizer(prompt, return_tensors="pt")["input_ids"]
        output = prompt
        
        st.markdown("### 📝 Generated Output")
        progress_bar = st.progress(0)
        placeholder = st.empty()
        
        for i in range(max_steps):
            with torch.no_grad():
                out = gpt2_model(ids)
                probs = torch.softmax(out.logits[0, -1], dim=0)
            next_id = torch.argmax(probs).unsqueeze(0).unsqueeze(0)
            next_token = gpt2_tokenizer.decode(next_id[0])
            ids = torch.cat([ids, next_id], dim=1)
            output += next_token
            
            placeholder.markdown(f"""
            <div style='background: #f5f7fa; padding: 1.5rem; border-radius: 12px; 
                        border: 2px solid #e2e8f0; font-family: monospace; 
                        font-size: 1rem; line-height: 1.6; color: #2d3748;'>
                {output}
                <span style='animation: blink 1s infinite; color: #667eea; font-weight: bold;'>▌</span>
            </div>
            """, unsafe_allow_html=True)
            progress_bar.progress((i+1) / max_steps)
            time.sleep(0.12)
        
        st.success(f"✅ Generated {max_steps} tokens!")
        if learning_path == "🌿 Intermediate" or learning_path == "🌳 Advanced":
            with st.expander("🔍 Generation Details"):
                st.write(f"**Total prompt length**: {len(ids[0])} tokens")
                st.write(f"**Final output**: {output}")
                st.write(f"**Model used**: GPT-2 (175M parameters)")

# ----------------- DEBUG SUMMARY -----------------
if sections["Debug Summary"]:
    st.markdown("## 📋 Learning Summary & Diagnostics")
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #e8f4f8 0%, #dbeafe 100%);
                border: 2px solid #667eea;
                border-left: 6px solid #667eea;
                border-radius: 16px;
                padding: 1.75rem;
                margin-bottom: 2rem;
                box-shadow: 0 4px 15px rgba(102,126,234,0.12);'>
        <h4 style='color: #667eea; margin: 0 0 0.75rem 0; font-weight: 800; font-size: 1.15rem;'>📚 What You're Learning</h4>
        <p style='color: #2d3748; margin: 0.5rem 0; line-height: 1.7; font-size: 0.95rem;'>
            This lab teaches you about the internal mechanisms of Large Language Models: how text is tokenized, 
            converted to embeddings, processed through attention layers, and used to predict the next token.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    token_count = len(tokens)
    confidence = top_probs[0] if len(top_probs) > 0 else 0.0
    avg_embedding_norm = embeddings.norm(dim=1).mean().item()
    
    # Enhanced metric cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #e0f2fe 0%, #dbeafe 100%);
                    border: 2px solid #667eea;
                    border-radius: 16px;
                    padding: 1.5rem;
                    text-align: center;
                    box-shadow: 0 4px 12px rgba(102,126,234,0.15);
                    transition: all 0.3s ease;'>
            <p style='color: #667eea; font-size: 2.5rem; font-weight: 800; margin: 0;'>{token_count}</p>
            <p style='color: #2d3748; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; margin: 0.5rem 0 0 0;'>Token Count</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #e0f5f0 0%, #d1fae5 100%);
                    border: 2px solid #48bb78;
                    border-radius: 16px;
                    padding: 1.5rem;
                    text-align: center;
                    box-shadow: 0 4px 12px rgba(72,187,120,0.15);
                    transition: all 0.3s ease;'>
            <p style='color: #48bb78; font-size: 2.5rem; font-weight: 800; margin: 0;'>{confidence*100:.1f}%</p>
            <p style='color: #2d3748; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; margin: 0.5rem 0 0 0;'>Confidence</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #fef5e7 0%, #fef9e7 100%);
                    border: 2px solid #f39c12;
                    border-radius: 16px;
                    padding: 1.5rem;
                    text-align: center;
                    box-shadow: 0 4px 12px rgba(243,156,18,0.15);
                    transition: all 0.3s ease;'>
            <p style='color: #f39c12; font-size: 2.5rem; font-weight: 800; margin: 0;'>{avg_embedding_norm:.2f}</p>
            <p style='color: #2d3748; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; margin: 0.5rem 0 0 0;'>Embedding Norm</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #f5e7ff 0%, #ede9fe 100%);
                    border: 2px solid #a78bfa;
                    border-radius: 16px;
                    padding: 1.5rem;
                    text-align: center;
                    box-shadow: 0 4px 12px rgba(167,139,250,0.15);
                    transition: all 0.3s ease;'>
            <p style='color: #a78bfa; font-size: 2.5rem; font-weight: 800; margin: 0;'>{len(bert_tokenizer):,}</p>
            <p style='color: #2d3748; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; margin: 0.5rem 0 0 0;'>Vocab Size</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Diagnostics & health checks
    st.markdown("""
    <h3 style='color: #1a1a2e; font-weight: 800; margin-top: 2rem; margin-bottom: 1.5rem;
                border-bottom: 3px solid #667eea;
                padding-bottom: 0.75rem;
                display: inline-block;'>🔍 Model Health Checks</h3>
    """, unsafe_allow_html=True)
    
    warnings = []
    
    if token_count < 4:
        warnings.append(("⚠️ Very Short Prompt", "Your prompt has fewer than 4 tokens. The model may struggle with context.", "warning"))
    if token_count > 60:
        warnings.append(("⚠️ Long Prompt", "Your prompt exceeds 60 tokens. This may impact speed and attention patterns.", "warning"))
    if confidence < 0.15:
        warnings.append(("⚠️ Low Confidence", "The model is uncertain about the next token. Try a more specific prompt.", "warning"))
    if confidence > 0.5:
        warnings.append(("✅ High Confidence", "The model is very confident in its prediction.", "success"))
    
    if warnings:
        for title, msg, wtype in warnings:
            if wtype == "success":
                st.success(f"**{title}**: {msg}")
            else:
                st.warning(f"**{title}**: {msg}")
    else:
        st.success("✅ All diagnostics passed! Your prompt is well-balanced.")
    
    # Learning path recommendations
    st.markdown("### 📚 Next Steps")
    if learning_path == "🌱 Beginner":
        st.markdown("""
        **Beginner Recommendations:**
        1. Try different prompts and observe how tokenization changes.
        2. Watch the embeddings visualization — notice which tokens cluster together.
        3. Experiment with the attention heatmaps — see how the model attends to different words.
        """)
    elif learning_path == "🌿 Intermediate":
        st.markdown("""
        **Intermediate Recommendations:**
        1. Analyze attention patterns across layers — notice how they specialize.
        2. Compare attention heads side-by-side to understand their roles.
        3. Explore token similarity matrices — understand semantic relationships.
        4. Try edge cases: very long prompts, ambiguous queries, technical text.
        """)
    else:  # Advanced
        st.markdown("""
        **Advanced Recommendations:**
        1. Debug specific behaviors: why does the model predict this token?
        2. Analyze failure cases: what prompts trip up the model?
        3. Study head specialization: which heads focus on what?
        4. Experiment with layer-wise interventions: how important is each layer?
        5. Consider probing: what linguistic information is encoded at each layer?
        """)
    
    # Export data
    st.markdown("### 💾 Export Your Analysis")
    df = pd.DataFrame({
        "Token": tokens, 
        "Embedding Norm": embeddings.norm(dim=1).detach().numpy(),
        "Index": range(len(tokens))
    })
    
    col1, col2 = st.columns([3, 1])
    with col2:
        st.download_button(
            "📥 Download CSV",
            df.to_csv(index=False),
            "llm_learning_analysis.csv",
            "text/csv",
            use_container_width=True
        )