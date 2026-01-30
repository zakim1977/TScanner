"""
CSS Styles for Crypto Scanner Pro
Dark theme with proper contrast
"""

import streamlit as st

def apply_custom_css():
    """Apply custom CSS styling to the app"""
    
    st.markdown("""
<style>
    /* ══════════════════════════════════════════════════════════════════════
       GLOBAL: Dark backgrounds everywhere in main area
       ══════════════════════════════════════════════════════════════════════ */
    .stApp {
        background-color: #0e1117 !important;
    }
    
    .main .block-container {
        background-color: #0e1117 !important;
    }
    
    /* ══════════════════════════════════════════════════════════════════════
       ALL TEXT IN MAIN = WHITE (nuclear option)
       EXCEPT: code blocks, pre tags, alerts (handled separately)
       ══════════════════════════════════════════════════════════════════════ */
    .main, 
    .main p, .main span, .main div, .main label,
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6,
    .main li, .main strong, .main b, .main em, .main i,
    .main td, .main th, .main a,
    .block-container,
    .element-container,
    [data-testid="stMarkdownContainer"],
    [data-testid="stVerticalBlock"],
    [data-testid="stMetric"],
    [data-testid="stMetricLabel"] {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    
    /* Exception for subtitle/secondary text - allow gray */
    .subtitle-text, .main .subtitle-text {
        color: #888888 !important;
        -webkit-text-fill-color: #888888 !important;
    }
    
    /* Exception for cyan accent text */
    .cyan-text, .main .cyan-text {
        color: #00d4ff !important;
        -webkit-text-fill-color: #00d4ff !important;
    }
    
    /* Metric values - cyan */
    [data-testid="stMetricValue"] {
        color: #00d4ff !important;
        -webkit-text-fill-color: #00d4ff !important;
    }
    
    /* ══════════════════════════════════════════════════════════════════════
       TOOLTIPS - Dark text on light background (override white text rule)
       ══════════════════════════════════════════════════════════════════════ */
    
    /* Streamlit native tooltips */
    [data-baseweb="tooltip"],
    [data-baseweb="tooltip"] *,
    [role="tooltip"],
    [role="tooltip"] *,
    .stTooltipIcon + div,
    .stTooltipIcon + div * {
        color: #1a1a2e !important;
        -webkit-text-fill-color: #1a1a2e !important;
        background-color: #ffffff !important;
    }
    
    /* Tooltip content specifically */
    [data-baseweb="tooltip"] [data-testid="stMarkdownContainer"],
    [data-baseweb="tooltip"] [data-testid="stMarkdownContainer"] *,
    [data-baseweb="tooltip"] p,
    [data-baseweb="tooltip"] span,
    [data-baseweb="tooltip"] div {
        color: #1a1a2e !important;
        -webkit-text-fill-color: #1a1a2e !important;
    }
    
    /* Popover tooltips (help icons) */
    [data-baseweb="popover"],
    [data-baseweb="popover"] *,
    [data-baseweb="popover"] [data-testid="stMarkdownContainer"],
    [data-baseweb="popover"] [data-testid="stMarkdownContainer"] * {
        color: #1a1a2e !important;
        -webkit-text-fill-color: #1a1a2e !important;
    }
    
    /* ══════════════════════════════════════════════════════════════════════
       EXPANDERS - Force dark background + white text
       ══════════════════════════════════════════════════════════════════════ */
    
    /* Expander container */
    [data-testid="stExpander"] {
        background-color: #1a1a2e !important;
        border: 1px solid #333 !important;
        border-radius: 8px !important;
    }
    
    /* Expander header/summary */
    [data-testid="stExpander"] > details > summary,
    [data-testid="stExpander"] summary,
    .streamlit-expanderHeader {
        background-color: #1a1a2e !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        padding: 12px !important;
        border-radius: 8px !important;
    }
    
    /* Everything inside expander header */
    [data-testid="stExpander"] > details > summary *,
    [data-testid="stExpander"] summary *,
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary div,
    .streamlit-expanderHeader * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        background-color: transparent !important;
    }
    
    /* Expander content/body */
    [data-testid="stExpander"] > details > div[data-testid="stExpanderDetails"],
    [data-testid="stExpander"] [data-testid="stExpanderDetails"],
    [data-testid="stExpanderDetails"] {
        background-color: #0e1117 !important;
        border-top: 1px solid #333 !important;
    }
    
    /* Hover state */
    [data-testid="stExpander"] summary:hover {
        background-color: #252540 !important;
    }
    
    /* ══════════════════════════════════════════════════════════════════════
       PROGRESS BAR
       ══════════════════════════════════════════════════════════════════════ */
    [data-testid="stProgress"] > div {
        background-color: #1a1a2e !important;
    }
    
    /* ══════════════════════════════════════════════════════════════════════
       BUTTONS - Simple Rule: White bg = Black text, Dark bg = White text
       ══════════════════════════════════════════════════════════════════════ */
    
    /* Default: Force dark background + white text */
    .main button,
    .main [data-testid="baseButton-secondary"],
    .main [data-testid="baseButton-primary"],
    .main [data-baseweb="button"],
    .main .stButton > button {
        background-color: #1a1a2e !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        border: 1px solid #444 !important;
    }
    
    .main button *,
    .main [data-baseweb="button"] * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    
    /* OVERRIDE: Any button/element with white/light background = BLACK text */
    button[style*="background-color: white"],
    button[style*="background-color: rgb(255"],
    button[style*="background: white"],
    button[style*="background: rgb(255"],
    [style*="background-color: white"],
    [style*="background-color: rgb(255, 255, 255)"],
    [style*="background: white"],
    [style*="background: rgb(255"] {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }
    
    /* Streamlit's default button styles - catch all variations */
    [data-baseweb="button"][kind="secondary"],
    [data-baseweb="button"][kind="tertiary"],
    button[kind="secondary"],
    button[kind="tertiary"] {
        background-color: #1a1a2e !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    
    /* Primary buttons - keep distinct color */
    [data-baseweb="button"][kind="primary"],
    button[kind="primary"],
    .main [data-testid="baseButton-primary"] {
        background-color: #00d4ff !important;
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }
    
    /* Hover states */
    .main button:hover {
        background-color: #252540 !important;
        border-color: #00d4ff !important;
    }
    
    /* ══════════════════════════════════════════════════════════════════════
       UNIVERSAL CONTRAST RULE: Any light background = Dark text
       ══════════════════════════════════════════════════════════════════════ */
    
    /* White backgrounds need black text */
    .main [style*="background-color: white"] *,
    .main [style*="background-color: #fff"] *,
    .main [style*="background-color: rgb(255, 255, 255)"] *,
    .main [style*="background: white"] *,
    .main [style*="background: #fff"] * {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }
    
    /* ══════════════════════════════════════════════════════════════════════
       SIDEBAR - BLACK TEXT (light background)
       ══════════════════════════════════════════════════════════════════════ */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] *,
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] * {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }
    
    /* ══════════════════════════════════════════════════════════════════════
       SIDEBAR EXPANDER (dark background) - WHITE TEXT FOR LABELS
       Only for expanders inside sidebar that have dark backgrounds
       ══════════════════════════════════════════════════════════════════════ */
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderDetails"],
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderDetails"] p,
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderDetails"] > div > label,
    [data-testid="stSidebar"] [data-testid="stExpander"] .stMarkdown,
    [data-testid="stSidebar"] [data-testid="stExpander"] .stMarkdown p,
    [data-testid="stSidebar"] [data-testid="stExpander"] .stCaption,
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stCaptionContainer"] {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    
    /* ══════════════════════════════════════════════════════════════════════
       SIDEBAR EXPANDER - RADIO BUTTONS 
       Radio buttons have WHITE background (from inline CSS), need BLACK text
       ══════════════════════════════════════════════════════════════════════ */
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stRadio"] > label,
    [data-testid="stSidebar"] [data-testid="stExpander"] .stRadio > label {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    
    /* Radio button OPTIONS (inside white container) need BLACK text */
    [data-testid="stSidebar"] [data-testid="stExpander"] [role="radiogroup"],
    [data-testid="stSidebar"] [data-testid="stExpander"] [role="radiogroup"] *,
    [data-testid="stSidebar"] [data-testid="stExpander"] [role="radiogroup"] label,
    [data-testid="stSidebar"] [data-testid="stExpander"] [role="radiogroup"] label span,
    [data-testid="stSidebar"] [data-testid="stExpander"] [role="radiogroup"] label p,
    [data-testid="stSidebar"] [data-testid="stExpander"] [role="radiogroup"] label div,
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-baseweb="radio"],
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-baseweb="radio"] *,
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-baseweb="radio"] label,
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-baseweb="radio"] label span,
    [data-testid="stSidebar"] [data-testid="stExpander"] .stRadio [role="radiogroup"],
    [data-testid="stSidebar"] [data-testid="stExpander"] .stRadio [role="radiogroup"] *,
    [data-testid="stSidebar"] [data-testid="stExpander"] .stRadio > div,
    [data-testid="stSidebar"] [data-testid="stExpander"] .stRadio > div *,
    [data-testid="stSidebar"] [data-testid="stExpander"] .stRadio > div > div,
    [data-testid="stSidebar"] [data-testid="stExpander"] .stRadio > div > div *,
    [data-testid="stSidebar"] [data-testid="stExpander"] .stRadio > div > label,
    [data-testid="stSidebar"] [data-testid="stExpander"] .stRadio > div > label * {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }
    
    /* ══════════════════════════════════════════════════════════════════════
       SIDEBAR EXPANDER - CHECKBOXES 
       Checkboxes may have WHITE background, need BLACK text for options
       ══════════════════════════════════════════════════════════════════════ */
    [data-testid="stSidebar"] [data-testid="stExpander"] .stCheckbox > label:first-child {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-baseweb="checkbox"],
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-baseweb="checkbox"] *,
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-baseweb="checkbox"] label,
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-baseweb="checkbox"] label span,
    [data-testid="stSidebar"] [data-testid="stExpander"] .stCheckbox label,
    [data-testid="stSidebar"] [data-testid="stExpander"] .stCheckbox label * {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }
    
    /* ══════════════════════════════════════════════════════════════════════
       SIDEBAR EXPANDER - SELECTBOX/DROPDOWN (BLACK text on WHITE background)
       This overrides the white text specifically for dropdown controls
       ══════════════════════════════════════════════════════════════════════ */
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-baseweb="select"],
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-baseweb="select"] *,
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-baseweb="select"] span,
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-baseweb="select"] div,
    [data-testid="stSidebar"] [data-testid="stExpander"] .stSelectbox [data-baseweb="select"],
    [data-testid="stSidebar"] [data-testid="stExpander"] .stSelectbox [data-baseweb="select"] *,
    [data-testid="stSidebar"] [data-testid="stExpander"] .stMultiSelect [data-baseweb="select"],
    [data-testid="stSidebar"] [data-testid="stExpander"] .stMultiSelect [data-baseweb="select"] * {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }
    
    /* Selectbox label should still be white */
    [data-testid="stSidebar"] [data-testid="stExpander"] .stSelectbox > label,
    [data-testid="stSidebar"] [data-testid="stExpander"] .stMultiSelect > label {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    
    /* ══════════════════════════════════════════════════════════════════════
       SIDEBAR EXPANDER - TEXT INPUT (BLACK text on WHITE background)
       ══════════════════════════════════════════════════════════════════════ */
    [data-testid="stSidebar"] [data-testid="stExpander"] input[type="text"],
    [data-testid="stSidebar"] [data-testid="stExpander"] input[type="number"],
    [data-testid="stSidebar"] [data-testid="stExpander"] input[type="password"],
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-baseweb="input"] input,
    [data-testid="stSidebar"] [data-testid="stExpander"] .stTextInput input,
    [data-testid="stSidebar"] [data-testid="stExpander"] .stNumberInput input {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        background-color: #ffffff !important;
    }
    
    /* Text input label should still be white */
    [data-testid="stSidebar"] [data-testid="stExpander"] .stTextInput > label,
    [data-testid="stSidebar"] [data-testid="stExpander"] .stNumberInput > label {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    
    /* ══════════════════════════════════════════════════════════════════════
       SIDEBAR EXPANDER - SUPER AGGRESSIVE FIX FOR ALL CONTROLS
       Force black text on all white-background input controls
       ══════════════════════════════════════════════════════════════════════ */
    
    /* Any div with white background inside expander - force black text */
    [data-testid="stSidebar"] [data-testid="stExpander"] div[style*="background: #ffffff"],
    [data-testid="stSidebar"] [data-testid="stExpander"] div[style*="background:#ffffff"],
    [data-testid="stSidebar"] [data-testid="stExpander"] div[style*="background: white"],
    [data-testid="stSidebar"] [data-testid="stExpander"] div[style*="background:white"] {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }
    
    /* Force ALL nested elements in white-bg containers to be black */
    [data-testid="stSidebar"] [data-testid="stExpander"] div[style*="background: #ffffff"] *,
    [data-testid="stSidebar"] [data-testid="stExpander"] div[style*="background:#ffffff"] *,
    [data-testid="stSidebar"] [data-testid="stExpander"] div[style*="background: white"] *,
    [data-testid="stSidebar"] [data-testid="stExpander"] div[style*="background:white"] * {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }
    
    /* ══════════════════════════════════════════════════════════════════════
       SIDEBAR EXPANDER - DATE INPUT (BLACK text on WHITE background)
       ══════════════════════════════════════════════════════════════════════ */
    [data-testid="stSidebar"] [data-testid="stExpander"] .stDateInput input,
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-baseweb="datepicker"] input,
    [data-testid="stSidebar"] [data-testid="stExpander"] input[type="date"] {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        background-color: #ffffff !important;
    }
    
    /* Date input label should be white */
    [data-testid="stSidebar"] [data-testid="stExpander"] .stDateInput > label {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    
    /* ══════════════════════════════════════════════════════════════════════
       SIDEBAR EXPANDER - IMAGE CONTAINERS (for upload previews)
       ══════════════════════════════════════════════════════════════════════ */
    [data-testid="stSidebar"] [data-testid="stExpander"] .stImage,
    [data-testid="stSidebar"] [data-testid="stExpander"] img {
        border-radius: 8px;
        border: 1px solid #444;
    }
    
    /* ══════════════════════════════════════════════════════════════════════
       ALERT/INFO/SUCCESS/WARNING/ERROR BOXES - Light bg, dark text
       ══════════════════════════════════════════════════════════════════════ */
    
    [data-testid="stAlert"],
    [data-baseweb="notification"],
    [role="alert"] {
        background-color: #f0f2f6 !important;
        border-radius: 8px !important;
    }
    
    [data-testid="stAlert"] *,
    [data-testid="stAlert"] p,
    [data-testid="stAlert"] span,
    [data-testid="stAlert"] div,
    [data-testid="stAlert"] li,
    [data-baseweb="notification"] *,
    [role="alert"] * {
        color: #1a1a1a !important;
        -webkit-text-fill-color: #1a1a1a !important;
        background-color: transparent !important;
    }
    
    /* SIDEBAR EXPANDER ALERTS - Force dark text (override white text rule) */
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stAlert"],
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stAlert"] *,
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stAlert"] p,
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stAlert"] span,
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stAlert"] div,
    [data-testid="stSidebar"] [data-testid="stExpander"] [role="alert"],
    [data-testid="stSidebar"] [data-testid="stExpander"] [role="alert"] * {
        color: #1a1a1a !important;
        -webkit-text-fill-color: #1a1a1a !important;
    }
    
    /* Success - green */
    [data-testid="stAlert"][data-baseweb*="positive"],
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stAlert"][data-baseweb*="positive"] {
        background-color: #d4edda !important;
        border-left: 4px solid #28a745 !important;
    }
    
    /* Warning - yellow */
    [data-testid="stAlert"][data-baseweb*="warning"],
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stAlert"][data-baseweb*="warning"] {
        background-color: #fff3cd !important;
        border-left: 4px solid #ffc107 !important;
    }
    
    /* Info - blue */
    [data-testid="stAlert"][data-baseweb*="info"],
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stAlert"][data-baseweb*="info"] {
        background-color: #d1ecf1 !important;
        border-left: 4px solid #17a2b8 !important;
    }
    
    /* Error - red */
    [data-testid="stAlert"][data-baseweb*="negative"],
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stAlert"][data-baseweb*="negative"] {
        background-color: #f8d7da !important;
        border-left: 4px solid #dc3545 !important;
    }
    
    /* ══════════════════════════════════════════════════════════════════════
       TABLES - White text on dark
       ══════════════════════════════════════════════════════════════════════ */
    .main table, .main table *,
    .main thead, .main thead *,
    .main tbody, .main tbody *,
    .main th, .main td {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        background-color: transparent !important;
    }
    
    /* ══════════════════════════════════════════════════════════════════════
       INPUT FIELDS in main area
       ══════════════════════════════════════════════════════════════════════ */
    .main input, .main textarea, .main select {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        background-color: #1a1a2e !important;
        border: 1px solid #444 !important;
    }
    
    /* NUMBER INPUT - Dark text on light background for visibility */
    .stNumberInput input,
    [data-testid="stNumberInput"] input,
    .stNumberInput [data-baseweb="input"] input,
    [data-baseweb="input"] input {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        background-color: #ffffff !important;
        border: 1px solid #ccc !important;
    }
    
    /* SLIDER INPUT - Same treatment */
    .stSlider input,
    [data-testid="stSlider"] input {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        background-color: #ffffff !important;
    }
    
    /* ══════════════════════════════════════════════════════════════════════
       ALL SELECTBOX / DROPDOWN - WHITE BACKGROUND, DARK TEXT (UNIVERSAL)
       NO EXCEPTIONS - All dropdowns in sidebar, main area, everywhere!
       ══════════════════════════════════════════════════════════════════════ */
    
    /* === SELECTBOX CONTAINER (the closed/collapsed state) === */
    [data-baseweb="select"],
    [data-baseweb="select"] > div,
    [data-baseweb="select"] > div > div,
    [data-baseweb="select"] > div > div > div,
    .stSelectbox > div,
    .stSelectbox > div > div,
    .stSelectbox [data-baseweb="select"],
    .stSelectbox [data-baseweb="select"] > div,
    .stMultiSelect > div,
    .stMultiSelect [data-baseweb="select"],
    .main [data-baseweb="select"],
    .main [data-baseweb="select"] > div,
    .block-container [data-baseweb="select"],
    .block-container [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-baseweb="select"],
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border-color: #ccc !important;
        border-radius: 6px !important;
    }
    
    /* === SELECTBOX TEXT (the selected value shown) === */
    [data-baseweb="select"] span,
    [data-baseweb="select"] div[class*="singleValue"],
    [data-baseweb="select"] [class*="valueContainer"] span,
    .stSelectbox span,
    .stSelectbox div,
    .stMultiSelect span,
    .main [data-baseweb="select"] span,
    .main .stSelectbox span,
    .block-container [data-baseweb="select"] span,
    [data-testid="stSidebar"] [data-baseweb="select"] span,
    [data-testid="stSidebar"] .stSelectbox span {
        color: #1a1a2e !important;
        -webkit-text-fill-color: #1a1a2e !important;
    }
    
    /* === DROPDOWN ARROW/ICON === */
    [data-baseweb="select"] svg,
    .stSelectbox svg {
        fill: #666 !important;
    }
    
    /* === DROPDOWN MENU (the expanded popover with options) === */
    [data-baseweb="popover"],
    [data-baseweb="popover"] > div,
    [data-baseweb="menu"],
    [data-baseweb="menu"] > div,
    [role="listbox"],
    [role="listbox"] > div,
    ul[role="listbox"],
    div[data-baseweb="popover"],
    div[data-baseweb="menu"] {
        background-color: #ffffff !important;
        border: 1px solid #ddd !important;
        border-radius: 6px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }
    
    /* === DROPDOWN OPTIONS (each item in the list) === */
    [data-baseweb="menu"] li,
    [data-baseweb="popover"] li,
    [role="option"],
    [role="listbox"] li,
    [role="listbox"] [role="option"],
    ul[role="listbox"] li {
        background-color: #ffffff !important;
        color: #1a1a2e !important;
        -webkit-text-fill-color: #1a1a2e !important;
        padding: 8px 12px !important;
    }
    
    /* === DROPDOWN OPTION TEXT (spans inside options) === */
    [data-baseweb="menu"] li *,
    [data-baseweb="popover"] li *,
    [role="option"] *,
    [role="option"] span,
    [role="option"] div,
    [role="listbox"] li span,
    [role="listbox"] li div {
        color: #1a1a2e !important;
        -webkit-text-fill-color: #1a1a2e !important;
        background-color: transparent !important;
    }
    
    /* === DROPDOWN OPTION HOVER === */
    [data-baseweb="menu"] li:hover,
    [data-baseweb="popover"] li:hover,
    [role="option"]:hover,
    [role="option"][data-highlighted="true"],
    [role="listbox"] li:hover {
        background-color: #e3f2fd !important;
    }
    
    /* === DROPDOWN SELECTED OPTION === */
    [aria-selected="true"],
    [role="option"][aria-selected="true"],
    [data-baseweb="menu"] li[aria-selected="true"],
    [role="listbox"] li[aria-selected="true"] {
        background-color: #bbdefb !important;
    }
    
    /* === MULTISELECT TAGS === */
    [data-baseweb="tag"],
    [class*="tagContainer"],
    span[data-baseweb="tag"],
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #e3f2fd !important;
        color: #1976d2 !important;
        -webkit-text-fill-color: #1976d2 !important;
        border: none !important;
        border-radius: 4px !important;
    }
    
    /* === MULTISELECT TAG CLOSE BUTTON === */
    [data-baseweb="tag"] svg,
    [data-baseweb="tag"] path {
        fill: #1976d2 !important;
    }
    
    /* === INPUT FIELDS IN DROPDOWNS (search box) === */
    [data-baseweb="select"] input,
    [data-baseweb="popover"] input,
    .stSelectbox input {
        color: #1a1a2e !important;
        -webkit-text-fill-color: #1a1a2e !important;
        background-color: #ffffff !important;
    }
    
    /* === PLACEHOLDER TEXT === */
    [data-baseweb="select"] [class*="placeholder"],
    .stSelectbox [class*="placeholder"] {
        color: #888 !important;
        -webkit-text-fill-color: #888 !important;
    }
    
    /* ══════════════════════════════════════════════════════════════════════
       🔴 TEXT AREAS & INPUT FIELDS - White background = Black text
       ══════════════════════════════════════════════════════════════════════ */
    
    /* Text area - the actual input box */
    .stTextArea textarea,
    [data-testid="stTextArea"] textarea,
    textarea {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        background-color: #ffffff !important;
    }
    
    /* Text input fields */
    .stTextInput input,
    [data-testid="stTextInput"] input,
    input[type="text"],
    input[type="number"],
    input[type="password"] {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        background-color: #ffffff !important;
    }
    
    /* Number input */
    .stNumberInput input,
    [data-testid="stNumberInput"] input {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        background-color: #ffffff !important;
    }
    
    /* Selectbox - the displayed value box */
    .stSelectbox [data-baseweb="select"] > div,
    [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background-color: #1a1a2e !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    
    /* ══════════════════════════════════════════════════════════════════════
       CAPTIONS - dimmer white
       ══════════════════════════════════════════════════════════════════════ */
    .main .stCaption, .main caption, .main small,
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {
        color: #aaaaaa !important;
        -webkit-text-fill-color: #aaaaaa !important;
    }
    
    /* ══════════════════════════════════════════════════════════════════════
       DIVIDERS/SEPARATORS
       ══════════════════════════════════════════════════════════════════════ */
    .main hr {
        border-color: #333 !important;
    }
    
    /* ══════════════════════════════════════════════════════════════════════
       COLUMNS - ensure dark bg
       ══════════════════════════════════════════════════════════════════════ */
    [data-testid="column"] {
        background-color: transparent !important;
    }
    
    /* ══════════════════════════════════════════════════════════════════════
       PLOTLY CHARTS - ensure proper colors
       ══════════════════════════════════════════════════════════════════════ */
    .js-plotly-plot, .plotly {
        background-color: transparent !important;
    }
    
    /* ══════════════════════════════════════════════════════════════════════
       🔴 FINAL OVERRIDE: ALL BUTTONS - Nuclear Option
       ══════════════════════════════════════════════════════════════════════ */
    
    .stButton button,
    .stButton > button,
    div.stButton > button,
    [data-testid="stButton"] button,
    [data-testid="baseButton-secondary"],
    [data-testid="baseButton-primary"],
    [data-testid="baseButton-minimal"],
    button[data-testid],
    button[kind],
    [data-baseweb="button"],
    .main button {
        background-color: #1e2130 !important;
        background: #1e2130 !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        border: 1px solid #444 !important;
    }
    
    .stButton button *,
    .stButton > button *,
    div.stButton > button *,
    [data-baseweb="button"] *,
    [data-baseweb="button"] span,
    [data-baseweb="button"] p,
    button span,
    button p,
    button div {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        background-color: transparent !important;
        background: transparent !important;
    }
    
    button[kind="primary"],
    [data-testid="baseButton-primary"],
    .stButton button[kind="primary"] {
        background-color: #00d4ff !important;
        background: #00d4ff !important;
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }
    
    button[kind="primary"] *,
    [data-testid="baseButton-primary"] * {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }
    
    /* ══════════════════════════════════════════════════════════════════════
       🔴 CODE BLOCKS = BLACK TEXT ON LIGHT BACKGROUND (NUCLEAR)
       ══════════════════════════════════════════════════════════════════════ */
    
    /* st.code() - FORCE BLACK TEXT */
    .stCode,
    .stCode *,
    .stCode pre,
    .stCode code,
    .stCode span,
    .stCodeBlock,
    .stCodeBlock *,
    .stCodeBlock pre,
    .stCodeBlock code,
    .stCodeBlock span,
    [data-testid="stCode"],
    [data-testid="stCode"] *,
    [data-testid="stCodeBlock"],
    [data-testid="stCodeBlock"] *,
    [data-testid="stCodeBlock"] pre,
    [data-testid="stCodeBlock"] code,
    [data-testid="stCodeBlock"] span,
    .main .stCode *,
    .main .stCodeBlock *,
    .main [data-testid="stCode"] *,
    .main [data-testid="stCodeBlock"] *,
    .element-container .stCode *,
    .element-container .stCodeBlock *,
    pre[class*="language-"],
    pre[class*="language-"] *,
    code[class*="language-"],
    code[class*="language-"] * {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        background-color: #f5f5f5 !important;
    }
    
    /* Inline code in markdown - cyan on dark */
    .main code:not(.stCodeBlock code):not(.stCode code):not([data-testid="stCodeBlock"] code),
    [data-testid="stMarkdownContainer"] > code,
    p code, span code, li code {
        color: #00d4ff !important;
        -webkit-text-fill-color: #00d4ff !important;
        background-color: #1a1a2e !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
    }
    
    /* Code blocks inside expanders - also black */
    [data-testid="stExpander"] .stCode *,
    [data-testid="stExpander"] .stCodeBlock *,
    [data-testid="stExpander"] [data-testid="stCodeBlock"] *,
    [data-testid="stExpanderDetails"] .stCode *,
    [data-testid="stExpanderDetails"] .stCodeBlock * {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }
    
    /* ══════════════════════════════════════════════════════════════════════
       🔴 UNIVERSAL CONTRAST LAW
       ══════════════════════════════════════════════════════════════════════ */
    
    *[style*="background-color: white"],
    *[style*="background-color: #fff"],
    *[style*="background-color: rgb(255"],
    *[style*="background: white"],
    *[style*="background: #fff"],
    *[style*="background: rgb(255"] {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }
    
    *[style*="background-color: white"] *,
    *[style*="background-color: #fff"] *,
    *[style*="background-color: rgb(255"] *,
    *[style*="background: white"] *,
    *[style*="background: #fff"] *,
    *[style*="background: rgb(255"] * {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }
    
</style>

<script>
// 🔴 NUCLEAR OPTION: Force button styling via JavaScript
function fixButtonContrast() {
    const buttons = document.querySelectorAll('button');
    buttons.forEach(btn => {
        const style = window.getComputedStyle(btn);
        const bgColor = style.backgroundColor;
        const rgb = bgColor.match(/[0-9]+/g);
        if (rgb) {
            const brightness = (parseInt(rgb[0]) * 299 + parseInt(rgb[1]) * 587 + parseInt(rgb[2]) * 114) / 1000;
            if (brightness > 128) {
                btn.style.setProperty('color', '#000000', 'important');
                btn.style.setProperty('-webkit-text-fill-color', '#000000', 'important');
                btn.querySelectorAll('*').forEach(child => {
                    child.style.setProperty('color', '#000000', 'important');
                    child.style.setProperty('-webkit-text-fill-color', '#000000', 'important');
                });
            } else {
                btn.style.setProperty('color', '#ffffff', 'important');
                btn.style.setProperty('-webkit-text-fill-color', '#ffffff', 'important');
                btn.querySelectorAll('*').forEach(child => {
                    child.style.setProperty('color', '#ffffff', 'important');
                    child.style.setProperty('-webkit-text-fill-color', '#ffffff', 'important');
                });
            }
        }
    });
}

// 🔴 FORCE CODE BLOCKS TO HAVE BLACK TEXT
function fixCodeBlocks() {
    // Target all code block elements
    const codeElements = document.querySelectorAll('.stCode, .stCodeBlock, [data-testid="stCode"], [data-testid="stCodeBlock"], pre, code');
    codeElements.forEach(el => {
        const style = window.getComputedStyle(el);
        const bgColor = style.backgroundColor;
        const rgb = bgColor.match(/[0-9]+/g);
        if (rgb) {
            const brightness = (parseInt(rgb[0]) * 299 + parseInt(rgb[1]) * 587 + parseInt(rgb[2]) * 114) / 1000;
            // If background is light (brightness > 128), use black text
            if (brightness > 128) {
                el.style.setProperty('color', '#000000', 'important');
                el.style.setProperty('-webkit-text-fill-color', '#000000', 'important');
                el.querySelectorAll('*').forEach(child => {
                    child.style.setProperty('color', '#000000', 'important');
                    child.style.setProperty('-webkit-text-fill-color', '#000000', 'important');
                });
            }
        }
    });
}

document.addEventListener('DOMContentLoaded', fixButtonContrast);
document.addEventListener('DOMContentLoaded', fixCodeBlocks);
setInterval(fixButtonContrast, 1000);
setInterval(fixCodeBlocks, 1000);
</script>
""", unsafe_allow_html=True)