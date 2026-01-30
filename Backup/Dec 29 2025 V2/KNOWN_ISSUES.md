# InvestorIQ - Known Issues & Fixes Log

## LAST UPDATED: December 26, 2025

---

## ⚠️ COMMON ISSUES TO AVOID

### 1. Raw HTML Showing Instead of Rendering
**Symptom:** `<div style='background:...'>` showing as text  
**Causes:** 
- Mixed markdown/HTML content with newlines (`\n`) and markdown bold (`**text**`)
- Unescaped HTML entities (`<`, `>`, `&`) in content breaking HTML parsing
- Multi-line f-strings with indentation causing whitespace issues

**Fix:** 
```python
import html
import re

# 1. Escape HTML entities FIRST
safe_content = html.escape(str(content))
# 2. Then convert newlines to <br>
safe_content = safe_content.replace('\n', '<br>')
# 3. Then convert markdown bold
safe_content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', safe_content)
# 4. Build HTML in single line (no indentation)
html = f"<div style='...'>{safe_content}</div>"
```

### 2. Streamlit HTML Not Rendering  
**Symptom:** HTML displays as plain text  
**Cause:** Missing `unsafe_allow_html=True`  
**Fix:** Always use:
```python
st.markdown(html_content, unsafe_allow_html=True)
```

### 3. CSS Backticks/Code Blocks in Dark Mode
**Symptom:** Code appears as white on white background  
**Cause:** Using backticks in markdown which creates code blocks  
**Fix:** NEVER use backticks in CSS styling - use colored divs instead

---

## 🔧 COINGLASS API V4 ISSUES

### 4. API 404 Errors
**Symptom:** `{"code":"404","msg":"Not Found"}`  
**Cause:** Multiple issues:
- Wrong parameter names (`symbol` vs `instrument`)
- Missing required parameters (`exchange`)
- Wrong symbol format (`BTC` vs `BTCUSDT`)

**CORRECT V4 API Format:**
```python
# ALL endpoints need:
params = {
    "exchange": "Binance",       # Required!
    "symbol": "BTCUSDT",         # Full pair name, not just BTC!
    "interval": "4h",            # 4h, 12h, 1d (Hobbyist: >=4h only)
    "limit": 500
}

# Endpoints (kebab-case):
/api/futures/open-interest/history
/api/futures/funding-rate/history
/api/futures/top-long-short-account-ratio/history
/api/futures/global-long-short-account-ratio/history
```

### 5. Hobbyist Tier Interval Limit
**Symptom:** API error or empty data  
**Cause:** Using interval < 4h on Hobbyist plan  
**Fix:** Enforce minimum interval:
```python
if interval in ['1h', '15m', '5m', '1m']:
    interval = '4h'  # Hobbyist tier minimum
```

### 6. Rate Limiting (30 req/min)
**Symptom:** 429 errors or blank responses  
**Fix:** Add 2.5s delay between requests:
```python
time.sleep(2.5)  # Safe for 30 req/min limit
```

---

## 🗄️ DATABASE ISSUES

### 7. "No column named data_source"
**Symptom:** `table whale_snapshots has no column named data_source`  
**Cause:** Old database schema missing new column  
**Fix:** Add migration in whale_data_store.py:
```python
def _init_db(self):
    # Check if column exists
    cursor.execute("PRAGMA table_info(whale_snapshots)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if 'data_source' not in columns:
        cursor.execute("ALTER TABLE whale_snapshots ADD COLUMN data_source TEXT DEFAULT 'live'")
```

---

## 🎨 UI/STYLING ISSUES

### 8. Dark Text on Dark Background (Sidebar)
**Symptom:** Text invisible in light sidebar mode  
**Cause:** Using dark backgrounds (#1a1a2e) with dark text  
**Fix for light sidebar:**
```python
# Use LIGHT backgrounds with dark text:
background: #f0f9f0  # Light green
color: #2e7d32       # Dark green text
```

### 8b. Grey Text on Dark Background (Main Area)
**Symptom:** Text barely visible on dark main area  
**Cause:** Using grey (#666, #888) text on dark backgrounds  
**Fix:** ALWAYS use white (#fff) or light colors on dark backgrounds:
```python
# WRONG:
color: #666  # Grey on dark = invisible

# CORRECT:
color: #fff  # White on dark = visible
color: #ccc  # Light grey = also OK
```

### 9. Emoji/Unicode Breaking Layout
**Symptom:** Layout shifts or breaks  
**Cause:** Emoji characters in certain contexts  
**Fix:** Test with and without emojis, use HTML entities if needed

---

## 📊 DATA ISSUES

### 10. Scanner Finds Coins Not in Historical DB
**Symptom:** "Insufficient data" in Historical Validation  
**Cause:** Only 20 coins imported, scanner finds others  
**Fix:** Import 300+ coins using bulk_import.py:
```bash
python bulk_import.py --api-key YOUR_KEY --days 365
```

### 11. Historical Data Location
**Location:** `TScanner/data/whale_history.db`  
**Note:** This persists! Only import once.

---

## 🔑 API KEYS

### Coinglass V4 API
- **Base URL:** `https://open-api-v4.coinglass.com`
- **Header:** `CG-API-KEY: your_key`
- **Plan limits:** Hobbyist = interval >=4h, 30 req/min

### Quiver API (Stocks)
- **Base URL:** `https://api.quiverquant.com/beta/`
- **Header:** `Authorization: Bearer your_key`

---

## ✅ PRE-DEPLOYMENT CHECKLIST

1. [ ] All `st.markdown()` calls with HTML have `unsafe_allow_html=True`
2. [ ] No backticks used for code display in dark mode
3. [ ] Database migration runs automatically
4. [ ] **Stories content escaped with `html.escape()` BEFORE HTML rendering**
5. [ ] **HTML built in single lines (no multiline f-strings with indentation)**
6. [ ] API endpoints use correct V4 format with full symbol
7. [ ] Rate limiting enforced (2.5s delay)
8. [ ] Error messages don't show raw HTML
9. [ ] Clear `__pycache__` folders when deploying updates

---

## 📍 PLACES THAT RENDER STORIES HTML

These files/locations iterate over stories and render HTML. Ensure all use proper escaping:

1. **`utils/formatters.py`** - `render_combined_learning_html()` function
2. **`app.py`** line ~3919 - Scanner popup detail view

Both must:
- Use `html.escape(content)` before inserting into HTML
- Use single-line HTML (no multiline f-strings)
- Convert `\n` to `<br>` and `**text**` to `<strong>text</strong>`

---

## 📝 VERSION HISTORY

- **v14:** Fixed Coinglass API (BTCUSDT format)
- **v15:** Added bulk_import.py for 300+ coins
- **v16:** Fixed raw HTML in stories (newline/markdown escaping)
