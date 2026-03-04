
import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import concurrent.futures
import xml.etree.ElementTree as ET
from urllib.parse import urlparse, urljoin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
from datetime import datetime

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LinkBuilder Pro",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── DUAL-THEME CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
:root {
    --lb-primary:         #1a5276;
    --lb-primary-light:   #2e86c1;
    --lb-success:         #1e8449;
    --lb-bg-card:         #f8f9fa;
    --lb-border:          #dee2e6;
    --lb-border-left:     #1a5276;
    --lb-text:            #212529;
    --lb-text-muted:      #6c757d;
    --lb-pill-pending-bg: #f0f0f0;
    --lb-pill-pending-fg: #888888;
    --lb-header-grad1:    #0f2942;
    --lb-header-grad2:    #1a5276;
    --lb-step-pending-bg: #ecf0f1;
    --lb-step-pending-fg: #95a5a6;
    --lb-caption-bg:      #fff8e1;
    --lb-caption-border:  #f0c040;
    --lb-caption-fg:      #5d4037;
    --lb-strategy-bg:     #eaf4fb;
    --lb-strategy-border: #2e86c1;
    --lb-strategy-fg:     #1a5276;
    --lb-strategy-bg2:    #eafaf1;
    --lb-strategy-border2:#27ae60;
    --lb-strategy-fg2:    #1e8449;
    --lb-cache-bg:        #f0fff4;
    --lb-cache-border:    #27ae60;
    --lb-cache-fg:        #1e8449;
}
[data-theme="dark"], .stApp[data-theme="dark"] {
    --lb-primary:         #2e86c1;
    --lb-primary-light:   #5dade2;
    --lb-success:         #27ae60;
    --lb-bg-card:         #1e2530;
    --lb-border:          #2e3b4e;
    --lb-border-left:     #2e86c1;
    --lb-text:            #e8edf2;
    --lb-text-muted:      #8fa3b1;
    --lb-pill-pending-bg: #2a3441;
    --lb-pill-pending-fg: #7f8c99;
    --lb-header-grad1:    #0a1929;
    --lb-header-grad2:    #0d3556;
    --lb-step-pending-bg: #2a3441;
    --lb-step-pending-fg: #7f8c99;
    --lb-caption-bg:      #2a2000;
    --lb-caption-border:  #7d6608;
    --lb-caption-fg:      #f5cba7;
    --lb-strategy-bg:     #0d2137;
    --lb-strategy-border: #2e86c1;
    --lb-strategy-fg:     #5dade2;
    --lb-strategy-bg2:    #0d2318;
    --lb-strategy-border2:#27ae60;
    --lb-strategy-fg2:    #58d68d;
    --lb-cache-bg:        #0d2318;
    --lb-cache-border:    #27ae60;
    --lb-cache-fg:        #58d68d;
}
@media (prefers-color-scheme: dark) {
    :root {
        --lb-primary:#2e86c1; --lb-primary-light:#5dade2; --lb-success:#27ae60;
        --lb-bg-card:#1e2530; --lb-border:#2e3b4e; --lb-border-left:#2e86c1;
        --lb-text:#e8edf2; --lb-text-muted:#8fa3b1;
        --lb-pill-pending-bg:#2a3441; --lb-pill-pending-fg:#7f8c99;
        --lb-header-grad1:#0a1929; --lb-header-grad2:#0d3556;
        --lb-step-pending-bg:#2a3441; --lb-step-pending-fg:#7f8c99;
        --lb-caption-bg:#2a2000; --lb-caption-border:#7d6608; --lb-caption-fg:#f5cba7;
        --lb-strategy-bg:#0d2137; --lb-strategy-border:#2e86c1; --lb-strategy-fg:#5dade2;
        --lb-strategy-bg2:#0d2318; --lb-strategy-border2:#27ae60; --lb-strategy-fg2:#58d68d;
        --lb-cache-bg:#0d2318; --lb-cache-border:#27ae60; --lb-cache-fg:#58d68d;
    }
}
.lb-header {
    background: linear-gradient(135deg, var(--lb-header-grad1) 0%, var(--lb-header-grad2) 100%);
    padding:2rem 2.5rem; border-radius:14px; color:#fff; margin-bottom:1.8rem;
    border:1px solid rgba(255,255,255,0.08); box-shadow:0 4px 20px rgba(0,0,0,0.25);
}
.lb-header h1 { margin:0 0 0.3rem 0; font-size:2rem; color:#fff; }
.lb-header p  { margin:0; opacity:0.85; font-size:0.95rem; color:#cfe2f3; }
.step-pill { padding:0.45rem 1rem; border-radius:20px; text-align:center;
             font-size:0.82rem; font-weight:600; margin-bottom:0.35rem; transition:all 0.2s; }
.pill-done    { background:var(--lb-success); color:#fff; }
.pill-active  { background:var(--lb-primary); color:#fff;
                border:2px solid var(--lb-primary-light); box-shadow:0 0 8px rgba(46,134,193,0.4); }
.pill-pending { background:var(--lb-pill-pending-bg); color:var(--lb-pill-pending-fg); }
.step-bar-done    { background:var(--lb-success); color:#fff; font-weight:600;
                    padding:0.5rem 0.2rem; border-radius:8px; text-align:center; font-size:0.8rem; }
.step-bar-active  { background:var(--lb-primary); color:#fff; font-weight:700;
                    padding:0.5rem 0.2rem; border-radius:8px; text-align:center; font-size:0.8rem;
                    box-shadow:0 0 10px rgba(46,134,193,0.5); border:2px solid var(--lb-primary-light); }
.step-bar-pending { background:var(--lb-step-pending-bg); color:var(--lb-step-pending-fg);
                    font-weight:400; padding:0.5rem 0.2rem; border-radius:8px;
                    text-align:center; font-size:0.8rem; }
.info-card { background:var(--lb-bg-card); border:1px solid var(--lb-border);
             border-left:5px solid var(--lb-border-left); border-radius:8px;
             padding:1rem 1.2rem; margin-bottom:1rem; color:var(--lb-text); line-height:1.6; }
.info-card b    { color:var(--lb-primary-light); }
.info-card code { background:rgba(46,134,193,0.15); color:var(--lb-primary-light);
                  padding:0.1rem 0.35rem; border-radius:4px; font-size:0.85rem; }
.strategy-card-domain { background:var(--lb-strategy-bg); border:2px solid var(--lb-strategy-border);
    border-radius:10px; padding:1.1rem 1.3rem; color:var(--lb-strategy-fg); margin-bottom:0.5rem; line-height:1.65; }
.strategy-card-page { background:var(--lb-strategy-bg2); border:2px solid var(--lb-strategy-border2);
    border-radius:10px; padding:1.1rem 1.3rem; color:var(--lb-strategy-fg2); margin-bottom:0.5rem; line-height:1.65; }
.strategy-badge { display:inline-block; padding:0.2rem 0.65rem; border-radius:12px;
                  font-size:0.75rem; font-weight:700; margin-bottom:0.4rem; }
.badge-domain { background:var(--lb-strategy-border);  color:#fff; }
.badge-page   { background:var(--lb-strategy-border2); color:#fff; }
.score-caption { background:var(--lb-caption-bg); border:1px solid var(--lb-caption-border);
                 border-radius:6px; padding:0.5rem 0.9rem; color:var(--lb-caption-fg);
                 font-size:0.82rem; margin:0.4rem 0; }
.cache-badge { background:var(--lb-cache-bg); border:1px solid var(--lb-cache-border);
               border-radius:6px; padding:0.4rem 0.8rem; color:var(--lb-cache-fg);
               font-size:0.8rem; display:inline-block; margin:0.2rem 0; }
div[data-testid="stDataFrame"] { border-radius:8px; }
.stDownloadButton > button { width:100%; }
div[data-testid="stProgressBar"] > div > div { background-color:var(--lb-primary-light) !important; }
div[data-testid="stMetricLabel"] > div        { color:var(--lb-text-muted) !important; }
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ────────────────────────────────────────────────────────────
DEFAULTS = {
    "step":           1,
    "keywords":       [],
    "domains":        [],
    "crawl_data":     {},
    "matches_df":     None,
    "enriched_df":    None,
    "tfidf_strategy": "Page-Level",
    # ── Ahrefs caches (persist across reruns in same session) ──────────────
    "ahrefs_domain_cache": {},   # {domain: {"dr": v, "dt": v}}
    "ahrefs_page_cache":   {},   # {page_url: traffic_int}
    "cfg": {
        "max_depth": 2,
        "max_pages": 50,
        "workers":   10,
        "threshold": 0.3,
        "exclude":   ["/tag/", "/category/", "/author/", "/page/", "/wp-json/"],
    }
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

cfg = st.session_state.cfg

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def to_base_url(domain: str) -> str:
    d = domain.strip()
    if not d.startswith(("http://", "https://")):
        d = "https://" + d
    p = urlparse(d)
    return f"{p.scheme}://{p.netloc}"

def extract_domain(url: str) -> str:
    return urlparse(url).netloc.replace("www.", "").lower()

def should_exclude(url: str) -> bool:
    return any(pat in url for pat in cfg["exclude"])

def clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    return " ".join(soup.get_text(separator=" ").split())[:50000]

CRAWL_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; LinkBuilderBot/1.0)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

def safe_get(url: str, timeout: int = 10) -> tuple:
    try:
        r = requests.get(url, headers=CRAWL_HEADERS, timeout=timeout, allow_redirects=True)
        return r.status_code, r.text
    except Exception:
        return 0, ""

# ─── SITEMAP ──────────────────────────────────────────────────────────────────
def parse_sitemap_xml(content: str, visited: set, max_pages: int) -> list:
    urls = []
    try:
        root = ET.fromstring(content)
        ns   = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        for st_tag in root.findall("sm:sitemap", ns):
            loc = st_tag.find("sm:loc", ns)
            if loc is not None and loc.text:
                sub = loc.text.strip()
                if sub not in visited and len(urls) < max_pages:
                    visited.add(sub)
                    sc, sc2 = safe_get(sub)
                    if sc == 200:
                        urls.extend(parse_sitemap_xml(sc2, visited, max_pages))
        for ue in root.findall("sm:url", ns):
            loc = ue.find("sm:loc", ns)
            if loc is not None and loc.text:
                urls.append(loc.text.strip())
    except Exception:
        pass
    return list(dict.fromkeys(urls))[:max_pages]

def get_sitemap_pages(base_url: str, max_pages: int) -> list:
    for path in ["/sitemap.xml", "/sitemap_index.xml", "/sitemap/", "/sitemap1.xml"]:
        url = base_url.rstrip("/") + path
        status, content = safe_get(url)
        if status == 200 and ("<urlset" in content or "<sitemapindex" in content):
            pages = parse_sitemap_xml(content, {url}, max_pages)
            if pages:
                return [p for p in pages if not should_exclude(p)][:max_pages]
    return []

def shallow_crawl(base_url: str, max_depth: int, max_pages: int) -> list:
    visited, queue, found = set(), [(base_url, 0)], [base_url]
    base_netloc = urlparse(base_url).netloc
    while queue and len(found) < max_pages:
        cur, depth = queue.pop(0)
        if cur in visited or depth > max_depth:
            continue
        visited.add(cur)
        status, content = safe_get(cur)
        if status != 200 or not content:
            continue
        soup = BeautifulSoup(content, "html.parser")
        for a in soup.find_all("a", href=True):
            full = urljoin(base_url, str(a["href"])).split("#")[0].split("?")[0]
            if urlparse(full).netloc != base_netloc:
                continue
            if should_exclude(full) or full in visited:
                continue
            if full not in found:
                found.append(full)
            if depth + 1 <= max_depth and (full, depth + 1) not in queue:
                queue.append((full, depth + 1))
    return found[:max_pages]

def crawl_domain_worker(domain: str) -> dict:
    base_url = to_base_url(domain)
    pages    = get_sitemap_pages(base_url, cfg["max_pages"])
    if pages:
        return {"domain": domain, "base_url": base_url, "pages": pages, "method": "sitemap"}
    pages = shallow_crawl(base_url, cfg["max_depth"], cfg["max_pages"])
    return {"domain": domain, "base_url": base_url, "pages": pages, "method": "crawl"}

def fetch_page_text(url: str) -> tuple:
    status, html = safe_get(url, timeout=12)
    if status == 200 and html:
        return url, clean_text(html)
    return url, ""

# ─── AHREFS API ───────────────────────────────────────────────────────────────
AHREFS_BASE = "https://api.ahrefs.com/v3/site-explorer"
TODAY       = datetime.today().strftime("%Y-%m-%d")

def _ahrefs_get(endpoint: str, params: dict, token: str):
    try:
        r = requests.get(
            f"{AHREFS_BASE}/{endpoint}",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            params=params,
            timeout=15,
        )
        if r.status_code == 200:
            return r.json()
        st.toast(f"Ahrefs {endpoint} [{r.status_code}]: {r.text[:100]}", icon="⚠️")
    except Exception as e:
        st.toast(f"Request error: {e}", icon="❌")
    return None

def get_domain_rating(domain: str, token: str):
    d = _ahrefs_get("domain-rating", {
        "target": domain, "date": TODAY,
        "mode": "subdomains", "protocol": "both",
    }, token)
    return d.get("domain_rating", {}).get("domain_rating") if d else None

def get_domain_traffic(domain: str, token: str):
    d = _ahrefs_get("metrics", {
        "target": domain, "date": TODAY,
        "mode": "subdomains", "protocol": "both",
    }, token)
    return d.get("metrics", {}).get("org_traffic") if d else None

def _normalise_url(url: str) -> str:
    """Ensure scheme present; strip trailing slash on non-root paths."""
    u = url.strip()
    if not u.startswith(("http://", "https://")):
        u = "https://" + u
    parsed = urlparse(u)
    # Only strip trailing slash when there is a real path beyond root
    if parsed.path not in ("", "/"):
        u = u.rstrip("/")
    return u

def get_page_traffic(page_url: str, token: str) -> int:
    """
    Fetch page-level organic traffic with two-pass strategy:
      Pass 1 — exact mode  (specific URL match)
      Pass 2 — prefix mode (catches trailing-slash / query-param variants)
    Returns best non-zero result, or 0 if both passes return nothing.
    """
    url = _normalise_url(page_url)

    # Pass 1: exact
    d1 = _ahrefs_get("metrics", {
        "target": url, "date": TODAY,
        "mode": "exact", "protocol": "both",
    }, token)
    t1 = (d1.get("metrics", {}).get("org_traffic") or 0) if d1 else 0
    if t1 > 0:
        return t1

    # Pass 2: prefix fallback
    d2 = _ahrefs_get("metrics", {
        "target": url, "date": TODAY,
        "mode": "prefix", "protocol": "both",
    }, token)
    t2 = (d2.get("metrics", {}).get("org_traffic") or 0) if d2 else 0
    return t2  # returns 0 if both failed — explicit, not None

# ─── RENDER HELPERS ───────────────────────────────────────────────────────────
def render_step_bar():
    labels = ["📋 Keywords", "🌐 Domains", "🕷️ Crawl", "🧮 TF-IDF", "📊 Ahrefs"]
    cols   = st.columns(5)
    for i, (col, label) in enumerate(zip(cols, labels), 1):
        cur = st.session_state.step
        cls = "step-bar-done" if i < cur else ("step-bar-active" if i == cur else "step-bar-pending")
        with col:
            st.markdown(f'''<div class="{cls}">{label}</div>''', unsafe_allow_html=True)

def info_card(html: str):
    st.markdown(f'''<div class="info-card">{html}</div>''', unsafe_allow_html=True)

def score_caption(text: str):
    st.markdown(f'''<div class="score-caption">{text}</div>''', unsafe_allow_html=True)

def cache_badge(text: str):
    st.markdown(f'''<div class="cache-badge">{text}</div>''', unsafe_allow_html=True)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔗 LinkBuilder Pro")
    st.markdown("---")
    step_names = ["📋 Keywords", "🌐 Domains", "🕷️ Crawl", "🧮 TF-IDF", "📊 Ahrefs"]
    for i, name in enumerate(step_names, 1):
        if i < st.session_state.step:
            cls, icon = "pill-done",    "✅"
        elif i == st.session_state.step:
            cls, icon = "pill-active",  "▶"
        else:
            cls, icon = "pill-pending", str(i)
        st.markdown(
            f'''<div class="step-pill {cls}">{icon} Step {i}: {name}</div>''',
            unsafe_allow_html=True
        )
    st.markdown("---")
    st.markdown("### ⚙️ Configuration")
    cfg["max_depth"] = st.slider("Crawl Depth",          1,    3,    cfg["max_depth"])
    cfg["max_pages"] = st.slider("Max Pages / Domain",   10,   100,  cfg["max_pages"],  step=10)
    cfg["workers"]   = st.slider("Parallel Workers",     5,    30,   cfg["workers"],    step=5)
    cfg["threshold"] = st.slider("Similarity Threshold", 0.05, 0.95, cfg["threshold"],  step=0.05)
    st.markdown("---")

    # ── Cache stats in sidebar ───────────────────────────────────────────
    dc = len(st.session_state.ahrefs_domain_cache)
    pc = len(st.session_state.ahrefs_page_cache)
    if dc > 0 or pc > 0:
        st.markdown("### 💾 Ahrefs Cache")
        st.markdown(f"Domains cached: **{dc}**")
        st.markdown(f"Pages cached:   **{pc}**")
        if st.button("🗑️ Clear Ahrefs Cache", use_container_width=True):
            st.session_state.ahrefs_domain_cache = {}
            st.session_state.ahrefs_page_cache   = {}
            st.rerun()
        st.markdown("---")

    if st.button("🔄 Reset All", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="lb-header">
    <h1>🔗 LinkBuilder Pro</h1>
    <p>Automated Link Opportunity Discovery — Crawl · Analyze · Match · Enrich</p>
</div>
""", unsafe_allow_html=True)

render_step_bar()
st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — UPLOAD KEYWORDS
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.step == 1:
    st.markdown("## Step 1 · Upload Keywords List")
    info_card(
        "Upload target keywords for link opportunity discovery. "
        "Supports <b>.txt</b> (one per line) or <b>.csv</b> (with a <code>keyword</code> column)."
    )
    col1, col2 = st.columns([3, 2])
    with col1:
        kw_file = st.file_uploader("Upload Keywords File (.csv / .txt)", type=["csv", "txt"])
        manual  = st.text_area("Or paste keywords (one per line)", height=200,
                               placeholder="online casino\nbest poker sites\nblackjack strategy")
    with col2:
        st.markdown("#### 📌 Format Examples")
        st.code("# TXT\nonline casino\nbest poker sites", language="text")
        st.code("# CSV\nkeyword\nonline casino\nbest poker sites", language="text")

    if st.button("▶ Load Keywords & Continue", type="primary", use_container_width=True):
        kws = []
        if kw_file:
            if kw_file.name.endswith(".txt"):
                kws += [k.strip() for k in kw_file.read().decode("utf-8").splitlines() if k.strip()]
            else:
                df_kw    = pd.read_csv(kw_file)
                col_name = next((c for c in df_kw.columns if "keyword" in c.lower()), df_kw.columns[0])
                kws     += df_kw[col_name].dropna().astype(str).str.strip().tolist()
        if manual.strip():
            kws += [k.strip() for k in manual.splitlines() if k.strip()]
        kws = list(dict.fromkeys(kws))
        if kws:
            st.session_state.keywords = kws
            st.session_state.step     = 2
            st.success(f"✅ {len(kws)} keywords loaded!")
            st.rerun()
        else:
            st.error("❌ No keywords found — upload a file or paste keywords manually.")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — UPLOAD DOMAIN LIST
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 2:
    st.markdown("## Step 2 · Upload Domain List")
    kw_count = len(st.session_state.keywords)
    st.success(
        f"✅ {kw_count} keywords loaded — "
        f"{', '.join(st.session_state.keywords[:6])}{'...' if kw_count > 6 else ''}"
    )
    info_card(
        "Upload domains to crawl for link opportunities. "
        "Supports <b>.csv</b> (with a <code>domain</code> column) or plain <b>.txt</b>. "
        "Domains can include or omit <code>https://</code>."
    )
    col1, col2 = st.columns([3, 2])
    with col1:
        dom_file = st.file_uploader("Upload Domain File (.csv / .txt)", type=["csv", "txt"])
        manual_d = st.text_area("Or paste domains (one per line)", height=200,
                                placeholder="example.com\nanotherdomain.com")
    with col2:
        st.markdown("#### 📌 Format Examples")
        st.code("# CSV\ndomain\nexample.com\nanotherdomain.com", language="text")
        st.code("# TXT\nexample.com\nanotherdomain.com", language="text")

    col_back, col_next = st.columns([1, 4])
    with col_back:
        if st.button("← Back"):
            st.session_state.step = 1; st.rerun()
    with col_next:
        if st.button("▶ Load Domains & Continue", type="primary", use_container_width=True):
            doms = []
            if dom_file:
                if dom_file.name.endswith(".txt"):
                    doms += [d.strip() for d in dom_file.read().decode("utf-8").splitlines() if d.strip()]
                else:
                    df_d     = pd.read_csv(dom_file)
                    col_name = next((c for c in df_d.columns if "domain" in c.lower()), df_d.columns[0])
                    doms    += df_d[col_name].dropna().astype(str).str.strip().tolist()
            if manual_d.strip():
                doms += [d.strip() for d in manual_d.splitlines() if d.strip()]
            doms = list(dict.fromkeys(doms))
            if doms:
                st.session_state.domains = doms
                st.session_state.step    = 3
                st.success(f"✅ {len(doms)} domains loaded!")
                st.rerun()
            else:
                st.error("❌ No domains found — upload a file or paste domains manually.")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — DOMAIN CRAWLING & PAGE EXPANSION
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 3:
    st.markdown("## Step 3 · Domain Crawling & Page Expansion")
    domains       = st.session_state.domains
    total_domains = len(domains)

    c1, c2, c3 = st.columns(3)
    c1.metric("Keywords",         len(st.session_state.keywords))
    c2.metric("Domains to Crawl", total_domains)
    c3.metric("Domains Loaded",   total_domains)

    st.markdown("---")
    col_a, col_b, col_c = st.columns(3)
    col_a.info(f"🔍 Crawl Depth: **{cfg['max_depth']}** levels")
    col_b.info(f"📄 Max Pages: **{cfg['max_pages']}** per domain")
    col_c.info(f"⚡ Workers: **{cfg['workers']}** threads")

    info_card(
        "<b>Crawling Strategy:</b><br>"
        "🗺️ <b>Sitemap-first</b> — checks <code>/sitemap.xml</code> and <code>/sitemap_index.xml</code><br>"
        "🔍 <b>Fallback</b> — shallow BFS crawl if no sitemap found<br>"
        "🚫 <b>Excluded:</b> <code>/tag/</code> · <code>/category/</code> · "
        "<code>/author/</code> · <code>/page/</code> · <code>/wp-json/</code>"
    )

    col_back, col_start = st.columns([1, 4])
    with col_back:
        if st.button("← Back"):
            st.session_state.step = 2; st.rerun()
    with col_start:
        start_crawl = st.button("🕷️ Start Crawling", type="primary", use_container_width=True)

    if st.session_state.crawl_data and not start_crawl:
        prev      = st.session_state.crawl_data
        total_pgs = sum(len(v["pages"]) for v in prev.values())
        st.success(f"✅ Previous crawl: **{len(prev)} domains** | **{total_pgs} pages**")
        prev_df = pd.DataFrame([
            {"Domain": d, "Pages Found": len(v["pages"]), "Method": v["method"]}
            for d, v in prev.items()
        ])
        st.dataframe(prev_df, use_container_width=True, height=300)
        if st.button("▶ Proceed to TF-IDF Matching", type="primary", use_container_width=True):
            st.session_state.step = 4; st.rerun()

    if start_crawl:
        pb        = st.progress(0, text="🕷️ Initializing crawl...")
        stat_box  = st.empty()
        results   = {}
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg["workers"]) as executor:
            future_map = {executor.submit(crawl_domain_worker, d): d for d in domains}
            for future in concurrent.futures.as_completed(future_map):
                try:
                    res = future.result(timeout=30)
                    results[res["domain"]] = res
                except Exception:
                    dk = future_map[future]
                    results[dk] = {"domain": dk, "pages": [], "method": "error"}
                completed += 1
                tp = sum(len(v["pages"]) for v in results.values())
                sn = sum(1 for v in results.values() if v.get("method") == "sitemap")
                cn = sum(1 for v in results.values() if v.get("method") == "crawl")
                pb.progress(completed / total_domains,
                            text=f"🕷️ {completed}/{total_domains} domains | {tp} pages")
                stat_box.markdown(
                    f"🗺️ Sitemap: **{sn}** &nbsp;·&nbsp; 🔍 Crawled: **{cn}** &nbsp;·&nbsp; 📄 Pages: **{tp}**"
                )
        st.session_state.crawl_data = results
        tp = sum(len(v["pages"]) for v in results.values())
        sn = sum(1 for v in results.values() if v.get("method") == "sitemap")
        pb.progress(1.0, text="✅ Crawling complete!")
        c1, c2, c3 = st.columns(3)
        c1.metric("Domains Crawled",   len(results))
        c2.metric("Total Pages Found", tp)
        c3.metric("Via Sitemap",       sn)
        summary_df = pd.DataFrame([
            {"Domain": d, "Pages Found": len(v["pages"]), "Method": v["method"]}
            for d, v in results.items()
        ])
        st.dataframe(summary_df, use_container_width=True, height=350)
        all_rows  = [{"domain": d, "page_url": p, "method": v["method"]}
                     for d, v in results.items() for p in v["pages"]]
        crawl_csv = pd.DataFrame(all_rows).to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Crawled Pages CSV", crawl_csv,
                           "crawled_pages.csv", "text/csv", use_container_width=True)
        if st.button("▶ Proceed to TF-IDF Matching", type="primary", use_container_width=True):
            st.session_state.step = 4; st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — TF-IDF VECTORIZATION & RELEVANCE MATCHING
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 4:
    st.markdown("## Step 4 · TF-IDF Vectorization & Relevance Matching")

    crawl_data   = st.session_state.crawl_data
    domain_count = len(st.session_state.domains)
    total_pages  = sum(len(v["pages"]) for v in crawl_data.values())

    c1, c2, c3 = st.columns(3)
    c1.metric("Domains",       domain_count)
    c2.metric("Crawled Pages", total_pages)
    c3.metric("Threshold",     cfg["threshold"])

    st.markdown("---")
    st.markdown("### 🎯 Select Matching Strategy")
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        st.markdown("""
        <div class="strategy-card-domain">
            <span class="strategy-badge badge-domain">🌐 Domain-Level</span><br>
            <b>Analyzes homepages only.</b><br>
            Fetches root URL per domain. Best for <b>large sets (1,000+)</b> where speed matters.
            <br><br>✅ Fast &nbsp;·&nbsp; ✅ Low cost &nbsp;·&nbsp; ⚠️ Less precise
        </div>
        """, unsafe_allow_html=True)
    with col_opt2:
        st.markdown("""
        <div class="strategy-card-page">
            <span class="strategy-badge badge-page">📄 Page-Level</span><br>
            <b>Analyzes every crawled URL.</b><br>
            Finds the exact best-fit page per domain. Best for <b>targeted sets</b>.
            <br><br>✅ Precise &nbsp;·&nbsp; ✅ Granular &nbsp;·&nbsp; ⚠️ Slower
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    strategy = st.radio(
        "Choose your TF-IDF matching strategy:",
        options=["Page-Level", "Domain-Level"],
        index=0 if st.session_state.tfidf_strategy == "Page-Level" else 1,
        horizontal=True,
    )
    st.session_state.tfidf_strategy = strategy

    if strategy == "Domain-Level":
        st.info(f"🌐 **Domain-Level** — will analyze **{domain_count} homepages**.")
    else:
        st.success(f"📄 **Page-Level** — will analyze **{total_pages} pages** across {domain_count} domains.")

    st.markdown("---")
    col_back, col_start = st.columns([1, 4])
    with col_back:
        if st.button("← Back"):
            st.session_state.step = 3; st.rerun()
    with col_start:
        start_match = st.button("🧮 Start TF-IDF Matching", type="primary", use_container_width=True)

    if st.session_state.matches_df is not None and not start_match:
        df_prev = st.session_state.matches_df
        if not df_prev.empty:
            st.success(f"✅ Previous results: **{len(df_prev)} matches**")
            st.dataframe(df_prev.head(50), use_container_width=True, height=350)
            csv_b = df_prev.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Matches CSV", csv_b,
                               "keyword_matches.csv", "text/csv", use_container_width=True)
            if st.button("▶ Proceed to Ahrefs Enrichment", type="primary", use_container_width=True):
                st.session_state.step = 5; st.rerun()
        else:
            st.warning("⚠️ Previous run returned 0 matches — try lower threshold or switch strategy.")

    if start_match:
        keywords  = st.session_state.keywords
        threshold = cfg["threshold"]
        all_pages = (
            [to_base_url(d) for d in st.session_state.domains]
            if strategy == "Domain-Level"
            else [p for v in crawl_data.values() for p in v["pages"]]
        )

        pb          = st.progress(0, text="📥 Fetching page content...")
        stat_box    = st.empty()
        page_texts  = {}
        completed   = 0
        total_fetch = len(all_pages)

        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg["workers"]) as executor:
            future_map = {executor.submit(fetch_page_text, url): url for url in all_pages}
            for future in concurrent.futures.as_completed(future_map):
                url, text = future.result()
                if text:
                    page_texts[url] = text
                completed += 1
                pb.progress(min(completed / total_fetch * 0.6, 0.59),
                            text=f"📥 {completed}/{total_fetch} ({len(page_texts)} OK)")

        pb.progress(0.65, text="🧮 Building TF-IDF matrix...")
        stat_box.empty()

        if not page_texts:
            st.error("❌ Could not fetch content from any pages.")
        else:
            valid_urls   = list(page_texts.keys())
            corpus       = [" ".join(keywords)] + [page_texts[u] for u in valid_urls]
            vectorizer   = TfidfVectorizer(stop_words="english", max_features=15000, ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform(corpus)
            pb.progress(0.80, text="📐 Computing cosine similarity...")
            similarities = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:])[0]
            pb.progress(0.95, text="🔎 Filtering by threshold...")
            max_score = float(np.max(similarities)) if len(similarities) > 0 else 0.0

            score_caption(
                f"🔍 Max similarity: <b>{max_score:.4f}</b> &nbsp;·&nbsp; "
                f"Threshold: <b>{threshold}</b> &nbsp;·&nbsp; Strategy: <b>{strategy}</b> &nbsp;·&nbsp; "
                f"{'✅ Matches expected' if max_score >= threshold else '⚠️ Lower threshold — no matches above cutoff'}"
            )

            matches = [
                {"keyword":    " | ".join(keywords[:8]),
                 "Page_URL":   url,
                 "Domain":     extract_domain(url),
                 "similarity": round(float(score), 4),
                 "strategy":   strategy}
                for url, score in zip(valid_urls, similarities)
                if score >= threshold
            ]

            matches_df = pd.DataFrame(matches)
            if not matches_df.empty:
                matches_df = matches_df.sort_values("similarity", ascending=False).reset_index(drop=True)
            else:
                matches_df = pd.DataFrame(columns=["keyword", "Page_URL", "Domain", "similarity", "strategy"])

            st.session_state.matches_df = matches_df
            pb.progress(1.0, text=f"✅ Done — {len(matches_df)} matches found!")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Pages Analyzed",  len(page_texts))
            c2.metric("Matches Found",   len(matches_df))
            c3.metric("Unique Domains",  matches_df["Domain"].nunique() if not matches_df.empty else 0)
            c4.metric("Avg Similarity",  f"{matches_df['similarity'].mean():.3f}" if not matches_df.empty else "—")

            if matches_df.empty:
                st.warning(
                    f"⚠️ No matches above **{threshold}**. Max score: **{max_score:.4f}**. "
                    "Lower the threshold in the sidebar or switch strategy."
                )
            else:
                st.markdown("### Top Matching Pages")
                st.dataframe(matches_df.head(100), use_container_width=True, height=400)
                csv_b = matches_df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Matches CSV", csv_b,
                                   "keyword_matches.csv", "text/csv", use_container_width=True)
                if st.button("▶ Proceed to Ahrefs Enrichment", type="primary", use_container_width=True):
                    st.session_state.step = 5; st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — AHREFS TRAFFIC & RANKING ENRICHMENT  (with full caching + page fix)
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 5:
    st.markdown("## Step 5 · Traffic & Ranking Enrichment (Ahrefs)")

    matches_df = st.session_state.matches_df
    if matches_df is None or matches_df.empty:
        st.error("❌ No matches found. Please complete Step 4 first.")
        if st.button("← Back to TF-IDF"):
            st.session_state.step = 4; st.rerun()
        st.stop()

    # ── Cache stats ──────────────────────────────────────────────────────
    dc = len(st.session_state.ahrefs_domain_cache)
    pc = len(st.session_state.ahrefs_page_cache)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("URLs to Enrich",  len(matches_df))
    c2.metric("Unique Domains",  matches_df["Domain"].nunique())
    c3.metric("Domains Cached",  dc)
    c4.metric("Pages Cached",    pc)

    if dc > 0 or pc > 0:
        cache_badge(
            f"💾 Session cache active — "
            f"<b>{dc}</b> domain metrics &amp; <b>{pc}</b> page traffic values will be reused "
            f"without additional API calls."
        )

    st.markdown("---")
    st.markdown("### 🔑 Ahrefs API Configuration")
    info_card(
        "Enter your Ahrefs API Bearer token. Never stored or logged. "
        "Find it at: <b>app.ahrefs.com → Account → API</b><br>"
        "<b>Page traffic fix:</b> uses exact-mode first, falls back to prefix-mode if traffic = 0."
    )

    api_token = st.text_input("Ahrefs API Bearer Token", type="password",
                              placeholder="Paste your API token here...")
    delay     = st.slider("API Request Delay (seconds)", 0.5, 3.0, 1.5, 0.25)

    st.markdown("#### Select Metrics to Fetch")
    c1, c2, c3 = st.columns(3)
    fetch_dr = c1.checkbox("Domain Rating (DR)",     value=True)
    fetch_dt = c2.checkbox("Monthly Domain Traffic", value=True)
    fetch_pt = c3.checkbox("Monthly Page Traffic",   value=True)

    col_back, col_start = st.columns([1, 4])
    with col_back:
        if st.button("← Back"):
            st.session_state.step = 4; st.rerun()
    with col_start:
        start_enrich = st.button("📊 Start Ahrefs Enrichment", type="primary",
                                 disabled=not api_token, use_container_width=True)

    if not api_token:
        st.warning("⚠️ Enter your Ahrefs API token to enable enrichment.")

    if st.session_state.enriched_df is not None and not start_enrich:
        st.success("✅ Previous enrichment results available:")
        st.dataframe(st.session_state.enriched_df, use_container_width=True, height=400)
        csv_b = st.session_state.enriched_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Enriched CSV", csv_b,
            f"linkbuilding_enriched_{datetime.today().strftime('%Y%m%d')}.csv",
            "text/csv", type="primary", use_container_width=True
        )

    if start_enrich and api_token:
        df = matches_df.copy()
        df["domain_rating"]          = None
        df["monthly_domain_traffic"] = None
        df["Monthly_Page_Traffic"]   = None

        # Pull references to session-level caches
        domain_cache = st.session_state.ahrefs_domain_cache
        page_cache   = st.session_state.ahrefs_page_cache

        total_rows  = len(df)
        api_calls   = 0
        cache_hits  = 0

        pb       = st.progress(0, text="📡 Starting Ahrefs enrichment...")
        stat_box = st.empty()
        info_box = st.empty()

        for idx, (i, row) in enumerate(df.iterrows()):
            domain   = str(row["Domain"]).strip()
            page_url = _normalise_url(str(row["Page_URL"]).strip())

            # ── Domain metrics (cached per domain) ───────────────────────
            if domain not in domain_cache:
                dr = None
                dt = None
                if fetch_dr:
                    dr = get_domain_rating(domain, api_token)
                    api_calls += 1
                    time.sleep(delay)
                if fetch_dt:
                    dt = get_domain_traffic(domain, api_token)
                    api_calls += 1
                    time.sleep(delay)
                domain_cache[domain] = {"dr": dr, "dt": dt}
            else:
                cache_hits += 1

            df.at[i, "domain_rating"]         = domain_cache[domain]["dr"]
            df.at[i, "monthly_domain_traffic"] = domain_cache[domain]["dt"]

            # ── Page traffic (cached per URL) ────────────────────────────
            if fetch_pt:
                if page_url not in page_cache:
                    pt = get_page_traffic(page_url, api_token)
                    # get_page_traffic already does exact + prefix fallback
                    page_cache[page_url] = pt
                    api_calls += 2   # counts both passes (worst case)
                    time.sleep(delay)
                else:
                    pt = page_cache[page_url]
                    cache_hits += 1
                df.at[i, "Monthly_Page_Traffic"] = pt

            # ── Update caches in session state ───────────────────────────
            st.session_state.ahrefs_domain_cache = domain_cache
            st.session_state.ahrefs_page_cache   = page_cache

            pb.progress(
                (idx + 1) / total_rows,
                text=f"📡 {idx+1}/{total_rows} · {domain}"
            )
            stat_box.markdown(
                f"**URL:** `{page_url[:65]}`  ·  "
                f"DR: **{domain_cache[domain]['dr']}**  ·  "
                f"Domain Traffic: **{domain_cache[domain]['dt']}**  ·  "
                f"Page Traffic: **{page_cache.get(page_url, '—')}**"
            )
            info_box.markdown(
                f"🔄 API calls made: **{api_calls}** &nbsp;|&nbsp; "
                f"💾 Cache hits: **{cache_hits}**"
            )

        pb.progress(1.0, text="✅ Ahrefs enrichment complete!")
        stat_box.empty()
        info_box.empty()

        # ── Final column order ────────────────────────────────────────────
        final_cols = ["keyword", "Page_URL", "similarity", "strategy",
                      "Monthly_Page_Traffic", "Domain",
                      "monthly_domain_traffic", "domain_rating"]
        df = df[[c for c in final_cols if c in df.columns]]
        st.session_state.enriched_df = df

        st.markdown("### 🎯 Final Link Opportunities")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Opportunities", len(df))
        avg_dr = df["domain_rating"].dropna().mean()          if "domain_rating"          in df.columns else 0
        avg_dt = df["monthly_domain_traffic"].dropna().mean() if "monthly_domain_traffic" in df.columns else 0
        avg_pt = df["Monthly_Page_Traffic"].dropna().mean()   if "Monthly_Page_Traffic"   in df.columns else 0
        c2.metric("Avg DR",             f"{avg_dr:.1f}")
        c3.metric("Avg Domain Traffic", f"{int(avg_dt):,}")
        c4.metric("Avg Page Traffic",   f"{int(avg_pt):,}")
        c5.metric("API Calls Saved",    cache_hits)

        st.dataframe(df, use_container_width=True, height=500)

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            csv_b = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Full Opportunities CSV", csv_b,
                f"linkbuilding_opportunities_{datetime.today().strftime('%Y%m%d')}.csv",
                "text/csv", type="primary", use_container_width=True
            )
        with col_d2:
            if st.button("🔄 Start New Analysis", use_container_width=True):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                st.rerun()
