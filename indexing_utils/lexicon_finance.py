# finance_lexicons_clean.py
# Minimal, news/finance-focused lexicons & finders (noise-reduced)

import re
from typing import Dict, List, Pattern

# ---------------------------
# Domain-focused lexicons
# ---------------------------

LEXICONS: Dict[str, List[str]] = {
    # 재무지표/라인아이템 (뉴스에 빈출, 의미 명확)
    "financial_metrics": [
        "revenue", "sales", "net sales", "net revenue", "turnover",
        "gross profit", "operating income", "operating profit", "EBIT",
        "EBITDA", "adjusted EBITDA", "net income", "net profit", "EPS",
        "diluted EPS", "free cash flow", "FCF", "cash from operations",
        "operating margin", "gross margin", "net margin",
        "guidance", "outlook", "run-rate", "ARR", "MRR", "bookings",
        "same-store sales", "comparable sales", "capex", "opex",
        "cost of goods sold", "COGS",
        "year-over-year", "y/y", "YoY", "quarter-over-quarter", "q/q", "QoQ"
    ],

    # 비율/지표
    "ratios_indicators": [
        "P/E", "price-to-earnings", "EV/EBITDA", "P/B", "price-to-book",
        "debt-to-equity", "current ratio", "quick ratio",
        "dividend yield", "return on equity", "ROE", "return on assets", "ROA",
        "return on invested capital", "ROIC"
    ],

    # 기업 이벤트/자본시장 이벤트
    "corp_events": [
        "merger", "acquisition", "M&A", "takeover",
        "buyback", "share repurchase", "tender offer",
        "dividend", "special dividend", "stock split", "reverse split",
        "spin-off", "spinoff", "carve-out", "IPO", "direct listing",
        "secondary offering", "follow-on offering", "private placement",
        "block sale", "block trade", "rights offering", "capital increase",
        "bond issuance", "note issuance", "convertible bond", "CB",
        "earnings release", "earnings call", "profit warning",
        "guidance raise", "guidance cut",
        "class action", "lawsuit", "antitrust", "investigation", "probe",
        "fine", "penalty", "recall", "sanction"
    ],

    # 원자재/벤치마크/거래소/단위
    "commodities_metals": [
        "gold", "silver", "copper", "aluminum", "nickel", "zinc", "lead", "tin",
        "iron ore", "steel", "lithium", "cobalt", "platinum", "palladium"
    ],
    "commodities_energy": [
        "crude", "oil", "Brent", "WTI", "natural gas", "LNG", "coal", "gasoline", "diesel",
        "heating oil", "naphtha"
    ],
    "commodities_agri": [
        "wheat", "corn", "soybeans", "coffee", "cocoa", "sugar", "cotton", "palm oil"
    ],
    "commodities_venues": [
        "LME", "COMEX", "CME", "NYMEX", "ICE", "MCX", "SHFE", "SGX", "DCE", "CBOT", "TOCOM", "DME", "LCH"
    ],
    "commodities_units": [
        "ton", "tons", "tonne", "tonnes", "mt", "kt", "mmt",
        "lb", "lbs", "pound", "pounds",
        "oz", "ounce", "ounces", "troy ounce",
        "bbl", "bbls", "mbbl", "mmbbl", "mbpd", "bpd", "boe", "mcfe", "mmcfe",
        "mcf", "mmcf", "bcf", "tcf", "mtpa"
    ],
    "commodities_market_terms": [
        "spread", "basis", "backwardation", "contango", "premium", "discount",
        "inventory", "stockpile", "warehouse stocks", "refined", "concentrate", "smelter", "refinery"
    ],

    # 통화/단위 (금액 파싱 보조)
    "currency_words": [
        "USD", "US$", "U.S. dollars", "dollar", "dollars", "EUR", "euro", "euros",
        "GBP", "pound", "pounds", "JPY", "yen", "CNY", "yuan", "KRW", "won",
        "INR", "rupee", "rupees", "AUD", "CAD", "CHF", "HKD", "SGD", "₩", "$", "€", "£", "¥"
    ],
    "magnitude": [
        "thousand", "k", "m", "mm", "million", "mln",
        "bn", "b", "billion", "bln", "tn", "trillion", "trln"
    ],

    # 공시/거래소/기간
    "sec_forms": [
        "10-K", "10-Q", "8-K", "20-F", "40-F", "6-K", "S-1", "S-3", "S-4",
        "F-1", "F-3", "F-4", "424B5", "13D", "13G", "13F", "DEF 14A", "SC 13D", "SC 13G",
        "Form 10-K", "Form 8-K", "Form 6-K", "Form 20-F",
        "DART filing", "FCA filing", "HKEX filing"
    ],
    "exchanges": [
        "NYSE", "Nasdaq", "AMEX", "LSE", "HKEX", "SSE", "SZSE", "TSE", "JPX",
        "Euronext", "TSX", "ASX", "KRX", "KOSPI", "KOSDAQ", "SGX", "SIX"
    ],
    "periods": [
        "Q1", "Q2", "Q3", "Q4", "first quarter", "second quarter",
        "third quarter", "fourth quarter", "fiscal first quarter",
        "FY", "fiscal year", "half-year", "H1", "H2", "nine months"
    ]
}

# --- Add to LEXICONS (append these keys) ---

LEXICONS.update({
    # 무역·관세·보조금/반덤핑
    "trade_measures": [
        "tariff", "tariffs", "Section 301", "anti-dumping", "countervailing duty", "CVD",
        "safeguard", "export control", "export controls", "entity list", "denied parties list",
        "secondary sanctions", "sanctions relief", "quota", "TRQ", "CBAM"
    ],

    # 법·정책(미국/EU/중국 중심, 제조/공급망 영향)
    "industrial_policy": [
        "CHIPS Act", "CHIPS and Science Act", "IRA", "Inflation Reduction Act",
        "Buy American", "Defense Production Act", "Critical Minerals",
        "Made in China 2025", "dual circulation", "supply chain resilience"
    ],

    # 규제기관/정책 부처(뉴스에 자주 명시되는 약칭 포함)
    "regulators_agencies": [
        "USTR", "BIS", "OFAC", "Commerce Department", "Treasury",
        "FTC", "DOJ Antitrust", "SEC", "CFTC",
        "European Commission", "DG COMP", "DG TRADE",
        "MOFCOM", "SAMR"
    ],

    # 중앙은행/지표(정책 헤드라인 키워드만)
    "macro_policy": [
        "FOMC", "rate hike", "rate cut", "dot plot", "quantitative tightening", "QT",
        "PCE", "core PCE", "CPI", "core CPI", "nonfarm payrolls", "NFP"
    ],

    # 에너지/공급 감산 블록
    "energy_blocs": [
        "OPEC", "OPEC+", "production cut", "output cut", "quota", "spare capacity"
    ],

    # 물류·지정학적 경로(공급망 충격 신호)
    "geo_routes": [
        "Red Sea", "Suez Canal", "Panama Canal", "Strait of Hormuz", "Taiwan Strait",
        "Cape of Good Hope reroute"
    ],

    # 지정학 이벤트/리스크
    "geopolitical_events": [
        "sanctions", "export ban", "import ban", "embargo", "boycott",
        "naval blockade", "missile strike", "ceasefire", "martial law"
    ],

    # 무역협정/블록
    "trade_blocs": [
        "USMCA", "RCEP", "CPTPP", "Mercosur", "EU-UK TCA"
    ],

    # 정책·정치 인물(뉴스 헤드라인에 빈출하고 경제·정책에 직결)
    "policy_figures": [
        "Donald Trump", "Jerome Powell", "Janet Yellen",
        "Christine Lagarde", "Kazuo Ueda", "Pan Gongsheng",
        "Ursula von der Leyen"
    ],
})

# --- [NEW] Pre-cleaners (place near Utilities) ---

HTML_GARBAGE_RE = re.compile(r"(?i)\b(?:ul><li|amp|nbsp|&amp;|&nbsp;)\b")
# Common news dateline heads: "NEW DELHI:", "LONDON, Oct 12 (Reuters) -", etc.
DATELINE_RE = re.compile(
    r"""(?ix)
    ^\s*
    (?:[A-Z][A-Z]+(?:\s+[A-Z][A-Z]+){0,3})      # ALL-CAPS city/country up to 4 tokens (e.g., NEW DELHI)
    (?:,\s*[A-Z][a-z]{2,9}\.?\s+\d{1,2})?       # optional: Month Day (e.g., Oct. 12)
    (?:\s*\(Reuters\))?                         # optional: (Reuters)
    \s*[:\-–]\s*
    """
)
# Strip bracketed newsroom boilerplate
BRACKETED_TAG_RE = re.compile(r"\[[^\]]{0,30}\bchar\b[^\]]{0,30}\]|\[[^\]]{1,40}\]", re.IGNORECASE)

# Single-letter and garbage tokens you saw in counts
SURFACE_NOISE = {
    "char","[ char","ul><li","amp","t","what'","article","Author","Price","Spot",
    "NEW","REUTERS","UPDATE"
}

def preclean_text(s: str) -> str:
    if not isinstance(s, str) or not s:
        return s
    x = s.strip()
    # Kill dateline at head once
    x = DATELINE_RE.sub("", x)
    # Remove bracketed boilerplates & HTML crumbs
    x = HTML_GARBAGE_RE.sub(" ", x)
    x = BRACKETED_TAG_RE.sub(" ", x)
    # Normalize multiple spaces
    x = re.sub(r"\s{2,}", " ", x)
    return x


# ---------------------------
# Regex (tight & news-tuned)
# ---------------------------

# 티커: 미국형/거래소접미사형/괄호형
TICKER_US_RE: Pattern = re.compile(r"\b[A-Z]{1,5}(?:\.[A-Z])?\b")
TICKER_RIC_RE: Pattern = re.compile(r"\b[A-Z0-9]{1,6}\.[A-Z]{1,3}\b")
TICKER_PAREN_RE: Pattern = re.compile(
    r"\((?:NYSE|Nasdaq|AMEX|LSE|HKEX|TSX|ASX|KRX|KOSPI|KOSDAQ|SGX|SIX)\s*:\s*[A-Z0-9\.]{1,8}\)",
    flags=re.I
)
TICKER_BLOCKLIST = {
    "AND","THE","FOR","WITH","THIS","THAT","FROM","WAS","ARE","IS",
    "CEO","CFO","EPS","EBITDA","ROE","ROA","GDP","CPI","PMI","ETF","ADR","IPO",
    "Q1","Q2","Q3","Q4","FY","H1","H2","US","U.S","U.S.","USA"
}

# 원자재 코드
COMMODITY_FUT_RIC_RE: Pattern = re.compile(r"\b[A-Z]{1,3}c[1-9]\b")   # LCOc1, CLc1, HGc1 ...
COMMODITY_SPOT_RIC_RE: Pattern = re.compile(r"\bX[A-Z]{2}=\b")       # XAU=, XAG=, XCU= ...
COMMODITY_PAREN_RE: Pattern = re.compile(
    r"\((?:COMEX|NYMEX|CME|ICE|LME|MCX|SHFE|SGX)\s*:\s*[A-Z0-9]{1,8}\)",
    flags=re.I
)
LME_THREEMO_RE: Pattern = re.compile(r"\bCM[A-Z]{2}\d\b")            # CMCU3 등
COMMODITY_SYMBOLS = {"GC","SI","HG","PA","PL","CL","NG","HO","RB","B"}  # 컨텍스트 있을 때만 허용

# 금액/퍼센트/기간
MONEY_WITH_UNIT_RE: Pattern = re.compile(
    r"(?ix)"
    r"(?:[\$€£¥₩]|USD|EUR|GBP|JPY|CNY|KRW)?\s*"
    r"([+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[+-]?\d+(?:\.\d+)?)\s*"
    r"(?:thousand|k|m|mm|million|mln|bn|b|billion|bln|tn|trillion|trln"
    r"|won|krw|억원|조원|억|조)?"
)
PERCENT_RE: Pattern = re.compile(r"(?i)([+-]?\d+(?:\.\d+)?)\s*%")
QUARTER_FY_RE: Pattern = re.compile(
    r"(?ix)\b(?:Q[1-4]\s*\d{2,4}|FY\s?\d{2,4}|FY\d{2,4}|(?:first|second|third|fourth)[ -]quarter(?:\s+\d{4})?)\b"
)
SEC_FORM_RE: Pattern = re.compile(
    r"\b(?:Form\s+)?(?:10-K|10-Q|8-K|20-F|40-F|6-K|S-1|S-3|S-4|F-1|F-3|F-4|424B5|13D|13G|13F|DEF\s*14A|SC\s*13D|SC\s*13G)\b",
    flags=re.I
)

# ---------------------------
# Utilities
# ---------------------------

def lexicon_regex(values: List[str]) -> Pattern:
    escaped = [re.escape(v) for v in sorted(values, key=len, reverse=True)]
    return re.compile(r"(?i)\b(" + r"|".join(escaped) + r")\b", flags=re.I)

LEX_RE = {k: lexicon_regex(v) for k, v in LEXICONS.items()}

def _looks_like_noise(tok: str) -> bool:
    if not tok:
        return True
    t = tok.strip()
    # pure digits, 한 글자 대문자, 보편적 대문자 단어
    if t.isdigit():
        return True
    if len(t) == 1 and t.isalpha() and t.isupper():
        return True
    if t in {"NEW","REUTERS","UPDATE"}:
        return True
    return False

def _denoise_entities(seq: List[str], protected: List[str]) -> List[str]:
    prot = set(protected)
    out = []
    for s in seq:
        if s in prot:
            out.append(s)
        elif not _looks_like_noise(s):
            out.append(s)
    # stable dedupe
    seen, dedup = set(), []
    for x in out:
        if x not in seen:
            dedup.append(x); seen.add(x)
    return dedup

# ---------------------------
# Finders
# ---------------------------

def find_lexicon_mentions(text: str) -> Dict[str, List[str]]:
    out = {}
    if not isinstance(text, str) or not text:
        return {k: [] for k in LEXICONS}
    for name, rx in LEX_RE.items():
        out[name] = list(dict.fromkeys(m.group(0) for m in rx.finditer(text)))
    return out

def _postfilter_ticker(c: str) -> bool:
    if c in TICKER_BLOCKLIST or c.isdigit() or len(c) > 6:
        return False
    return True

def find_tickers(text: str) -> List[str]:
    if not isinstance(text, str) or not text:
        return []
    out = []
    out += [m.group(0) for m in TICKER_RIC_RE.finditer(text)]  # 우선 특이패턴
    # 괄호형
    for m in TICKER_PAREN_RE.finditer(text):
        inner = m.group(0)
        t = re.findall(r":\s*([A-Z0-9\.]{1,8})\)", inner, flags=re.I)
        if t:
            out.append(t[0])
    # 미국형 (필터)
    out += [c for c in (m.group(0) for m in TICKER_US_RE.finditer(text)) if _postfilter_ticker(c)]
    # dedupe
    seen, dedup = set(), []
    for x in out:
        if x not in seen:
            dedup.append(x); seen.add(x)
    return dedup

def find_commodity_codes(text: str) -> List[str]:
    if not isinstance(text, str) or not text:
        return []
    out = []
    out += [m.group(0) for m in COMMODITY_FUT_RIC_RE.finditer(text)]
    out += [m.group(0) for m in COMMODITY_SPOT_RIC_RE.finditer(text)]
    out += [m.group(0) for m in COMMODITY_PAREN_RE.finditer(text)]
    out += [m.group(0) for m in LME_THREEMO_RE.finditer(text)]
    # 심볼 단독: 컨텍스트 있을 때만 허용
    ctx_ok = any(w in text for w in ("futures","future","contract","contracts","COMEX","NYMEX","ICE","LME","MCX"))
    if ctx_ok:
        out += [s for s in re.findall(r"\b[A-Z]{1,3}\b", text) if s in COMMODITY_SYMBOLS]
    # dedupe
    seen, dedup = set(), []
    for x in out:
        if x not in seen:
            dedup.append(x); seen.add(x)
    return dedup

def find_money_with_units(text: str) -> List[str]:
    if not isinstance(text, str) or not text:
        return []
    return [m.group(0) for m in MONEY_WITH_UNIT_RE.finditer(text)]

def find_percentages(text: str) -> List[str]:
    if not isinstance(text, str) or not text:
        return []
    return [m.group(0) for m in PERCENT_RE.finditer(text)]

def find_periods(text: str) -> List[str]:
    if not isinstance(text, str) or not text:
        return []
    return [m.group(0) for m in QUARTER_FY_RE.finditer(text)]

def enrich_entities_with_finance(text: str) -> Dict[str, List[str]]:
    """
    Return dict of high-signal finance/news entities with noise suppressed.
    """
    text = preclean_text(text)

    result = find_lexicon_mentions(text)

    stock_tickers = find_tickers(text)
    commodity_codes = find_commodity_codes(text)
    # money = find_money_with_units(text)
    # perc = find_percentages(text)
    # periods = find_periods(text)

    # protected = set(stock_tickers) | set(commodity_codes) | set(perc) | set(money)
    protected = set(stock_tickers) | set(commodity_codes)
    for k, vals in result.items():
        result[k] = _denoise_entities(vals, protected=list(protected))

    result["tickers"] = stock_tickers
    result["commodity_codes"] = commodity_codes
    # result["money_with_units"] = money
    # result["percents"] = perc
    # result["periods_found"] = periods
    return result
