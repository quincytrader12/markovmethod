"""Sector taxonomy — the standard eleven, plus the things that are not equities.

The scan groups grew ad hoc and it showed. "Megacap" is a size, "energy" is a
sector and "crypto" is an asset class, all sitting in one list as if they were
the same kind of thing, and whole sectors were simply missing: no Real Estate,
no Materials, no Utilities, and Communication Services hiding under the name
"telecom" while Alphabet sat in a bucket called "megacap".

This is the GICS eleven, which is the classification exchanges, index providers
and screeners actually use. Alphabet is Communication Services next to Meta and
Netflix; Visa is Financials, not technology; Amazon and Tesla are Consumer
Discretionary. Where GICS reclassified something, the current placement is used
rather than where it used to live.

Two deliberate additions to the eleven. ETFs are not a sector — an index fund
holds every sector at once — so they get their own section, and so does crypto.
Size and style stay separate too: mega caps and high-beta growth names cut
across sectors rather than sitting beside them.

One honest limit. Alpaca's asset list carries no sector field, so this map only
covers names written down here. Everything the full-market sweep finds beyond
them is ranked normally but reported as unclassified rather than guessed at.
"""

from __future__ import annotations

# ── the eleven ──────────────────────────────────────────────────────────────
SECTORS: dict[str, dict] = {
    "communication": {
        "name": "Communication Services",
        "blurb": "Media, telecom, interactive entertainment — where Alphabet and Meta live",
        "symbols": [
            "GOOGL", "GOOG", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS",
            "CHTR", "WBD", "PARA", "FOXA", "NWSA", "OMC", "IPG", "LYV", "EA",
            "TTWO", "MTCH", "SPOT", "PINS", "SNAP", "RBLX", "TTD", "ROKU",
        ],
    },
    "discretionary": {
        "name": "Consumer Discretionary",
        "blurb": "What people buy when they feel comfortable — retail, autos, travel, leisure",
        "symbols": [
            "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "BKNG", "TJX",
            "ORLY", "AZO", "CMG", "MAR", "HLT", "GM", "F", "RIVN", "LCID",
            "YUM", "DPZ", "DRI", "ROST", "ULTA", "LULU", "DECK", "BBY", "EBAY",
            "ETSY", "CHWY", "CVNA", "W", "RCL", "CCL", "NCLH", "LVS", "MGM",
            "DKNG", "ABNB", "GPS", "M", "TM", "HMC", "BABA", "PDD", "JD",
        ],
    },
    "staples": {
        "name": "Consumer Staples",
        "blurb": "What people buy regardless — food, drink, household, discount retail",
        "symbols": [
            "PG", "KO", "PEP", "WMT", "COST", "CL", "KMB", "GIS", "K", "HSY",
            "MDLZ", "KHC", "STZ", "TAP", "MO", "PM", "KR", "SYY", "EL", "CLX",
            "TGT", "DG", "DLTR", "ADM", "HRL", "CAG", "CPB", "MKC", "SJM", "UL",
        ],
    },
    "energy": {
        "name": "Energy",
        "blurb": "Oil, gas, drilling and refining",
        "symbols": [
            "XOM", "CVX", "COP", "OXY", "SLB", "HAL", "BKR", "EOG", "DVN",
            "FANG", "HES", "MPC", "PSX", "VLO", "KMI", "WMB", "OKE", "LNG",
            "TRGP", "APA", "MRO", "CTRA", "SHEL", "BP", "TTE", "E",
        ],
    },
    "financials": {
        "name": "Financials",
        "blurb": "Banks, insurers, exchanges and payments — Visa and Mastercard sit here, not tech",
        "symbols": [
            "BRK.B", "JPM", "BAC", "WFC", "GS", "MS", "C", "SCHW", "BLK", "BX",
            "KKR", "AXP", "V", "MA", "PYPL", "COF", "USB", "PNC", "TFC", "BK",
            "STT", "SPGI", "MCO", "ICE", "CME", "NDAQ", "AIG", "MET", "PRU",
            "ALL", "TRV", "PGR", "CB", "AFRM", "SOFI", "HOOD", "COIN",
            "HSBC", "BCS", "MUFG", "SMFG", "ING", "BBVA", "SAN",
        ],
    },
    "health": {
        "name": "Health Care",
        "blurb": "Pharma, biotech, devices and insurers",
        "symbols": [
            "UNH", "LLY", "JNJ", "PFE", "MRK", "ABBV", "ABT", "TMO", "DHR",
            "BMY", "AMGN", "GILD", "CVS", "CI", "ELV", "HCA", "MDT", "SYK",
            "BSX", "ISRG", "ZTS", "REGN", "VRTX", "BIIB", "MRNA", "IDXX", "EW",
            "DXCM", "ALGN", "HUM", "MCK", "VTRS", "AZN", "GSK", "NVO", "SNY",
        ],
    },
    "industrials": {
        "name": "Industrials",
        "blurb": "Machinery, defence, transport and airlines",
        "symbols": [
            "CAT", "DE", "BA", "GE", "HON", "MMM", "LMT", "RTX", "NOC", "GD",
            "UNP", "CSX", "NSC", "UPS", "FDX", "DAL", "UAL", "AAL", "LUV",
            "EMR", "ETN", "PH", "ITW", "CMI", "PCAR", "ROK", "URI", "WM",
            "RSG", "JCI", "CARR", "OTIS", "PWR", "FAST", "GWW", "RKLB", "ASTS",
        ],
    },
    "technology": {
        "name": "Information Technology",
        "blurb": "Hardware, semiconductors and software",
        "symbols": [
            "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "AMD", "ADBE", "ACN",
            "CSCO", "INTC", "IBM", "TXN", "QCOM", "NOW", "INTU", "AMAT", "MU",
            "LRCX", "KLAC", "ADI", "PANW", "SNPS", "CDNS", "ANET", "MSI", "APH",
            "GLW", "HPQ", "HPE", "DELL", "NTAP", "STX", "WDC", "NXPI", "MCHP",
            "ON", "SWKS", "QRVO", "MRVL", "TER", "ENTG", "ARM", "SMCI", "TSM",
            "ASML", "CRWD", "SNOW", "DDOG", "NET", "ZS", "MDB", "TEAM", "OKTA",
            "TWLO", "PLTR", "U", "PATH", "S", "WDAY", "ADSK", "ANSS", "PTC",
            "TYL", "JNPR", "FFIV", "AKAM", "KEYS", "SHOP", "SAP", "INFY", "WIT",
        ],
    },
    "materials": {
        "name": "Materials",
        "blurb": "Chemicals, metals, mining and building materials",
        "symbols": [
            "LIN", "SHW", "APD", "ECL", "FCX", "NEM", "NUE", "DOW", "DD", "PPG",
            "VMC", "MLM", "ALB", "CE", "IFF", "LYB", "STLD", "CF", "MOS",
            "RIO", "BHP", "VALE",
        ],
    },
    "realestate": {
        "name": "Real Estate",
        "blurb": "REITs — property, data centres, towers and warehouses",
        "symbols": [
            "PLD", "AMT", "EQIX", "CCI", "PSA", "SPG", "O", "WELL", "DLR",
            "VICI", "AVB", "EQR", "EXR", "MAA", "INVH", "ARE", "VTR", "IRM",
            "SBAC", "WY",
        ],
    },
    "utilities": {
        "name": "Utilities",
        "blurb": "Power, water and gas distribution — the defensive corner",
        "symbols": [
            "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "ED", "PEG",
            "WEC", "ES", "AWK", "DTE", "PPL", "FE", "AEE", "CMS", "CNP", "ATO",
        ],
    },
}

# ── not sectors ─────────────────────────────────────────────────────────────
# An index fund holds every sector at once, so filing it under one would be
# wrong. Same for crypto, which is not an equity at all.
NON_EQUITY: dict[str, dict] = {
    "etf": {
        "name": "ETFs & Funds",
        "blurb": "Index, sector and theme funds — trade a whole market or a whole sector",
        "symbols": [
            "SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "ARKK", "SCHD", "VIG", "VYM",
            "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE",
            "XLC", "SMH", "IBB", "XBI", "ITB", "XRT", "XOP", "JETS", "KRE", "GDX",
            "URA", "EEM", "EFA", "TLT", "HYG", "LQD", "AGG", "GLD", "SLV", "USO",
            "VNQ", "IYR",
        ],
    },
    "crypto": {
        "name": "Crypto",
        "blurb": "Digital assets, traded around the clock",
        "symbols": ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD"],
    },
}

# ── style, which cuts across sectors rather than sitting beside them ────────
STYLES: dict[str, dict] = {
    "megacap": {
        "name": "Mega caps",
        "blurb": "The largest companies, whatever sector they are in",
        "symbols": [
            "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO",
            "NFLX", "COST", "JPM", "V", "UNH", "LLY", "XOM", "HD", "WMT", "PG",
            "MA", "ORCL", "BRK.B", "JNJ", "ABBV", "CVX", "MRK",
        ],
    },
    "growth": {
        "name": "High-beta growth",
        "blurb": "Fast movers where a regime flip tends to show up earliest",
        "symbols": [
            "PLTR", "COIN", "CRWD", "SNOW", "DDOG", "NET", "ZS", "MDB", "TEAM",
            "HOOD", "SOFI", "AFRM", "RBLX", "DKNG", "ABNB", "SHOP", "TTD",
            "ROKU", "PINS", "SNAP", "U", "PATH", "S", "OKTA", "TWLO", "ETSY",
            "CHWY", "CVNA", "RIVN", "LCID", "ENPH", "FSLR", "IONQ", "RKLB",
            "ASTS", "SMCI", "ARM", "MRNA", "AI",
        ],
    },
}


def _build_lookup() -> dict[str, str]:
    """symbol -> group key. Sectors win; style groups never claim a symbol."""
    out: dict[str, str] = {}
    for key, grp in list(SECTORS.items()) + list(NON_EQUITY.items()):
        for sym in grp["symbols"]:
            out.setdefault(sym, key)
    return out


SECTOR_OF = _build_lookup()

# Everything selectable, in the order the UI should offer it: the eleven
# sectors alphabetically by display name, then the non-equity sections, then
# the style cuts.
ALL_GROUPS: dict[str, dict] = {
    **{k: SECTORS[k] for k in sorted(SECTORS, key=lambda k: SECTORS[k]["name"])},
    **NON_EQUITY,
    **STYLES,
}


def sector_name(symbol: str) -> str:
    """Display name of a symbol's sector, or an honest blank.

    Alpaca does not publish sector data, so anything the sweep turns up beyond
    the map above is unclassified. Saying so beats guessing from the ticker.
    """
    key = SECTOR_OF.get((symbol or "").upper())
    if not key:
        return ""
    return (SECTORS.get(key) or NON_EQUITY.get(key, {})).get("name", "")


def sector_key(symbol: str) -> str:
    return SECTOR_OF.get((symbol or "").upper(), "")


def equity_universe() -> list[str]:
    """Every classified equity — the eleven sectors, deduplicated, no funds."""
    seen: dict[str, None] = {}
    for grp in SECTORS.values():
        for sym in grp["symbols"]:
            seen.setdefault(sym, None)
    return list(seen)


def full_universe() -> list[str]:
    """Every curated name: sectors, funds and crypto."""
    seen: dict[str, None] = {}
    for grp in list(SECTORS.values()) + list(NON_EQUITY.values()):
        for sym in grp["symbols"]:
            seen.setdefault(sym, None)
    return list(seen)


def counts() -> dict[str, int]:
    return {k: len(v["symbols"]) for k, v in ALL_GROUPS.items()}
