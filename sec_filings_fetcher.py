"""
Conditional SEC filing fetcher for DanDanDonkeyBot.
Only fetches SEC data when questions contain finance/corporate keywords.
"""

import logging
import os
from typing import Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Financial/corporate keywords that trigger SEC data fetch
FINANCIAL_KEYWORDS = {
    # Company-specific
    "earnings", "revenue", "profit", "stock", "ipo", "merger", "acquisition",
    "dividend", "buyback", "bankruptcy", "debt", "loan", "credit rating",
    "guidance", "forecast", "eps", "pe ratio", "market cap", "valuation",
    
    # SEC filings
    "10-k", "10-q", "8-k", "sec filing", "edgar", "annual report",
    "quarterly", "financial statement", "balance sheet", "income statement",
    
    # Market events
    "sec", "sec investigation", "sec ruling", "delisting", "listing",
    "insider trading", "dilution", "share split", "shareholder",
    
    # Company performance
    "cash flow", "fcf", "gross margin", "ebitda", "enterprise value",
    "roe", "roa", "debt-to-equity", "current ratio",
    
    # Industry-specific finance
    "bank", "insurance", "pharmaceutical", "tech ipo", "spac",
    "private equity", "venture capital", "startup valuation",
    
    # Corporate actions
    "layoffs", "restructuring", "spinoff", "ceo departure", "board change"
}


def _safe_get_sec(url: str, timeout: int = 12, params: dict = None) -> dict | list | None:
    """Make HTTP request to SEC EDGAR API."""
    import requests
    try:
        headers = {
            "User-Agent": "DanDanDonkeyBot/1.0 (Forecasting Research)"
        }
        resp = requests.get(url, params=params, timeout=timeout, headers=headers)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.warning(f"SEC API request failed for {url[:80]}: {e}")
    return None


def _contains_financial_keywords(query_text: str) -> bool:
    """Check if question contains financial/corporate keywords."""
    query_lower = query_text.lower()
    return any(kw in query_lower for kw in FINANCIAL_KEYWORDS)


def _extract_ticker_from_query(query_text: str) -> Optional[str]:
    """
    Try to extract stock ticker from question text.
    Looks for patterns like "TICKER" in all caps (2-5 chars).
    """
    import re
    # Match 1-5 uppercase letters surrounded by word boundaries or spaces
    matches = re.findall(r'\b([A-Z]{1,5})\b', query_text)
    # Filter for likely tickers (2-4 chars are most common)
    tickers = [m for m in matches if 2 <= len(m) <= 4]
    return tickers[0] if tickers else None


def fetch_sec_filings(query_text: str) -> str:
    """
    Fetch recent SEC filings for a company.
    Only activates if question contains financial keywords.
    """
    # Early exit if not finance-related
    if not _contains_financial_keywords(query_text):
        return ""
    
    ticker = _extract_ticker_from_query(query_text)
    if not ticker:
        return ""
    
    try:
        # Query EDGAR CIK lookup by ticker
        ticker_lower = ticker.lower()
        cik_data = _safe_get(
            "https://www.sec.gov/files/company_tickers.json",
            timeout=10
        )
        
        if not cik_data:
            return ""
        
        # Search for matching CIK
        cik = None
        for entry in cik_data.values():
            if entry.get("ticker", "").lower() == ticker_lower:
                cik = str(entry.get("cik_str", "")).zfill(10)
                break
        
        if not cik:
            return ""
        
        # Fetch recent filings (10-K, 10-Q, 8-K)
        filings_data = _safe_get(
            f"https://data.sec.gov/submissions/CIK{cik}.json",
            timeout=10
        )
        
        if not filings_data or "filings" not in filings_data:
            return ""
        
        recent_filings = filings_data["filings"].get("recent", [])
        if not recent_filings:
            return ""
        
        # Filter for relevant filing types and get last 5
        relevant_types = {"10-K", "10-Q", "8-K"}
        filtered = [
            f for f in recent_filings
            if f.get("form", "") in relevant_types
        ][:5]
        
        if not filtered:
            return ""
        
        lines = [f"**SEC Filings ({ticker})**:"]
        for filing in filtered:
            form_type = filing.get("form", "?")
            filing_date = filing.get("filingDate", "?")
            accession = filing.get("accessionNumber", "").replace("-", "")
            
            # Calculate days ago
            try:
                fdate = datetime.strptime(filing_date, "%Y-%m-%d")
                days_ago = (datetime.now() - fdate).days
                days_str = f"{days_ago}d ago" if days_ago < 365 else filing_date
            except:
                days_str = filing_date
            
            lines.append(f"  - {form_type} filed {days_str} ({filing_date})")
        
        return "\n".join(lines)
    
    except Exception as e:
        logger.warning(f"SEC filing fetch failed for {ticker}: {e}")
        return ""


def fetch_sec_company_info(query_text: str) -> str:
    """
    Fetch company name, CIK, and SIC code from SEC EDGAR.
    Used for context when dealing with corporate questions.
    """
    if not _contains_financial_keywords(query_text):
        return ""
    
    ticker = _extract_ticker_from_query(query_text)
    if not ticker:
        return ""
    
    try:
        ticker_lower = ticker.lower()
        
        # Get company info from ticker lookup
        cik_data = _safe_get(
            "https://www.sec.gov/files/company_tickers.json",
            timeout=10
        )
        
        if not cik_data:
            return ""
        
        company_info = None
        for entry in cik_data.values():
            if entry.get("ticker", "").lower() == ticker_lower:
                company_info = entry
                break
        
        if not company_info:
            return ""
        
        cik = str(company_info.get("cik_str", "")).zfill(10)
        company_name = company_info.get("title", "Unknown")
        
        # Fetch additional details
        company_data = _safe_get(
            f"https://data.sec.gov/submissions/CIK{cik}.json",
            timeout=10
        )
        
        if not company_data:
            return f"**Company**: {company_name} ({ticker})"
        
        entity_info = company_data.get("entityInformation", {})
        sic = entity_info.get("sic", "N/A")
        business = entity_info.get("businessAddress", {})
        
        lines = [
            f"**Company**: {company_name}",
            f"**Ticker**: {ticker}",
            f"**SIC**: {sic}",
        ]
        
        if business:
            state = business.get("stateOrCountry", "")
            if state:
                lines.append(f"**Location**: {state}")
        
        return " | ".join(lines)
    
    except Exception as e:
        logger.warning(f"SEC company info fetch failed for {ticker}: {e}")
        return ""
