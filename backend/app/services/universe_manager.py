"""Dynamic market-universe ingestion and tiered refresh scheduling.

Index constituents are reference candidates, not a claim that every asset is
liquid or currently covered by the scoring engine. New symbols are inserted
as ``pending_refresh`` and remain subject to price, fundamentals, and liquidity
validation before they are useful in portfolio analysis.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from io import StringIO
from typing import Any

import httpx
import pandas as pd
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

MIN_ADV_USD = 200_000.0
CORE_UNIVERSE_LIMIT = 500
ROTATION_DAYS = 3
ROTATION_ANCHOR = date(2020, 1, 6)  # Monday


@dataclass(frozen=True)
class IndexSource:
    """A public constituents page and the table columns used to read it."""

    key: str
    url: str
    ticker_columns: tuple[str, ...]
    name_columns: tuple[str, ...] = ()
    core_candidate: bool = False


INDEX_SOURCES: tuple[IndexSource, ...] = (
    IndexSource(
        "sp500",
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        ("Symbol", "Ticker"),
        ("Security", "Company"),
        True,
    ),
    IndexSource(
        "nasdaq100",
        "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies",
        ("Ticker", "Ticker symbol"),
        ("Company", "Security"),
        False,
    ),
    # The public iShares IWM holdings CSV is used as a practical current
    # Russell 2000 basket proxy; fund tracking/replication may differ slightly
    # from FTSE Russell's licensed constituent file.
    IndexSource(
        "russell2000",
        "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/latest-holdings.csv",
        ("Ticker", "Symbol", "Ticker symbol"),
        ("Company", "Name"),
    ),
    IndexSource(
        "stoxx600",
        "https://en.wikipedia.org/wiki/STOXX_Europe_600",
        ("Ticker", "Ticker symbol", "Symbol"),
        ("Company", "Name"),
    ),
)

MAJOR_ETFS: Mapping[str, str] = {
    "SPY": "SPDR S&P 500 ETF Trust",
    "VOO": "Vanguard S&P 500 ETF",
    "VTI": "Vanguard Total Stock Market ETF",
    "QQQ": "Invesco QQQ Trust",
    "IWM": "iShares Russell 2000 ETF",
    "VT": "Vanguard Total World Stock ETF",
    "URTH": "iShares MSCI World ETF",
    "IEFA": "iShares Core MSCI EAFE ETF",
    "IEF": "iShares 7-10 Year Treasury Bond ETF",
    "TLT": "iShares 20+ Year Treasury Bond ETF",
    "AGG": "iShares Core U.S. Aggregate Bond ETF",
    "GLD": "SPDR Gold Shares",
    "IVV": "iShares Core S&P 500 ETF",
    "SPLG": "SPDR Portfolio S&P 500 ETF",
    "ITOT": "iShares Core S&P Total U.S. Stock Market ETF",
    "SCHB": "Schwab U.S. Broad Market ETF",
    "SCHX": "Schwab U.S. Large-Cap ETF",
    "IWB": "iShares Russell 1000 ETF",
    "VO": "Vanguard Mid-Cap ETF",
    "VB": "Vanguard Small-Cap ETF",
    "IJR": "iShares Core S&P Small-Cap ETF",
    "SCHA": "Schwab U.S. Small-Cap ETF",
    "QQQM": "Invesco NASDAQ 100 ETF",
    "ONEQ": "Fidelity NASDAQ Composite Index ETF",
    "RSP": "Invesco S&P 500 Equal Weight ETF",
    "SCHD": "Schwab U.S. Dividend Equity ETF",
    "VYM": "Vanguard High Dividend Yield ETF",
    "DGRO": "iShares Core Dividend Growth ETF",
    "HDV": "iShares Core High Dividend ETF",
    "NOBL": "ProShares S&P 500 Dividend Aristocrats ETF",
    "QUAL": "iShares MSCI USA Quality Factor ETF",
    "MTUM": "iShares MSCI USA Momentum Factor ETF",
    "USMV": "iShares MSCI USA Min Vol Factor ETF",
    "VLUE": "iShares MSCI USA Value Factor ETF",
    "SPLV": "Invesco S&P 500 Low Volatility ETF",
    "VUG": "Vanguard Growth ETF",
    "VTV": "Vanguard Value ETF",
    "VBR": "Vanguard Small-Cap Value ETF",
    "VOT": "Vanguard Mid-Cap Growth ETF",
    "XLK": "Technology Select Sector SPDR Fund",
    "XLF": "Financial Select Sector SPDR Fund",
    "XLE": "Energy Select Sector SPDR Fund",
    "XLV": "Health Care Select Sector SPDR Fund",
    "XLI": "Industrial Select Sector SPDR Fund",
    "XLP": "Consumer Staples Select Sector SPDR Fund",
    "XLY": "Consumer Discretionary Select Sector SPDR Fund",
    "XLRE": "Real Estate Select Sector SPDR Fund",
    "XLU": "Utilities Select Sector SPDR Fund",
    "XLC": "Communication Services Select Sector SPDR Fund",
    "XLB": "Materials Select Sector SPDR Fund",
    "SMH": "VanEck Semiconductor ETF",
    "SOXX": "iShares Semiconductor ETF",
    "IGV": "iShares Expanded Tech-Software Sector ETF",
    "IBB": "iShares Biotechnology ETF",
    "XBI": "SPDR S&P Biotech ETF",
    "ITA": "iShares U.S. Aerospace & Defense ETF",
    "KRE": "SPDR S&P Regional Banking ETF",
    "KBE": "SPDR S&P Bank ETF",
    "XRT": "SPDR S&P Retail ETF",
    "XOP": "SPDR S&P Oil & Gas Exploration & Production ETF",
    "XME": "SPDR S&P Metals & Mining ETF",
    "ACWI": "iShares MSCI ACWI ETF",
    "VXUS": "Vanguard Total International Stock ETF",
    "IXUS": "iShares Core MSCI Total International Stock ETF",
    "VEA": "Vanguard FTSE Developed Markets ETF",
    "VEU": "Vanguard FTSE All-World ex-US ETF",
    "VWO": "Vanguard FTSE Emerging Markets ETF",
    "EFA": "iShares MSCI EAFE ETF",
    "EEM": "iShares MSCI Emerging Markets ETF",
    "IEMG": "iShares Core MSCI Emerging Markets ETF",
    "EWJ": "iShares MSCI Japan ETF",
    "EWZ": "iShares MSCI Brazil ETF",
    "EWG": "iShares MSCI Germany ETF",
    "EWU": "iShares MSCI United Kingdom ETF",
    "INDA": "iShares MSCI India ETF",
    "MCHI": "iShares MSCI China ETF",
    "FXI": "iShares China Large-Cap ETF",
    "EWC": "iShares MSCI Canada ETF",
    "EWA": "iShares MSCI Australia ETF",
    "EWT": "iShares MSCI Taiwan ETF",
    "EWS": "iShares MSCI Singapore ETF",
    "EWY": "iShares MSCI South Korea ETF",
    "IDV": "iShares International Select Dividend ETF",
    "IQLT": "iShares MSCI Intl Quality Factor ETF",
    "BND": "Vanguard Total Bond Market ETF",
    "BNDX": "Vanguard Total International Bond ETF",
    "SHY": "iShares 1-3 Year Treasury Bond ETF",
    "SHV": "iShares Short Treasury Bond ETF",
    "VGIT": "Vanguard Intermediate-Term Treasury ETF",
    "VGSH": "Vanguard Short-Term Treasury ETF",
    "VGLT": "Vanguard Long-Term Treasury ETF",
    "GOVT": "iShares U.S. Treasury Bond ETF",
    "LQD": "iShares iBoxx Investment Grade Corporate Bond ETF",
    "HYG": "iShares iBoxx High Yield Corporate Bond ETF",
    "JNK": "SPDR Bloomberg High Yield Bond ETF",
    "VCIT": "Vanguard Intermediate-Term Corporate Bond ETF",
    "VCSH": "Vanguard Short-Term Corporate Bond ETF",
    "EMB": "iShares J.P. Morgan USD Emerging Markets Bond ETF",
    "TIP": "iShares TIPS Bond ETF",
    "SCHP": "Schwab U.S. TIPS ETF",
    "MUB": "iShares National Muni Bond ETF",
    "IGSB": "iShares 1-5 Year Investment Grade Corporate Bond ETF",
    "ICSH": "iShares Ultra Short-Term Bond ETF",
    "IUSB": "iShares Core Total USD Bond Market ETF",
    "BIV": "Vanguard Intermediate-Term Bond ETF",
    "BSV": "Vanguard Short-Term Bond ETF",
    "BLV": "Vanguard Long-Term Bond ETF",
    "SPAB": "SPDR Portfolio Aggregate Bond ETF",
    "FLOT": "iShares Floating Rate Bond ETF",
    "SLV": "iShares Silver Trust",
    "DBC": "Invesco DB Commodity Index Tracking Fund",
    "USO": "United States Oil Fund",
    "PDBC": "Invesco Optimum Yield Diversified Commodity Strategy ETF",
    "UNG": "United States Natural Gas Fund",
    "IAU": "iShares Gold Trust",
    "IBIT": "iShares Bitcoin Trust ETF",
    "FBTC": "Fidelity Wise Origin Bitcoin Fund",
}

TICKER_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.^=\-]{0,14}$")


def normalize_ticker(value: Any) -> str | None:
    """Normalize a provider/index symbol while rejecting malformed values."""
    if value is None:
        return None
    ticker = re.sub(r"\s+", "-", str(value).strip().upper())
    if re.fullmatch(r"[A-Z0-9]+\.[A-Z]", ticker):
        ticker = ticker.replace(".", "-")  # Yahoo convention for share classes.
    if not ticker or not TICKER_PATTERN.fullmatch(ticker):
        return None
    return ticker


def trading_day_slot(run_date: date, rotation_days: int = ROTATION_DAYS) -> int:
    """Return a stable weekday-only rotation slot, independent of weekends."""
    if rotation_days < 1:
        raise ValueError("rotation_days must be at least 1")
    days = (run_date - ROTATION_ANCHOR).days
    weeks, remainder = divmod(days, 7)
    trading_days = weeks * 5 + min(remainder, 5)
    return trading_days % rotation_days


def is_liquid_enough(adv_usd: float | None, minimum_adv_usd: float = MIN_ADV_USD) -> bool:
    """Return whether a measured USD average daily traded value clears policy."""
    if minimum_adv_usd <= 0:
        raise ValueError("minimum_adv_usd must be positive")
    try:
        value = float(adv_usd) if adv_usd is not None else float("nan")
    except (TypeError, ValueError):
        return False
    return pd.notna(value) and value > minimum_adv_usd


def select_refresh_tickers(
    memberships: Sequence[Mapping[str, Any]],
    *,
    portfolio_tickers: Sequence[str] = (),
    queued_tickers: Sequence[str] = (),
    run_date: date | None = None,
    core_limit: int = CORE_UNIVERSE_LIMIT,
    rotation_days: int = ROTATION_DAYS,
    minimum_adv_usd: float = MIN_ADV_USD,
) -> list[str]:
    """Select daily core, user-held, queued, and one extended rotation batch.

    Core ranking is descending by measured USD ADV, then market cap, with
    ticker as a deterministic tie-breaker. Missing metrics sort last. The
    3-day slot counts trading weekdays only. Complexity is ``O(U log U)``.
    """
    if core_limit < 1:
        raise ValueError("core_limit must be at least 1")
    high_priority, normal_priority = select_refresh_cohorts(
        memberships,
        portfolio_tickers=portfolio_tickers,
        queued_tickers=queued_tickers,
        run_date=run_date,
        core_limit=core_limit,
        rotation_days=rotation_days,
        minimum_adv_usd=minimum_adv_usd,
    )
    return high_priority + normal_priority


def select_refresh_cohorts(
    memberships: Sequence[Mapping[str, Any]],
    *,
    portfolio_tickers: Sequence[str] = (),
    queued_tickers: Sequence[str] = (),
    run_date: date | None = None,
    core_limit: int = CORE_UNIVERSE_LIMIT,
    rotation_days: int = ROTATION_DAYS,
    minimum_adv_usd: float = MIN_ADV_USD,
) -> tuple[list[str], list[str]]:
    """Return disjoint ``(high_priority, normal_priority)`` daily cohorts.

    All S&P 500, NASDAQ-100 and curated major ETF members are daily assets,
    regardless of the legacy tier flag. Saved portfolio tickers and explicit
    on-demand queue entries are also always daily, even when their liquidity
    is unknown or below the screener threshold. The rotating cohort contains
    eligible extended members for the current weekday slot, excluding every
    high-priority ticker. In the deployed schema saved configurations live in
    ``portfolios``; every row is active because that table has no archive flag.

    ``core_limit`` applies only to fallback ``tier=core`` rows without one of
    the explicit daily index/ETF memberships. Complexity is ``O(U log U + P)``
    for U memberships and P portfolio/queued symbols.
    """
    if core_limit < 1:
        raise ValueError("core_limit must be at least 1")
    slot = trading_day_slot(run_date or date.today(), rotation_days)
    daily_memberships = {"sp500", "nasdaq100", "major_etf"}
    explicit_daily = [
        row for row in memberships
        if daily_memberships.intersection(
            {str(item).lower() for item in (row.get("memberships") or [])}
        )
    ]
    fallback_core = [
        row for row in memberships
        if row.get("tier") == "core"
        and not daily_memberships.intersection(
            {str(item).lower() for item in (row.get("memberships") or [])}
        )
        and (row.get("adv_usd") is None or is_liquid_enough(row.get("adv_usd"), minimum_adv_usd))
    ]
    fallback_core.sort(key=lambda row: (
        -(float(row.get("adv_usd") or 0.0)),
        -(float(row.get("market_cap_usd") or 0.0)),
        str(row.get("ticker", "")),
    ))
    high = [normalize_ticker(row.get("ticker")) for row in explicit_daily]
    high.extend(normalize_ticker(row.get("ticker")) for row in fallback_core[:core_limit])
    high.extend(normalize_ticker(ticker) for ticker in portfolio_tickers)
    high.extend(normalize_ticker(ticker) for ticker in queued_tickers)
    high_priority = list(dict.fromkeys(ticker for ticker in high if ticker))
    high_set = set(high_priority)

    normal_priority = []
    for row in memberships:
        ticker = normalize_ticker(row.get("ticker"))
        if (
            ticker
            and ticker not in high_set
            and row.get("tier") == "extended"
            and int(row.get("rotation_batch", -1)) == slot
            and (row.get("adv_usd") is None or is_liquid_enough(row.get("adv_usd"), minimum_adv_usd))
        ):
            normal_priority.append(ticker)
    return high_priority, list(dict.fromkeys(normal_priority))


async def load_portfolio_tickers(client: Any) -> set[str]:
    """Read the union of saved target tickers and held tickers from Supabase.

    The canonical configuration table is ``portfolios`` (the repository's
    ``PortfolioRepository``); the holdings table is ``portfolio_holdings``.
    Pagination avoids silently missing symbols once an account has many rows.
    Errors for either table are isolated so a deployment missing the optional
    holdings migration still refreshes configured portfolios. Complexity is
    ``O(R)`` over returned portfolio and holding rows.
    """
    tickers: set[str] = set()

    async def rows_for(table: str, columns: str, order_column: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            query = client.table(table).select(columns).order(order_column).range(offset, offset + 999)
            result = query.execute()
            if hasattr(result, "__await__"):
                result = await result
            page = result.data or []
            rows.extend(page)
            if len(page) < 1000:
                break
            offset += 1000
        return rows

    try:
        for row in await rows_for("portfolio_holdings", "ticker", "ticker"):
            ticker = normalize_ticker(row.get("ticker"))
            if ticker:
                tickers.add(ticker)
    except Exception:
        logger.info("Portfolio holdings unavailable; using saved target configurations")
    try:
        for row in await rows_for("portfolios", "tickers", "id"):
            for value in row.get("tickers") or []:
                ticker = normalize_ticker(value)
                if ticker:
                    tickers.add(ticker)
    except Exception:
        logger.warning("Saved portfolio configurations unavailable during universe scheduling", exc_info=True)
    return tickers


class UniverseManager:
    """Fetch, deduplicate, persist, and validate a global investment universe.

    The provider and DB clients are injectable for offline tests. The default
    constituent sources are public HTML pages/files and receive an explicit
    User-Agent. Russell coverage uses iShares IWM equity holdings as a current
    basket proxy, which can differ slightly from official licensed membership.
    Refreshes are sequential and spaced to avoid hammering source
    sites. Database writes are batched and idempotent.
    """

    def __init__(
        self,
        *,
        http_client: Any | None = None,
        db_client: Any | None = None,
        sources: Sequence[IndexSource] = INDEX_SOURCES,
        source_fetchers: Mapping[str, Callable[[IndexSource], Awaitable[Any]]] | None = None,
        ticker_lookup: Callable[[str], Awaitable[Mapping[str, Any] | None]] | None = None,
        minimum_adv_usd: float = MIN_ADV_USD,
        request_spacing_seconds: float = 1.0,
        sleeper: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ) -> None:
        if minimum_adv_usd <= 0:
            raise ValueError("minimum_adv_usd must be positive")
        self.http_client = http_client
        self.db_client = db_client
        self.sources = tuple(sources)
        self.source_fetchers = dict(source_fetchers or {})
        self.ticker_lookup = ticker_lookup
        self.minimum_adv_usd = float(minimum_adv_usd)
        self.request_spacing_seconds = max(0.0, float(request_spacing_seconds))
        self._sleep = sleeper
        self._last_request_at: float | None = None

    async def _get_json(self, url: str) -> Mapping[str, Any]:
        await self._space_requests()
        if self.http_client is not None:
            response = await self.http_client.get(url, timeout=15, headers={"User-Agent": "MoatCheck research tool contact: support@moatcheck.app"})
        else:
            async with httpx.AsyncClient(timeout=15, headers={"User-Agent": "MoatCheck research tool contact: support@moatcheck.app"}) as client:
                response = await client.get(url)
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, Mapping) else {}

    async def _space_requests(self) -> None:
        if self._last_request_at is not None and self.request_spacing_seconds:
            remaining = self.request_spacing_seconds - (time.monotonic() - self._last_request_at)
            if remaining > 0:
                await self._sleep(remaining)
        self._last_request_at = time.monotonic()

    async def fetch_source(self, source: IndexSource) -> list[dict[str, Any]]:
        """Fetch one constituent table and return normalized membership rows."""
        fetcher = self.source_fetchers.get(source.key)
        if fetcher is not None:
            tables = await fetcher(source)
        else:
            await self._space_requests()
            if self.http_client is not None:
                response = await self.http_client.get(
                    source.url,
                    timeout=25,
                    headers={"User-Agent": "MoatCheck research tool contact: support@moatcheck.app"},
                )
            else:
                async with httpx.AsyncClient(timeout=25) as client:
                    response = await client.get(
                        source.url,
                        headers={"User-Agent": "MoatCheck research tool contact: support@moatcheck.app"},
                    )
            response.raise_for_status()
            tables = (
                [pd.read_csv(StringIO(response.text), skiprows=8)]
                if source.url.lower().endswith(".csv")
                else self._parse_html_tables(response.text)
            )
        if isinstance(tables, pd.DataFrame):
            tables = [tables]
        if isinstance(tables, list) and tables and isinstance(tables[0], Mapping):
            tables = [pd.DataFrame(tables)]
        if not isinstance(tables, Sequence) or isinstance(tables, (str, bytes)):
            raise ValueError(f"{source.key} constituent response is not tabular")

        table = next((candidate for candidate in tables if isinstance(candidate, pd.DataFrame) and self._column(candidate, source.ticker_columns)), None)
        if table is None:
            raise ValueError(f"No ticker column found for {source.key}")
        ticker_col = self._column(table, source.ticker_columns)
        name_col = self._column(table, source.name_columns)
        asset_class_col = self._column(table, ("Asset Class", "Asset type"))
        rows: list[dict[str, Any]] = []
        for _, record in table.iterrows():
            if asset_class_col and str(record.get(asset_class_col, "")).strip().casefold() != "equity":
                continue
            ticker = normalize_ticker(record.get(ticker_col))
            if ticker:
                rows.append({
                    "ticker": ticker,
                    "name": str(record[name_col]).strip() if name_col and pd.notna(record.get(name_col)) else None,
                    "membership": source.key,
                    "core_candidate": source.core_candidate,
                })
        if not rows:
            raise ValueError(f"{source.key} constituent response contained no tickers")
        return rows

    @staticmethod
    def _column(frame: pd.DataFrame, names: Sequence[str]) -> str | None:
        lookup = {str(column).strip().casefold(): str(column) for column in frame.columns}
        return next((lookup[name.casefold()] for name in names if name.casefold() in lookup), None)

    @staticmethod
    def _parse_html_tables(html: str) -> list[pd.DataFrame]:
        """Extract simple index tables without requiring a native HTML parser."""
        soup = BeautifulSoup(html, "html.parser")
        frames: list[pd.DataFrame] = []
        for table in soup.find_all("table"):
            rows = table.find_all("tr")
            header_index = next(
                (index for index, row in enumerate(rows) if row.find_all("th", recursive=False)),
                None,
            )
            if header_index is None:
                continue
            headers = [cell.get_text(" ", strip=True) for cell in rows[header_index].find_all("th", recursive=False)]
            records: list[list[str]] = []
            for row in rows[header_index + 1:]:
                cells = row.find_all(["td", "th"], recursive=False)
                if len(cells) != len(headers):
                    continue
                records.append([cell.get_text(" ", strip=True) for cell in cells])
            if headers and records:
                frames.append(pd.DataFrame(records, columns=headers))
        return frames

    async def fetch_universe(self) -> dict[str, dict[str, Any]]:
        """Fetch all index constituents and ETFs into a deduplicated ticker map."""
        combined: dict[str, dict[str, Any]] = {}
        source_errors: dict[str, str] = {}
        successful_sources = 0
        for source in self.sources:
            try:
                rows = await self.fetch_source(source)
                successful_sources += 1
                for row in rows:
                    ticker = row["ticker"]
                    current = combined.setdefault(ticker, {
                        "ticker": ticker,
                        "name": row["name"],
                        "memberships": [],
                        "core_candidate": False,
                        "asset_type": "equity",
                    })
                    if row["membership"] not in current["memberships"]:
                        current["memberships"].append(row["membership"])
                    current["name"] = current["name"] or row["name"]
                    current["core_candidate"] = current["core_candidate"] or row["core_candidate"]
            except Exception as exc:
                source_errors[source.key] = type(exc).__name__
                logger.warning("Could not refresh %s membership (%s)", source.key, type(exc).__name__)
        if successful_sources == 0:
            self.source_errors = source_errors
            raise RuntimeError("No index constituent source succeeded; refusing an ETF-only universe refresh")
        for ticker, name in MAJOR_ETFS.items():
            current = combined.setdefault(ticker, {
                "ticker": ticker,
                "name": name,
                "memberships": [],
                "core_candidate": False,
                "asset_type": "etf",
            })
            current["name"] = current["name"] or name
            current["asset_type"] = "etf"
            if "major_etf" not in current["memberships"]:
                current["memberships"].append("major_etf")
        if not combined:
            raise RuntimeError("No market constituents were fetched from any configured source")
        self.source_errors = source_errors
        return combined

    async def _execute(self, query: Any) -> Any:
        result = query.execute()
        return await result if hasattr(result, "__await__") else result

    async def _db(self) -> Any:
        if self.db_client is not None:
            return self.db_client
        from app.db.supabase import get_async_supabase_client
        return await get_async_supabase_client()

    async def sync(self) -> dict[str, Any]:
        """Upsert universe memberships and register unseen tickers for refresh."""
        universe = await self.fetch_universe()
        client = await self._db()
        tickers = list(universe)
        existing: set[str] = set()
        for start in range(0, len(tickers), 500):
            result = await self._execute(
                client.table("stocks").select("ticker").in_("ticker", tickers[start:start + 500])
            )
            existing.update(str(row["ticker"]).upper() for row in (result.data or []))

        prior_memberships: dict[str, Mapping[str, Any]] = {}
        for start in range(0, len(tickers), 500):
            result = await self._execute(
                client.table("market_universe_memberships")
                .select("ticker,adv_usd,market_cap_usd,is_liquid")
                .in_("ticker", tickers[start:start + 500])
            )
            prior_memberships.update({
                str(row["ticker"]).upper(): row for row in (result.data or [])
            })
        for ticker, row in universe.items():
            prior = prior_memberships.get(ticker, {})
            row["adv_usd"] = prior.get("adv_usd")
            row["market_cap_usd"] = prior.get("market_cap_usd")

        new_stocks = [
            {
                "ticker": row["ticker"],
                "name": row["name"],
                "is_benchmark": False,
                "asset_type": row["asset_type"],
                "currency": "USD" if row["asset_type"] == "etf" or any(
                    item in row["memberships"] for item in ("sp500", "nasdaq100", "russell2000")
                ) else None,
                # Becomes active after a successful first price refresh.
                "status": "pending_refresh",
            }
            for row in universe.values()
            if row["ticker"] not in existing
        ]
        await self._upsert_batches(client, "stocks", new_stocks, "ticker")

        memberships = []
        core_candidates = sorted(
            (row for row in universe.values() if row["core_candidate"] and row["asset_type"] == "equity"),
            # Existing measured USD ADV / market cap rank known members;
            # new members without measurements sort last deterministically.
            key=lambda row: (
                -(float(row.get("adv_usd") or 0.0)),
                -(float(row.get("market_cap_usd") or 0.0)),
                row["ticker"],
            ),
        )
        core_tickers = {row["ticker"] for row in core_candidates[:CORE_UNIVERSE_LIMIT]}
        daily_memberships = {"sp500", "nasdaq100", "major_etf"}
        core_tickers.update(
            row["ticker"] for row in universe.values()
            if daily_memberships.intersection({str(item).lower() for item in row["memberships"]})
        )
        for row in universe.values():
            memberships.append({
                "ticker": row["ticker"],
                "memberships": row["memberships"],
                "asset_type": row["asset_type"],
                "tier": "core" if row["ticker"] in core_tickers else "extended",
                "rotation_batch": stable_rotation_batch(row["ticker"], ROTATION_DAYS),
                "minimum_adv_usd": self.minimum_adv_usd,
                "is_liquid": is_liquid_enough(row.get("adv_usd"), self.minimum_adv_usd),
            })
        await self._upsert_batches(client, "market_universe_memberships", memberships, "ticker")
        pruned_memberships = 0
        if not getattr(self, "source_errors", {}):
            prune_result = await self._execute(client.rpc(
                "prune_market_universe_memberships", {"p_current_tickers": tickers}
            ))
            pruned_memberships = int(getattr(prune_result, "data", 0) or 0)
        else:
            # A failed source must not make its previous members appear
            # delisted. Preserve the catalog until every configured source
            # has refreshed successfully.
            logger.warning("Skipping universe pruning because one or more index sources failed")
        # Index/ETF core and user portfolio symbols get the high-priority
        # daily queue. The remaining extended universe is selected separately
        # by DataPipelineOrchestrator in its stable 3-day rotating batch.
        portfolio_tickers = await load_portfolio_tickers(client)
        high_priority_tickers = sorted(core_tickers | portfolio_tickers)
        await self._enqueue_refresh_tickers(client, high_priority_tickers)
        return {
            "tickers": len(universe),
            "new_tickers": len(new_stocks),
            "memberships_upserted": len(memberships),
            "memberships_pruned": pruned_memberships,
            "high_priority_tickers": len(high_priority_tickers),
            "portfolio_tickers": len(portfolio_tickers),
            "source_errors": getattr(self, "source_errors", {}),
        }

    async def _upsert_batches(self, client: Any, table: str, rows: list[dict[str, Any]], conflict: str) -> None:
        for start in range(0, len(rows), 500):
            await self._execute(client.table(table).upsert(rows[start:start + 500], on_conflict=conflict))

    async def _insert_refresh_queue(self, client: Any, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        await self._enqueue_refresh_tickers(client, [str(row["ticker"]) for row in rows])

    async def _enqueue_refresh_tickers(self, client: Any, tickers: Sequence[str]) -> None:
        """Call the migration's idempotent RPC for the queue's partial index."""
        normalized = list(dict.fromkeys(t for value in tickers if (t := normalize_ticker(value))))
        if not normalized:
            return
        for start in range(0, len(normalized), 500):
            await self._execute(client.rpc(
                "enqueue_refresh_tickers", {"p_tickers": normalized[start:start + 500]}
            ))

    async def ingest_on_demand(self, ticker: str) -> dict[str, Any]:
        """Validate an unknown search symbol and queue it for a real-data refresh.

        The lookup must return an exact ticker match. Liquidity is checked when
        quote-volume data is available; unknown ADV is left pending and is not
        represented as a liquid asset. No portfolio or market score is faked.
        """
        normalized = normalize_ticker(ticker)
        if not normalized:
            raise ValueError("ticker has an invalid symbol format")
        if self.ticker_lookup is None:
            lookup = await self._yahoo_lookup(normalized)
        else:
            lookup = await self.ticker_lookup(normalized)
        if not lookup or normalize_ticker(lookup.get("symbol")) != normalized:
            raise LookupError(f"No exact market symbol match was found for {normalized}")
        adv_usd = lookup.get("adv_usd")
        if adv_usd is not None and not is_liquid_enough(float(adv_usd), self.minimum_adv_usd):
            raise ValueError(f"{normalized} is below the minimum average daily traded value")
        asset_type = "etf" if str(lookup.get("asset_type", "equity")).lower() == "etf" else "equity"
        client = await self._db()
        stock = {
            "ticker": normalized,
            "is_benchmark": False,
            "asset_type": asset_type,
            "status": "pending_refresh",
        }
        # Do not replace existing good metadata with nulls when the lookup
        # provider has only a partial symbol response.
        if lookup.get("name"):
            stock["name"] = lookup["name"]
        currency = str(lookup.get("currency") or "").strip().upper()
        if len(currency) == 3 and currency.isalpha():
            stock["currency"] = currency
        await self._execute(client.table("stocks").upsert(stock, on_conflict="ticker"))
        await self._execute(client.table("market_universe_memberships").upsert({
            "ticker": normalized,
            "memberships": ["user_search"],
            "asset_type": asset_type,
            "tier": "core",
            "rotation_batch": 0,
            "adv_usd": adv_usd,
            "minimum_adv_usd": self.minimum_adv_usd,
            "is_liquid": adv_usd is not None and is_liquid_enough(float(adv_usd), self.minimum_adv_usd),
        }, on_conflict="ticker"))
        await self._insert_refresh_queue(client, [{"ticker": normalized, "priority": "high"}])
        return {
            "ticker": normalized,
            "name": lookup.get("name"),
            "currency": stock.get("currency"),
            "asset_type": asset_type,
            "status": "pending_refresh",
            "liquidity_verified": adv_usd is not None,
            "adv_usd": adv_usd,
        }

    async def _yahoo_lookup(self, ticker: str) -> Mapping[str, Any] | None:
        payload = await self._get_json(
            "https://query1.finance.yahoo.com/v1/finance/search?"
            + httpx.QueryParams({"q": ticker, "quotesCount": 5, "newsCount": 0}).__str__()
        )
        for item in payload.get("quotes", []):
            if normalize_ticker(item.get("symbol")) == ticker:
                quote_type = str(item.get("quoteType", "EQUITY")).upper()
                return {
                    "symbol": ticker,
                    "name": item.get("shortname") or item.get("longname"),
                    "asset_type": "etf" if quote_type == "ETF" else "equity",
                    "currency": item.get("currency"),
                    # Yahoo search does not reliably return ADV. Leave it
                    # unverified until stored price/volume observations arrive.
                    "adv_usd": None,
                }
        return None


async def _async_main() -> int:
    report = await UniverseManager().sync()
    logger.info("Universe sync complete: %s", report)
    return 0


def main() -> int:
    """CLI entrypoint used by the scheduled workflow before price ingestion."""
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    return asyncio.run(_async_main())


def stable_rotation_batch(ticker: str, rotation_days: int = ROTATION_DAYS) -> int:
    """Assign stable extended-universe batch membership by ticker hash."""
    if rotation_days < 1:
        raise ValueError("rotation_days must be at least 1")
    # Do not use Python's randomized hash(), or GitHub workers would assign
    # different rotation batches on each run.
    return sum((index + 1) * ord(character) for index, character in enumerate(ticker.upper())) % rotation_days


if __name__ == "__main__":
    raise SystemExit(main())
