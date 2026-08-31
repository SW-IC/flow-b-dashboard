import os
import json
import time
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from pathlib import Path

"""
FLOW B — PARAMETER SWEEP (EMA21 ENTRY DISTANCE + EMA/SMA AUTOPSY)

For each pair of consecutive earnings dates:
1. Let the first session that trades the print finish (BMO/intraday = that
   day; AMC = next day). Anchor on that close. Start looking for the dip
   on the following trading day (BMO day 1 → start day 2; AMC day 1 → start day 3).
2. Enter at the first later close at or below the drawdown threshold AND far
   enough below EMA21.
3. Reject if a disqualifying volume spike occurred from anchor through entry.
4. No trade until SMA200 exists (200 daily closes). Then reject if price
   spent more than N days below SMA200 from anchor through entry
   (N swept: 10, 20, 25).
5. Skip the pair if consecutive Yahoo dates are more than
   MAX_EARNINGS_GAP_DAYS apart (a hole in the calendar, not one quarter).
6. Exit at t+3 around the next earnings date, only after entry.
7. At entry, capture 9/21/50 EMA distance and above/below flag.
   SMA 20/50/200 is tracked separately for comparison only.

Autopsy compares top 10% vs bottom 10% trades on MA position + distance.
"""
import numpy as np
import pandas as pd
import requests
import yfinance as yf

SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR / "flow_b_cache"
CACHE_DIR.mkdir(exist_ok=True)

EARNINGS_EXIT_OFFSET_GRID = [3]  # t+3 fixed
# One quarter only. Q3→Q4 is routinely 105–116 calendar days; cache p99 is 136.
# Wider than this is a skipped print (Yahoo hole), not the next earnings.
MAX_EARNINGS_GAP_DAYS = 140
DROPS = [-0.20, -0.25]
VOL_MULTS = [7.0, 10.0]  # not in user list; leftover from looser group
EMA21_DIST_THRESH_GRID = [-0.13, -0.15, -0.18]  # require dist_ema21_pct <= threshold
SMA200_MAX_DAYS_BELOW_GRID = [10, 20, 25]
MIN_TRADES = 500  # optimizer constraint: n_trades > MIN_TRADES

HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/csv,application/json,text/html;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# iShares concatenates share-class suffixes (BRKB). Yahoo uses a hyphen (BRK-B).
ISHARES_TO_YAHOO = {
    "BRKA": "BRK-A",
    "BRKB": "BRK-B",
    "BFA": "BF-A",
    "BFB": "BF-B",
    "LENB": "LEN-B",
    "UHALB": "UHAL-B",
    "HEIA": "HEI-A",
    "MOGA": "MOG-A",
    "MOGB": "MOG-B",
    "GEFB": "GEF-B",
    "CWENA": "CWEN-A",
    "CRDA": "CRD-A",
    "CRDB": "CRD-B",
    "BIOB": "BIO-B",
    "WSOB": "WSO-B",
}

MA_NAMES = ("ema9", "ema21", "ema50", "sma20", "sma50", "sma200")


# ── cache ────────────────────────────────────────────────────────────────
def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.pkl"


def _cache_meta_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.meta.json"


def _save_cache(name: str, obj, *, fingerprint: str = ""):
    _cache_path(name).parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        obj.to_pickle(_cache_path(name))
    else:
        pd.to_pickle(obj, _cache_path(name))
    _cache_meta_path(name).write_text(json.dumps({"fingerprint": fingerprint, "version": 2}))


def _load_cache(name: str):
    p = _cache_path(name)
    if not p.exists():
        return None
    try:
        return pd.read_pickle(p)
    except Exception as e:
        print(f"  [cache] failed to read {name}: {e}")
        return None


def _cache_fingerprint(name: str) -> str:
    meta_p = _cache_meta_path(name)
    if not meta_p.exists():
        return ""
    try:
        return json.loads(meta_p.read_text()).get("fingerprint", "")
    except Exception:
        return ""


def _today() -> str:
    return pd.Timestamp.now().strftime("%Y-%m-%d")


# ── HTTP / tickers ───────────────────────────────────────────────────────
def _http_get(url: str, *, retries: int = 3, timeout: int = 30) -> requests.Response:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=HTTP_HEADERS, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"GET failed after {retries} tries: {url} ({last_err})") from last_err


def _looks_like_html(text: str) -> bool:
    head = text.lstrip()[:80].lower()
    return head.startswith("<!doctype") or head.startswith("<html")


def _to_yahoo_ticker(raw: str) -> str:
    """Map a holdings ticker onto the symbol yfinance actually understands."""
    t = str(raw).strip().upper()
    t = t.replace(".", "-").replace("/", "-")
    t = ISHARES_TO_YAHOO.get(t, t)
    return t


def _hyphen_class_variant(ticker: str) -> str | None:
    """BRKB -> BRK-B. Only used as a last-resort retry after a failed download."""
    t = ticker.strip().upper()
    if "-" in t or len(t) < 3:
        return None
    if t[-1] in "ABC" and t[:-1].isalpha():
        return f"{t[:-1]}-{t[-1]}"
    return None


def _dedupe_tickers(tickers: list[str]) -> list[str]:
    out = []
    seen = set()
    for t in tickers:
        t = _to_yahoo_ticker(t)
        if not t or t in {"-", "NAN", "NONE"} or t in seen:
            continue
        if not any(ch.isalpha() for ch in t):
            continue
        seen.add(t)
        out.append(t)
    return out


def _parse_ishares_holdings(text: str) -> list[str]:
    if _looks_like_html(text):
        raise ValueError("iShares returned HTML instead of CSV")
    lines = text.splitlines()
    header_idx = next((i for i, line in enumerate(lines) if line.startswith("Ticker,")), None)
    if header_idx is None:
        raise ValueError(
            f"'Ticker,' header not found ({len(lines)} lines, start={text[:180]!r})"
        )
    raw = pd.read_csv(StringIO("\n".join(lines[header_idx:])), thousands=",")
    raw.columns = [c.strip() for c in raw.columns]
    if "Ticker" not in raw.columns:
        raise ValueError(f"No Ticker column in iShares CSV: {list(raw.columns)}")

    if "Asset Class" in raw.columns:
        asset = raw["Asset Class"].astype(str).str.strip().str.lower()
        raw = raw[asset.eq("equity")]
    tickers = raw["Ticker"].dropna().astype(str)
    tickers = tickers[tickers.str.strip().ne("-")]
    return _dedupe_tickers(tickers.tolist())


def _load_spx_wikipedia() -> list[str]:
    """S&P 1500 (500 + 400 + 600) as a fallback universe."""
    urls = [
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
        "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
    ]
    tickers = []
    for url in urls:
        resp = _http_get(url)
        tables = pd.read_html(StringIO(resp.text))
        df = next((t for t in tables if "Symbol" in t.columns), tables[0])
        col = "Symbol" if "Symbol" in df.columns else df.columns[0]
        tickers.extend(df[col].astype(str).tolist())
    return _dedupe_tickers(tickers)


def _load_sp500_github() -> list[str]:
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
    df = pd.read_csv(url)
    return _dedupe_tickers(df["Symbol"].astype(str).tolist())


def _cached_ticker_list(name: str) -> list[str] | None:
    cached = _load_cache(name)
    if cached is None:
        return None
    if isinstance(cached, pd.Series):
        return [str(x) for x in cached.tolist()]
    if isinstance(cached, (list, tuple)):
        return [str(x) for x in cached]
    return None


def _try_ishares_holdings(urls: list[str], *, min_n: int, label: str):
    """Return (tickers, None) on success, else (None, last_error)."""
    last_err = None
    for url in urls:
        try:
            resp = _http_get(url, retries=2)
            universe = _parse_ishares_holdings(resp.text)
            if len(universe) >= min_n:
                print(f"  [r3000] {label}: {len(universe)} names")
                return universe, None
            last_err = ValueError(f"only {len(universe)} tickers from {url}")
            print(f"  [r3000] {last_err}")
        except Exception as e:
            last_err = e
            print(f"  [r3000] {label} fetch failed: {e}")
    return None, last_err


def _load_r3000() -> list[str]:
    """
    Russell 3000 proxy via iShares IWV holdings (today's constituents).

    Broader than IWB/R1000: keeps names that dropped from large-cap into
    small/mid. Still a *current* snapshot — delisted names are not in IWV
    and yfinance will not resurrect them.

    Names are normalized to yfinance symbols (BRKB -> BRK-B). Falls back to
    IWB+IWM union, Wikipedia S&P 1500, GitHub S&P 500, then stale cache.
    """
    today_fp = _today()
    if _cache_fingerprint("r3000") == today_fp:
        cached = _dedupe_tickers(_cached_ticker_list("r3000") or [])
        if cached and len(cached) >= 2000:
            print(f"  [r3000] using today's cache ({len(cached)} names)")
            return cached

    last_err = None
    iwv, err = _try_ishares_holdings(
        [
            "https://www.ishares.com/us/products/239714/ishares-russell-3000-etf/latest-holdings.csv",
            "https://www.ishares.com/us/products/239714/ishares-russell-3000-etf/1467271812596.ajax?fileType=csv&fileName=IWV_holdings&dataType=fund",
        ],
        min_n=2000,
        label="iShares IWV (R3000)",
    )
    last_err = err or last_err
    if iwv:
        _save_cache("r3000", pd.Series(iwv), fingerprint=today_fp)
        return iwv

    # R1000 + R2000 is the same index split across two ETFs.
    iwb, err = _try_ishares_holdings(
        [
            "https://www.ishares.com/us/products/239707/ishares-russell-1000-etf/latest-holdings.csv",
            "https://www.ishares.com/us/products/239707/ishares-russell-1000-etf/1467271812596.ajax?fileType=csv&fileName=IWB_holdings&dataType=fund",
        ],
        min_n=800,
        label="iShares IWB (R1000)",
    )
    last_err = err or last_err
    iwm, err = _try_ishares_holdings(
        [
            "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/latest-holdings.csv",
            "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund",
        ],
        min_n=1500,
        label="iShares IWM (R2000)",
    )
    last_err = err or last_err
    if iwb and iwm:
        union = _dedupe_tickers(iwb + iwm)
        if len(union) >= 2000:
            print(f"  [r3000] IWB+IWM union: {len(union)} names")
            _save_cache("r3000", pd.Series(union), fingerprint=today_fp)
            return union

    for loader, label, min_n in (
        (_load_spx_wikipedia, "Wikipedia S&P 1500", 1200),
        (_load_sp500_github, "GitHub S&P 500", 400),
    ):
        try:
            universe = loader()
            if len(universe) >= min_n:
                print(f"  [r3000] fallback {label}: {len(universe)} names")
                _save_cache("r3000", pd.Series(universe), fingerprint=today_fp)
                return universe
            print(f"  [r3000] {label} too small ({len(universe)})")
        except Exception as e:
            print(f"  [r3000] {label} failed: {e}")

    for cache_name, min_n in (("r3000", 1500), ("top1000", 400)):
        stale = _dedupe_tickers(_cached_ticker_list(cache_name) or [])
        if stale and len(stale) >= min_n:
            print(f"  [r3000] using stale {cache_name} cache ({len(stale)} names). Last error: {last_err}")
            return stale
    raise RuntimeError(f"Could not load a ticker universe. Last error: {last_err}")


# ── prices ───────────────────────────────────────────────────────────────
def _chunks(lst: list, size: int):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def _extract_close_volume(data: pd.DataFrame, chunk: list[str]):
    if data is None or data.empty:
        return pd.DataFrame(), pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].copy() if "Close" in data.columns.get_level_values(0) else pd.DataFrame()
        volume = data["Volume"].copy() if "Volume" in data.columns.get_level_values(0) else pd.DataFrame()
        if isinstance(close, pd.Series):
            close = close.to_frame(chunk[0])
        if isinstance(volume, pd.Series):
            volume = volume.to_frame(chunk[0])
        return close, volume
    close = data[["Close"]].rename(columns={"Close": chunk[0]}) if "Close" in data.columns else pd.DataFrame()
    volume = data[["Volume"]].rename(columns={"Volume": chunk[0]}) if "Volume" in data.columns else pd.DataFrame()
    return close, volume


def _yf_download(chunk: list[str], period: str, threads: bool) -> pd.DataFrame:
    return yf.download(
        chunk,
        period=period,
        auto_adjust=True,
        progress=False,
        threads=threads and len(chunk) > 1,
        timeout=60,
        group_by="column",
    )


def _chunked_download(tickers: list[str], *, period: str, label: str = "download"):
    """
    Bulk yf.download in shrinking chunks. Missing names are retried in
    smaller batches, then one-by-one, then with a hyphenated share-class
    variant (BRKB -> BRK-B) so dual-class names are not silently dropped.
    """
    tickers = list(dict.fromkeys(tickers))
    close_parts, volume_parts = [], []
    remaining = list(tickers)
    renamed = {}  # original -> yahoo symbol actually downloaded

    passes = (
        (80, 0.6, True),
        (20, 0.4, True),
        (1, 0.12, False),
    )
    for chunk_size, pause_sec, threads in passes:
        if not remaining:
            break
        n_chunks = (len(remaining) + chunk_size - 1) // chunk_size
        still = []
        for idx, chunk in enumerate(_chunks(remaining, chunk_size), start=1):
            print(f"  [{label}] size={chunk_size} chunk {idx}/{n_chunks} ({len(chunk)} tickers)")
            try:
                data = _yf_download(chunk, period, threads)
            except Exception as e:
                print(f"    chunk failed ({e}); will retry")
                still.extend(chunk)
                time.sleep(pause_sec)
                continue
            close, volume = _extract_close_volume(data, chunk)
            got = [t for t in chunk if t in close.columns and close[t].notna().any()]
            missed = [t for t in chunk if t not in got]
            if got:
                close_parts.append(close[got])
                if not volume.empty:
                    keep = [t for t in got if t in volume.columns]
                    if keep:
                        volume_parts.append(volume[keep])
            still.extend(missed)
            if idx < n_chunks and pause_sec:
                time.sleep(pause_sec)
        remaining = still

    # Last-chance: hyphenated share-class symbol for names that still failed.
    if remaining:
        print(f"  [{label}] trying share-class variants for {len(remaining)} leftover tickers")
        still = []
        for t in remaining:
            alt = _hyphen_class_variant(t)
            if not alt or alt == t:
                still.append(t)
                continue
            try:
                data = _yf_download([alt], period, threads=False)
                close, volume = _extract_close_volume(data, [alt])
                if alt in close.columns and close[alt].notna().any():
                    close_parts.append(close.rename(columns={alt: t}))
                    if alt in volume.columns:
                        volume_parts.append(volume.rename(columns={alt: t}))
                    renamed[t] = alt
                    print(f"    rescued {t} as {alt}")
                else:
                    still.append(t)
            except Exception:
                still.append(t)
        remaining = still

    close_all = pd.concat(close_parts, axis=1) if close_parts else pd.DataFrame()
    volume_all = pd.concat(volume_parts, axis=1) if volume_parts else pd.DataFrame()
    if not close_all.empty:
        close_all = close_all.loc[:, ~close_all.columns.duplicated()]
    if not volume_all.empty:
        volume_all = volume_all.loc[:, ~volume_all.columns.duplicated()]

    if remaining:
        preview = ", ".join(remaining[:20]) + (" ..." if len(remaining) > 20 else "")
        print(f"  [{label}] dropped {len(remaining)} with no Yahoo data: {preview}")
    return close_all, volume_all, remaining, renamed


def _load_price_volume(tickers: list[str], period: str = "5y"):
    """Reuse cached prices; only download names that are actually missing."""
    close_cache = _load_cache("prices_close")
    volume_cache = _load_cache("prices_volume")
    today_fp = _today()
    cache_fp = _cache_fingerprint("prices_close")

    have_cache = isinstance(close_cache, pd.DataFrame) and isinstance(volume_cache, pd.DataFrame)
    if have_cache:
        cached_tickers = set(close_cache.columns)
        missing = [t for t in tickers if t not in cached_tickers]
        if not missing:
            if cache_fp and cache_fp != today_fp:
                print(f"  [prices] using cache from {cache_fp} ({close_cache.shape[1]} tickers)")
            return close_cache, volume_cache
        print(f"  [prices] cache hit {len(cached_tickers)}; downloading {len(missing)} missing")
        new_close, new_vol, _, _ = _chunked_download(missing, period=period, label="prices-missing")
        prices_all = pd.concat([close_cache, new_close], axis=1)
        volumes_all = pd.concat([volume_cache, new_vol], axis=1)
        prices_all = prices_all.loc[:, ~prices_all.columns.duplicated()]
        volumes_all = volumes_all.loc[:, ~volumes_all.columns.duplicated()]
        _save_cache("prices_close", prices_all, fingerprint=today_fp)
        _save_cache("prices_volume", volumes_all, fingerprint=today_fp)
        return prices_all, volumes_all

    prices_all, volumes_all, _, _ = _chunked_download(tickers, period=period, label="prices")
    _save_cache("prices_close", prices_all, fingerprint=today_fp)
    _save_cache("prices_volume", volumes_all, fingerprint=today_fp)
    return prices_all, volumes_all


# ── earnings ─────────────────────────────────────────────────────────────
def _naive_timestamps(idx) -> list[pd.Timestamp]:
    """Keep clock time (16:00 AMC vs 08:00 BMO). Do not floor to midnight."""
    idx = pd.DatetimeIndex(idx)
    if idx.tz is not None:
        idx = idx.tz_convert("America/New_York").tz_localize(None)
    return sorted(idx.unique())


def _as_naive_day(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tz is not None:
        ts = ts.tz_convert("America/New_York").tz_localize(None)
    return ts.normalize()


def _calendar_gap_days(start, end) -> int:
    return int((_as_naive_day(end) - _as_naive_day(start)).days)


def _priced_session_idx(index: pd.DatetimeIndex, ts) -> int:
    """
    Index of the first daily bar whose session traded with earnings public.

    BMO / during market (hour < 16 ET): that calendar day's close.
    AMC (hour >= 16 ET): the next session's close (today's close is pre-print).

    Caller then starts the dip window on the *next* bar, so:
      during/BMO on day 1 → start counting on day 2
      AMC on day 1        → start counting on day 3
    """
    ts = pd.Timestamp(ts)
    if ts.tz is not None:
        ts = ts.tz_convert("America/New_York").tz_localize(None)
    day = ts.normalize()
    if ts.hour >= 16:
        return int(index.searchsorted(day + pd.Timedelta(days=1)))
    return int(index.searchsorted(day))


def get_earnings_dates(ticker: str) -> list[pd.Timestamp]:
    t = yf.Ticker(ticker)
    df = None
    for _ in range(2):
        try:
            df = t.get_earnings_dates(limit=24)
            break
        except Exception:
            time.sleep(0.4)
    if df is None or df.empty:
        return []
    try:
        return _naive_timestamps(df.index)
    except Exception:
        return []


def _load_earnings_dates(tickers: list[str], workers: int = 6) -> dict:
    """Fetch only missing/empty names. Parallel, with a same-day skip for empties."""
    today_fp = _today()
    cache = _load_cache("earnings_dates")
    cached = cache.to_dict() if isinstance(cache, pd.Series) else {}
    cached = {str(k): (list(v) if v is not None else []) for k, v in cached.items()}

    if _cache_fingerprint("earnings_dates") == today_fp:
        missing = [t for t in tickers if t not in cached]
    else:
        missing = [t for t in tickers if t not in cached or not cached[t]]

    if missing:
        print(f"  [earnings] fetching {len(missing)} tickers ({workers} workers)")
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(get_earnings_dates, t): t for t in missing}
            for fut in as_completed(futs):
                tkr = futs[fut]
                try:
                    cached[tkr] = fut.result()
                except Exception:
                    cached[tkr] = []
                done += 1
                if done % 50 == 0 or done == len(missing):
                    print(f"  [earnings] {done}/{len(missing)}")
        flat = pd.Series({k: v for k, v in cached.items()})
        _save_cache("earnings_dates", flat, fingerprint=today_fp)
    else:
        print("  [earnings] cache hit")

    return {t: cached.get(t, []) for t in tickers}


# ── MA / volume helpers ──────────────────────────────────────────────────
def volume_spike_prefix(volume, mult):
    trailing_avg = volume.shift(1).rolling(window=20, min_periods=1).mean()
    spike = (volume > mult * trailing_avg) & trailing_avg.notna()
    return spike.astype(int).cumsum()


def has_volume_spike(spike_prefix, start_idx, end_idx):
    before = spike_prefix.iloc[start_idx - 1] if start_idx > 0 else 0
    return spike_prefix.iloc[end_idx] - before > 0


def _compute_mas(px: pd.Series) -> dict:
    return {
        "ema9": px.ewm(span=9, adjust=False, min_periods=9).mean(),
        "ema21": px.ewm(span=21, adjust=False, min_periods=21).mean(),
        "ema50": px.ewm(span=50, adjust=False, min_periods=50).mean(),
        "sma20": px.rolling(20).mean(),
        "sma50": px.rolling(50).mean(),
        "sma200": px.rolling(200).mean(),
    }


def _dist(price, ma):
    if pd.isna(ma) or ma == 0:
        return np.nan
    return (price - ma) / ma


def _align_naive(s: pd.Series) -> pd.Series:
    s = s.copy()
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_convert("America/New_York").tz_localize(None)
    s.index = s.index.normalize()
    return s[~s.index.duplicated(keep="last")]


def _prepare_universe(tickers, prices_all, volumes_all, earnings_dict, vol_mults):
    """Compute MAs + volume-spike prefixes once per ticker (not once per combo)."""
    prepared = {}
    skipped = {"no_price": 0, "no_earnings": 0}
    for tkr in tickers:
        if tkr not in prices_all.columns or tkr not in volumes_all.columns:
            skipped["no_price"] += 1
            continue
        dates = earnings_dict.get(tkr) or []
        if len(dates) < 2:
            skipped["no_earnings"] += 1
            continue
        market = pd.concat(
            [_align_naive(prices_all[tkr]).rename("price"),
             _align_naive(volumes_all[tkr]).rename("volume")],
            axis=1,
        ).dropna()
        if market.empty:
            skipped["no_price"] += 1
            continue
        px = market["price"]
        prepared[tkr] = {
            "px": px,
            "mas": _compute_mas(px),
            "spike": {m: volume_spike_prefix(market["volume"], m) for m in vol_mults},
            "dates": dates,
        }
    print(
        f"  Prepared {len(prepared)} tickers "
        f"(no price/volume: {skipped['no_price']}, <2 earnings dates: {skipped['no_earnings']})"
    )
    return prepared


def _ma_fields(entry_price, mas, entry_date) -> dict:
    vals = {}
    for name in MA_NAMES:
        series = mas[name]
        vals[name] = series.loc[entry_date] if entry_date in series.index else np.nan
    out = {}
    for name in MA_NAMES:
        val = vals[name]
        dist = _dist(entry_price, val)
        above = entry_price > val if pd.notna(val) else np.nan
        out[name] = val
        out[f"dist_{name}_pct"] = dist
        out[f"above_{name}"] = above
    # Primary MA aliases are true EMAs (no SMA fallback).
    for n in (9, 21, 50):
        out[f"ma{n}"] = out[f"ema{n}"]
        out[f"dist_ma{n}_pct"] = out[f"dist_ema{n}_pct"]
        out[f"above_ma{n}"] = out[f"above_ema{n}"]
    return out


def find_entries(prepared, drop_thresh, vol_mult, ema21_dist_thresh):
    """Entry-side trades only. Exit offset is applied afterwards (cheap)."""
    rows = []
    for tkr, data in prepared.items():
        dates = data["dates"]
        px = data["px"]
        mas = data["mas"]
        spike_prefix = data["spike"][vol_mult]
        ema21 = mas["ema21"]
        sma200 = mas["sma200"]

        for i in range(len(dates) - 1):
            ev_date, next_ev_date = dates[i], dates[i + 1]
            gap_days = _calendar_gap_days(ev_date, next_ev_date)
            if gap_days < 1 or gap_days > MAX_EARNINGS_GAP_DAYS:
                continue
            start_idx = _priced_session_idx(px.index, ev_date)
            end_idx = _priced_session_idx(px.index, next_ev_date)
            if start_idx >= len(px) or end_idx >= len(px) or end_idx <= start_idx:
                continue
            # SMA200 needs 200 daily closes. Skip the window until it exists
            # at the earnings close so the days-below veto is not a no-op (NaN < x is False).
            if pd.isna(sma200.iloc[start_idx]):
                continue

            anchor_price = px.iloc[start_idx]
            threshold_price = anchor_price * (1 + drop_thresh)
            entry_window = px.iloc[start_idx + 1:end_idx + 1]
            ema21_window = ema21.iloc[start_idx + 1:end_idx + 1]
            dist_ema21_window = (entry_window - ema21_window) / ema21_window
            qualifying = entry_window[
                (entry_window <= threshold_price) & (dist_ema21_window <= ema21_dist_thresh)
            ]
            if qualifying.empty:
                continue

            entry_date = qualifying.index[0]
            entry_idx = int(px.index.get_indexer([entry_date])[0])
            if entry_idx < 0:
                continue
            if has_volume_spike(spike_prefix, start_idx, entry_idx):
                continue

            sma200_window = sma200.iloc[start_idx:entry_idx + 1]
            if sma200_window.isna().any() or pd.isna(sma200.iloc[entry_idx]):
                continue
            px_window = px.iloc[start_idx:entry_idx + 1]
            days_below_sma200 = int((px_window < sma200_window).sum())

            trading_days_in_period = end_idx - start_idx
            entry_offset_in_period = entry_idx - start_idx
            entry_price = float(px.iloc[entry_idx])
            rows.append({
                "ticker": tkr,
                "earnings_start": ev_date,
                "earnings_next": next_ev_date,
                "earn_gap_days": gap_days,
                "entry_date": entry_date,
                "entry_price": entry_price,
                "entry_drawdown": entry_price / float(anchor_price) - 1,
                "entry_trading_day": entry_offset_in_period,
                "period_trading_days": trading_days_in_period,
                "entry_pct": entry_offset_in_period / trading_days_in_period,
                "days_below_sma200": days_below_sma200,
                "_entry_idx": entry_idx,
                "_end_idx": end_idx,
                **_ma_fields(entry_price, mas, entry_date),
            })
    return rows


def apply_exit(entries, prepared, exit_offset):
    rows = []
    for e in entries:
        gap = e.get("earn_gap_days")
        if gap is None:
            gap = _calendar_gap_days(e["earnings_start"], e["earnings_next"])
        if gap < 1 or gap > MAX_EARNINGS_GAP_DAYS:
            continue
        px = prepared[e["ticker"]]["px"]
        exit_idx = e["_end_idx"] + exit_offset
        if exit_idx < 0 or exit_idx >= len(px) or exit_idx <= e["_entry_idx"]:
            continue
        exit_ts = px.index[exit_idx]
        exit_price = float(px.iloc[exit_idx])
        row = {k: v for k, v in e.items() if not k.startswith("_")}
        row.update({
            "holding_days": exit_idx - e["_entry_idx"],
            "calendar_hold_days": _calendar_gap_days(e["entry_date"], exit_ts),
            "earn_gap_days": int(gap),
            "exit_date": exit_ts,
            "exit_price": exit_price,
            "exit_offset": exit_offset,
            "outcome_return": exit_price / e["entry_price"] - 1,
        })
        rows.append(row)
    return rows


# ── trade autopsy ────────────────────────────────────────────────────────
def _numeric_summary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    stats = []
    for c in cols:
        s = df[c].dropna()
        if len(s) == 0:
            continue
        stats.append({
            "metric": c,
            "count": len(s),
            "mean": s.mean(),
            "median": s.median(),
            "std": s.std(),
            "p10": s.quantile(0.10),
            "p90": s.quantile(0.90),
            "min": s.min(),
            "max": s.max(),
        })
    return pd.DataFrame(stats)


def _above_below_summary(df: pd.DataFrame, col: str) -> dict:
    s = df[col]
    n = len(s)
    n_above = (s == True).sum()
    n_below = (s == False).sum()
    n_nan = s.isna().sum()
    return {
        "n": n,
        "n_above": int(n_above),
        "n_below": int(n_below),
        "n_nan": int(n_nan),
        "pct_above": n_above / n if n else np.nan,
        "pct_below": n_below / n if n else np.nan,
        "pct_nan": n_nan / n if n else np.nan,
    }


def run_trade_autopsy(trades_df: pd.DataFrame, out_dir: str, *, tag: str = "20pct_loose_sma200"):
    if trades_df.empty:
        print("\n[Autopsy] No trades to analyse.")
        return

    trades_df = trades_df.copy()
    n = len(trades_df)
    lo_thresh = trades_df["outcome_return"].quantile(0.10)
    hi_thresh = trades_df["outcome_return"].quantile(0.90)
    bottom10 = trades_df[trades_df["outcome_return"] <= lo_thresh].copy()
    top10 = trades_df[trades_df["outcome_return"] >= hi_thresh].copy()

    dist_cols_ema = ["dist_ema9_pct", "dist_ema21_pct", "dist_ema50_pct"]
    dist_cols_sma = ["dist_sma20_pct", "dist_sma50_pct", "dist_sma200_pct"]
    above_cols_ema = ["above_ema9", "above_ema21", "above_ema50"]
    above_cols_sma = ["above_sma20", "above_sma50", "above_sma200"]
    all_dist_cols = [c for c in dist_cols_ema + dist_cols_sma if c in trades_df.columns]

    W = 88
    print("\n" + "=" * W)
    print("TRADE AUTOPSY — EMA (Exponential) / SMA (Simple)  |  TOP 10% vs BOTTOM 10%")
    print("=" * W)
    print(f"Total trades: {n:,}  |  Decile cutoffs: bottom <= {lo_thresh:+.4f}  |  top >= {hi_thresh:+.4f}")
    print(f"  Bottom 10% n={len(bottom10):,}  |  Top 10% n={len(top10):,}")
    print(f"  Overall outcome — mean {trades_df['outcome_return'].mean():+.4f}  "
          f"median {trades_df['outcome_return'].median():+.4f}  "
          f"win rate {(trades_df['outcome_return'] > 0).mean():.1%}")
    print(f"\n  MA availability at entry (% non-NaN):")
    for c in all_dist_cols:
        cov_all = trades_df[c].notna().mean()
        cov_bot = bottom10[c].notna().mean() if len(bottom10) else np.nan
        cov_top = top10[c].notna().mean() if len(top10) else np.nan
        print(f"    {c:18s}  all {cov_all:.1%}  bottom10 {cov_bot:.1%}  top10 {cov_top:.1%}")

    print("\n" + "-" * W)
    print("1) ABOVE / BELOW moving average at entry  (share of trades)")
    print("-" * W)

    def _print_above_table(label, cols):
        print(f"\n  {label}:")
        rows_ab = []
        for c in cols:
            if c not in trades_df.columns:
                continue
            all_s = _above_below_summary(trades_df, c)
            bot_s = _above_below_summary(bottom10, c)
            top_s = _above_below_summary(top10, c)
            rows_ab.append({
                "ma": c,
                "all_above": all_s["pct_above"],
                "all_below": all_s["pct_below"],
                "bottom_above": bot_s["pct_above"],
                "top_above": top_s["pct_above"],
                "delta_top_minus_bottom": top_s["pct_above"] - bot_s["pct_above"],
                "all_n_nan": all_s["n_nan"],
            })
        ab_df = pd.DataFrame(rows_ab)
        if not ab_df.empty:
            print(ab_df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))
        return ab_df

    ab_ema = _print_above_table("EMA (Exponential Moving Average) — above = price > EMA", above_cols_ema)
    ab_sma = _print_above_table("SMA (Simple Moving Average) — above = price > SMA", above_cols_sma)

    print("\n" + "-" * W)
    print("2) DISTANCE from MA at entry  —  (price - MA) / MA  (negative = below MA)")
    print("   e.g. -0.05 = 5% below MA, +0.03 = 3% above MA")
    print("-" * W)

    dist_compare = []
    for c in all_dist_cols:
        bot = bottom10[c].dropna()
        top = top10[c].dropna()
        all_s = trades_df[c].dropna()
        if len(bot) == 0 or len(top) == 0:
            continue
        dist_compare.append({
            "metric": c,
            "all_mean": all_s.mean(),
            "all_median": all_s.median(),
            "bottom_mean": bot.mean(),
            "bottom_median": bot.median(),
            "bottom_std": bot.std(),
            "top_mean": top.mean(),
            "top_median": top.median(),
            "top_std": top.std(),
            "delta_mean (T-B)": top.mean() - bot.mean(),
            "delta_median (T-B)": top.median() - bot.median(),
            "bottom_p10": bot.quantile(0.10),
            "bottom_p90": bot.quantile(0.90),
            "top_p10": top.quantile(0.10),
            "top_p90": top.quantile(0.90),
        })
    dist_df = pd.DataFrame(dist_compare)
    if not dist_df.empty:
        print(dist_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"\n  Bottom 10% distance full summary:")
        print(_numeric_summary(bottom10, all_dist_cols).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"\n  Top 10% distance full summary:")
        print(_numeric_summary(top10, all_dist_cols).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"\n  All trades distance summary:")
        print(_numeric_summary(trades_df, all_dist_cols).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * W)
    print("3) AUTO-INSIGHTS — EMA/SMA patterns that separate top vs bottom decile")
    print("=" * W)
    insights = []

    for ab_df, label in [(ab_ema, "EMA"), (ab_sma, "SMA")]:
        if ab_df is None or ab_df.empty:
            continue
        for _, row in ab_df.iterrows():
            gap = row["delta_top_minus_bottom"]
            if pd.notna(gap) and abs(gap) >= 0.08:
                side = "TOP" if gap > 0 else "BOTTOM"
                insights.append(
                    f"  • {row['ma']} ({label}): {side} 10% more often ABOVE MA by {abs(gap):.1%} "
                    f"(top {row['top_above']:.1%} vs bottom {row['bottom_above']:.1%})"
                )

    for _, row in dist_df.iterrows():
        pooled_std = np.sqrt((row["bottom_std"]**2 + row["top_std"]**2) / 2) if pd.notna(row["bottom_std"]) and pd.notna(row["top_std"]) else np.nan
        delta = row["delta_mean (T-B)"]
        if pd.notna(delta) and pd.notna(pooled_std) and pooled_std != 0 and abs(delta) > 0.25 * pooled_std:
            direction = "CLOSER to / more ABOVE" if delta > 0 else "FURTHER BELOW"
            insights.append(
                f"  • {row['metric']}: top-10% is {direction} than bottom-10% by {delta:+.4f} "
                f"(mean) / {row['delta_median (T-B)']:+.4f} (median)  ~{abs(delta)/pooled_std:.2f} pooled-std  "
                f"[bottom {row['bottom_mean']:+.4f} vs top {row['top_mean']:+.4f}]"
            )

    corr_cols = [c for c in all_dist_cols if c in trades_df.columns]
    if corr_cols:
        corr = trades_df[corr_cols + ["outcome_return"]].corr(numeric_only=True)["outcome_return"].drop("outcome_return", errors="ignore").sort_values(key=lambda s: s.abs(), ascending=False)
        print("\n  Correlation of distance-from-MA with outcome_return (all trades):")
        for metric, c in corr.items():
            print(f"    {metric:18s}  r={c:+.4f}")
        for metric, c in corr.items():
            if abs(c) >= 0.05:
                insights.append(
                    f"  • Correlation: {metric} vs return  r={c:+.4f} ({'positive' if c > 0 else 'negative'}) — "
                    f"{'being further above MA correlates with better return' if c > 0 else 'being further below MA correlates with better return'}"
                )

    if insights:
        print("\n  Key MA patterns:")
        for line in insights:
            print(line)
    else:
        print("\n  No strong MA separation detected (|delta| < 0.25 pooled-std, |r| < 0.05, <8pp above/below gap).")

    print(f"\n  Outcome gap: bottom-10% mean {bottom10['outcome_return'].mean():+.4f}  "
          f"vs top-10% mean {top10['outcome_return'].mean():+.4f}  "
          f"(spread {top10['outcome_return'].mean() - bottom10['outcome_return'].mean():.4f})")
    print(f"              median {bottom10['outcome_return'].median():+.4f}  "
          f"vs median {top10['outcome_return'].median():+.4f}")

    if not dist_df.empty:
        dist_df.to_csv(os.path.join(out_dir, f"flow_b_{tag}_decile_distance.csv"), index=False)
    if ab_ema is not None and not ab_ema.empty:
        ab_ema.to_csv(os.path.join(out_dir, f"flow_b_{tag}_above_below_ema.csv"), index=False)
    if ab_sma is not None and not ab_sma.empty:
        ab_sma.to_csv(os.path.join(out_dir, f"flow_b_{tag}_above_below_sma.csv"), index=False)
    _numeric_summary(bottom10, all_dist_cols).to_csv(os.path.join(out_dir, f"flow_b_{tag}_bottom10_summary.csv"), index=False)
    _numeric_summary(top10, all_dist_cols).to_csv(os.path.join(out_dir, f"flow_b_{tag}_top10_summary.csv"), index=False)
    _numeric_summary(trades_df, all_dist_cols).to_csv(os.path.join(out_dir, f"flow_b_{tag}_all_summary.csv"), index=False)
    print("\n  Autopsy CSVs saved to", out_dir)
    print("=" * W)


def main():
    print("Loading ticker universe...")
    UNIVERSE = _load_r3000()
    print(f"Russell 3000 universe: {len(UNIVERSE)} tickers")

    print("Loading price + volume history (cached, downloading missing only)...")
    prices_all, volumes_all = _load_price_volume(UNIVERSE)
    priced = [t for t in UNIVERSE if t in prices_all.columns and prices_all[t].notna().any()]
    dropped = len(UNIVERSE) - len(priced)
    if dropped:
        print(f"  {len(priced)} tickers with price history ({dropped} dropped — no Yahoo data)")
    UNIVERSE = priced

    print("Loading earnings dates (cached, fetching missing only)...")
    earnings_dates = _load_earnings_dates(UNIVERSE)
    n_with_dates = sum(1 for v in earnings_dates.values() if len(v) >= 2)
    total_dates = sum(len(v) for v in earnings_dates.values())
    print(f"  {total_dates} earnings dates across {len(earnings_dates)} tickers "
          f"({n_with_dates} with 2+ dates)")

    print("Precomputing moving averages + volume-spike prefixes...")
    prepared = _prepare_universe(UNIVERSE, prices_all, volumes_all, earnings_dates, VOL_MULTS)

    entry_combos = list(itertools.product(DROPS, VOL_MULTS, EMA21_DIST_THRESH_GRID))
    results = []
    all_trades = []
    n_combos = (
        len(EARNINGS_EXIT_OFFSET_GRID) * len(entry_combos) * len(SMA200_MAX_DAYS_BELOW_GRID)
    )

    print(f"\nRunning {n_combos} parameter combos "
          f"({len(EARNINGS_EXIT_OFFSET_GRID)} exit offsets x {len(DROPS)} drops x "
          f"{len(VOL_MULTS)} vol mults x {len(EMA21_DIST_THRESH_GRID)} EMA21-distance thresholds x "
          f"{len(SMA200_MAX_DAYS_BELOW_GRID)} SMA200 day caps)\n")

    for drop, vol_mult, ema21_dist_thresh in entry_combos:
        entries = find_entries(prepared, drop, vol_mult, ema21_dist_thresh)
        for sma200_max_days in SMA200_MAX_DAYS_BELOW_GRID:
            filtered = [e for e in entries if e["days_below_sma200"] <= sma200_max_days]
            for exit_offset in EARNINGS_EXIT_OFFSET_GRID:
                events = apply_exit(filtered, prepared, exit_offset)
                n = len(events)
                if n == 0:
                    avg_ret = median_ret = win_rate = mean_entry_pct = median_entry_pct = np.nan
                else:
                    rets = np.array([e["outcome_return"] for e in events], dtype=float)
                    entry_pcts = np.array([e["entry_pct"] for e in events], dtype=float)
                    avg_ret = float(np.mean(rets))
                    median_ret = float(np.median(rets))
                    win_rate = float(np.mean(rets > 0))
                    mean_entry_pct = float(np.mean(entry_pcts))
                    median_entry_pct = float(np.median(entry_pcts))
                    for e in events:
                        all_trades.append({
                            **e,
                            "drop_pct": f"{drop:.0%}",
                            "vol_mult": vol_mult,
                            "ema21_dist_thresh": ema21_dist_thresh,
                            "sma200_max_days": sma200_max_days,
                        })
                results.append({
                    "exit_offset": exit_offset,
                    "drop_pct": f"{drop:.0%}",
                    "vol_mult": vol_mult,
                    "ema21_dist_thresh": ema21_dist_thresh,
                    "sma200_max_days": sma200_max_days,
                    "n_trades": n,
                    "avg_return": avg_ret,
                    "median_return": median_ret,
                    "win_rate": win_rate,
                    "mean_entry_pct": mean_entry_pct,
                    "median_entry_pct": median_entry_pct,
                })

    df = pd.DataFrame(results).sort_values(
        ["sma200_max_days", "exit_offset", "vol_mult", "ema21_dist_thresh", "drop_pct"]
    )
    drop_label = ", ".join(f"{x:.0%}" for x in DROPS)
    ema_label = ", ".join(f"{x:.0%}" for x in EMA21_DIST_THRESH_GRID)
    sma_label = ", ".join(str(x) for x in SMA200_MAX_DAYS_BELOW_GRID)

    df["balance"] = df["median_return"] * np.sqrt(df["n_trades"].clip(lower=0))

    print("=" * 80)
    print("OPTIMIZER — max mean return subject to n_trades > 500 | exit t+3 fixed")
    print(f"Universe: {len(prepared)} tickers | drops: {drop_label} | "
          f"vol: {VOL_MULTS} | EMA21: {ema_label} | SMA200 max days below: {sma_label}")
    print("=" * 80)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "nan"))
    print("=" * 80)

    feasible = df[df["n_trades"] > MIN_TRADES].copy()
    print(f"\nFEASIBLE cells (n_trades > {MIN_TRADES}): {len(feasible)} / {len(df)}")
    if feasible.empty:
        print("No combo clears n > 500.")
    else:
        ranked = feasible.sort_values("avg_return", ascending=False)
        print("\nRANKED by mean return (n > 500):")
        print(ranked.to_string(index=False, float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "nan"))
        best = ranked.iloc[0]
        print("\n" + "=" * 80)
        print("WINNER — highest mean return, n > 500")
        print(f"  drop {best['drop_pct']}  vol {best['vol_mult']:.0f}x  "
              f"EMA21 {best['ema21_dist_thresh']:.0%}  SMA200 {int(best['sma200_max_days'])}d  t+3")
        print(f"  n={int(best['n_trades'])}  mean={best['avg_return']:+.2%}  "
              f"median={best['median_return']:+.2%}  win={best['win_rate']:.1%}  "
              f"balance(median*sqrt(n))={best['balance']:.4f}")
        print("=" * 80)
        bal = feasible.sort_values("balance", ascending=False).iloc[0]
        if not (
            bal["drop_pct"] == best["drop_pct"]
            and bal["vol_mult"] == best["vol_mult"]
            and bal["ema21_dist_thresh"] == best["ema21_dist_thresh"]
            and bal["sma200_max_days"] == best["sma200_max_days"]
        ):
            print("\nMost balanced among n>500 (median * sqrt(n)), different cell:")
            print(f"  drop {bal['drop_pct']}  vol {bal['vol_mult']:.0f}x  "
                  f"EMA21 {bal['ema21_dist_thresh']:.0%}  SMA200 {int(bal['sma200_max_days'])}d  t+3")
            print(f"  n={int(bal['n_trades'])}  mean={bal['avg_return']:+.2%}  "
                  f"median={bal['median_return']:+.2%}  win={bal['win_rate']:.1%}  "
                  f"balance={bal['balance']:.4f}")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "flow_b_sweep_opt_t3_n500.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSweep results saved to {out_path}")

    trades_df = pd.DataFrame(all_trades)
    trades_out = os.path.join(out_dir, "flow_b_trades_opt_t3_n500.csv")
    if not trades_df.empty:
        trades_df.to_csv(trades_out, index=False)
        print(f"Individual trades saved to {trades_out}  ({len(trades_df):,} rows)")
    else:
        print(f"No trades generated — skipping {trades_out}")


if __name__ == "__main__":
    main()
