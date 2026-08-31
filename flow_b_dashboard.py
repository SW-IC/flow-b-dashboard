"""
Flow B parameter tuner.

Loads prices / earnings / R3000 from flow_b_cache only. Never hits Yahoo.
First run builds a prepared-universe pickle (MAs); later runs reuse it.

Launch:
    streamlit run flow_b_dashboard.py
"""
from __future__ import annotations

import importlib.util
import itertools
import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

SCRIPT_DIR = Path(__file__).resolve().parent
_ENGINE_CANDIDATES = [
    SCRIPT_DIR / "flow_b_engine.py",
    SCRIPT_DIR / "flow_b_ema21_entry_sweep (9).py",
]
ENGINE_PATH = next((p for p in _ENGINE_CANDIDATES if p.exists()), _ENGINE_CANDIDATES[0])
PREPARED_NAME = "prepared_dashboard"
PREPARED_VERSION = 1

DEFAULTS = {
    "drop_pct": 25,          # stored as positive %; engine uses negative
    "vol_mult": 7.0,
    "ema21_pct": 18,         # % below EMA21
    "sma200_max_days": 25,
    "exit_offset": 3,
    "min_entry_price": 0.0,
}


def _load_engine():
    if not ENGINE_PATH.exists():
        raise FileNotFoundError(f"Sweep engine not found: {ENGINE_PATH}")
    spec = importlib.util.spec_from_file_location("flow_b_engine", ENGINE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _engine_stamp() -> str:
    stt = ENGINE_PATH.stat()
    return f"{stt.st_mtime_ns}:{stt.st_size}"


@st.cache_resource(show_spinner="Loading engine…")
def get_engine(stamp: str):
    return _load_engine()


def _prepared_fingerprint(engine) -> str:
    prices_fp = engine._cache_fingerprint("prices_close")
    earn_fp = engine._cache_fingerprint("earnings_dates")
    r3000_fp = engine._cache_fingerprint("r3000")
    return json.dumps(
        {
            "version": PREPARED_VERSION,
            "prices": prices_fp,
            "earnings": earn_fp,
            "r3000": r3000_fp,
        }
    )


def _cache_series_to_list(cached) -> list[str]:
    if cached is None:
        return []
    if isinstance(cached, pd.Series):
        return [str(x) for x in cached.tolist()]
    if isinstance(cached, (list, tuple)):
        return [str(x) for x in cached]
    return []


def _earnings_to_dict(cached) -> dict:
    if cached is None:
        return {}
    if isinstance(cached, pd.Series):
        raw = cached.to_dict()
    elif isinstance(cached, dict):
        raw = cached
    else:
        return {}
    return {str(k): (list(v) if v is not None else []) for k, v in raw.items()}


def load_cache_only(engine):
    """Read pickle cache. Raise if anything required is missing. No Yahoo."""
    missing = []
    r3000 = engine._load_cache("r3000")
    prices = engine._load_cache("prices_close")
    volumes = engine._load_cache("prices_volume")
    earnings = engine._load_cache("earnings_dates")
    if r3000 is None:
        missing.append("r3000.pkl")
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        missing.append("prices_close.pkl")
    if not isinstance(volumes, pd.DataFrame) or volumes.empty:
        missing.append("prices_volume.pkl")
    if earnings is None:
        missing.append("earnings_dates.pkl")
    if missing:
        raise FileNotFoundError(
            "Cache missing: "
            + ", ".join(missing)
            + f". Run the sweep script once to fill {engine.CACHE_DIR}"
        )
    universe = engine._dedupe_tickers(_cache_series_to_list(r3000))
    priced = [t for t in universe if t in prices.columns and prices[t].notna().any()]
    earnings_dict = _earnings_to_dict(earnings)
    meta = {
        "cache_dir": str(engine.CACHE_DIR),
        "r3000_fp": engine._cache_fingerprint("r3000") or "unknown",
        "prices_fp": engine._cache_fingerprint("prices_close") or "unknown",
        "earnings_fp": engine._cache_fingerprint("earnings_dates") or "unknown",
        "n_universe": len(universe),
        "n_priced": len(priced),
        "n_price_cols": int(prices.shape[1]),
        "price_rows": int(prices.shape[0]),
    }
    return priced, prices, volumes, earnings_dict, meta


def prepare_universe(engine, tickers, prices_all, volumes_all, earnings_dict):
    """Same as engine._prepare_universe, but keep volume so any vol_mult works."""
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
            [
                engine._align_naive(prices_all[tkr]).rename("price"),
                engine._align_naive(volumes_all[tkr]).rename("volume"),
            ],
            axis=1,
        ).dropna()
        if market.empty:
            skipped["no_price"] += 1
            continue
        px = market["price"]
        prepared[tkr] = {
            "px": px,
            "mas": engine._compute_mas(px),
            "spike": {},
            "volume": market["volume"],
            "dates": dates,
        }
    return prepared, skipped


def ensure_spike(engine, prepared, vol_mult: float):
    for data in prepared.values():
        if vol_mult not in data["spike"]:
            data["spike"][vol_mult] = engine.volume_spike_prefix(data["volume"], vol_mult)


@st.cache_resource(show_spinner="Loading cache + building MAs (first time only)…")
def get_prepared(stamp: str):
    engine = get_engine(stamp)
    tickers, prices, volumes, earnings_dict, meta = load_cache_only(engine)
    fp = _prepared_fingerprint(engine)
    cached = engine._load_cache(PREPARED_NAME)
    cached_fp = engine._cache_fingerprint(PREPARED_NAME)
    if isinstance(cached, dict) and cached_fp == fp and "prepared" in cached:
        prepared = cached["prepared"]
        skipped = cached.get("skipped", {})
        meta = {**meta, **cached.get("meta", {}), "prepared_source": "pickle"}
        return engine, prepared, meta, skipped

    prepared, skipped = prepare_universe(engine, tickers, prices, volumes, earnings_dict)
    payload = {"prepared": prepared, "skipped": skipped, "meta": meta}
    engine._save_cache(PREPARED_NAME, payload, fingerprint=fp)
    meta = {**meta, "prepared_source": "built", "n_prepared": len(prepared)}
    return engine, prepared, meta, skipped


def summarize(trades: list[dict]) -> dict:
    if not trades:
        return {
            "n": 0,
            "n_unique_entries": 0,
            "n_tickers": 0,
            "mean": np.nan,
            "median": np.nan,
            "win": np.nan,
            "gap": np.nan,
            "p10": np.nan,
            "p90": np.nan,
            "min": np.nan,
            "max": np.nan,
            "skew": np.nan,
            "mean_hold": np.nan,
            "median_hold": np.nan,
            "max_hold": np.nan,
            "max_cal_hold": np.nan,
            "max_earn_gap": np.nan,
            "mean_entry_pct": np.nan,
            "n_sub5": 0,
            "pct_sub5": np.nan,
            "n_doubles": 0,
        }
    df = pd.DataFrame(trades)
    rets = df["outcome_return"].astype(float)
    unique = df.drop_duplicates(["ticker", "entry_date"])
    n_sub5 = int((df["entry_price"] < 5).sum())
    return {
        "n": int(len(df)),
        "n_unique_entries": int(len(unique)),
        "n_tickers": int(df["ticker"].nunique()),
        "mean": float(rets.mean()),
        "median": float(rets.median()),
        "win": float((rets > 0).mean()),
        "gap": float(rets.mean() - rets.median()),
        "p10": float(rets.quantile(0.10)),
        "p90": float(rets.quantile(0.90)),
        "min": float(rets.min()),
        "max": float(rets.max()),
        "skew": float(rets.skew()),
        "mean_hold": float(df["holding_days"].mean()),
        "median_hold": float(df["holding_days"].median()),
        "max_hold": int(df["holding_days"].max()),
        "max_cal_hold": int(df["calendar_hold_days"].max())
        if "calendar_hold_days" in df.columns
        else np.nan,
        "max_earn_gap": int(df["earn_gap_days"].max())
        if "earn_gap_days" in df.columns
        else np.nan,
        "mean_entry_pct": float(df["entry_pct"].mean()),
        "n_sub5": n_sub5,
        "pct_sub5": n_sub5 / len(df),
        "n_doubles": int((rets >= 1.0).sum()),
    }


def year_mix(trades: list[dict]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame(trades)
    df["year"] = pd.to_datetime(df["entry_date"]).dt.year
    g = df.groupby("year")["outcome_return"].agg(
        n="count", mean="mean", median="median", win=lambda s: (s > 0).mean()
    )
    return g.reset_index()


def run_combo(
    engine,
    prepared,
    *,
    drop: float,
    vol_mult: float,
    ema21_dist: float,
    sma200_max_days: int,
    exit_offset: int,
    min_entry_price: float,
):
    vol_mult = float(vol_mult)
    ensure_spike(engine, prepared, vol_mult)
    entries = engine.find_entries(prepared, drop, vol_mult, ema21_dist)
    entries = [e for e in entries if e["days_below_sma200"] <= sma200_max_days]
    if min_entry_price > 0:
        entries = [e for e in entries if e["entry_price"] >= min_entry_price]
    trades = engine.apply_exit(entries, prepared, int(exit_offset))
    stats = summarize(trades)
    stats.update(
        {
            "drop": drop,
            "vol_mult": vol_mult,
            "ema21_dist": ema21_dist,
            "sma200_max_days": int(sma200_max_days),
            "exit_offset": int(exit_offset),
            "min_entry_price": float(min_entry_price),
        }
    )
    return trades, stats


SWEEP_PARAMS = [
    {
        "key": "drop_pct",
        "label": "Drop",
        "unit": "% below earnings close",
        "kind": "int",
        "const_default": 25,
        "mode_default": "Sweep",
        "sweep_default": "20, 25",
        "min": 10,
        "max": 40,
        "step": 1,
        "engine_key": "drop",
        "to_engine": lambda xs: [-float(x) / 100.0 for x in xs],
    },
    {
        "key": "vol_mult",
        "label": "Vol spike",
        "unit": "× prior 20d avg",
        "kind": "float",
        "const_default": 7.0,
        "mode_default": "Constant",
        "sweep_default": "7, 10",
        "min": 3.0,
        "max": 15.0,
        "step": 0.5,
        "engine_key": "vol_mult",
        "to_engine": lambda xs: [float(x) for x in xs],
    },
    {
        "key": "ema21_pct",
        "label": "EMA21",
        "unit": "% below",
        "kind": "int",
        "const_default": 18,
        "mode_default": "Sweep",
        "sweep_default": "13, 15, 18",
        "min": 5,
        "max": 30,
        "step": 1,
        "engine_key": "ema21_dist",
        "to_engine": lambda xs: [-float(x) / 100.0 for x in xs],
    },
    {
        "key": "sma200_max_days",
        "label": "SMA200",
        "unit": "max days below",
        "kind": "int",
        "const_default": 25,
        "mode_default": "Sweep",
        "sweep_default": "10, 20, 25",
        "min": 0,
        "max": 60,
        "step": 1,
        "engine_key": "sma200_max_days",
        "to_engine": lambda xs: [int(x) for x in xs],
    },
    {
        "key": "exit_offset",
        "label": "Exit",
        "unit": "sessions after next print",
        "kind": "int",
        "const_default": 3,
        "mode_default": "Constant",
        "sweep_default": "0, 3",
        "min": 0,
        "max": 10,
        "step": 1,
        "engine_key": "exit_offset",
        "to_engine": lambda xs: [int(x) for x in xs],
    },
    {
        "key": "min_entry_price",
        "label": "Px floor",
        "unit": "$ (0 = off)",
        "kind": "float",
        "const_default": 0.0,
        "mode_default": "Constant",
        "sweep_default": "0, 5",
        "min": 0.0,
        "max": 50.0,
        "step": 1.0,
        "engine_key": "min_entry_price",
        "to_engine": lambda xs: [float(x) for x in xs],
    },
]

MAX_SWEEP_COMBOS = 64
MAX_ENTRY_PASSES = 16
SEC_PER_ENTRY_PASS = 12.5
MAX_VALUES_PER_PARAM = 8


def parse_num_list(text: str, spec: dict) -> list:
    raw = (text or "").replace(";", ",").replace("|", ",")
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError(f"{spec['label']}: empty list")
    out = []
    seen = set()
    for p in parts:
        try:
            val = int(float(p)) if spec["kind"] == "int" else float(p)
        except ValueError as e:
            raise ValueError(f"{spec['label']}: cannot parse {p!r}") from e
        lo, hi = spec["min"], spec["max"]
        if val < lo or val > hi:
            raise ValueError(f"{spec['label']}: {val} outside {lo}–{hi}")
        if val in seen:
            continue
        seen.add(val)
        out.append(val)
    if len(out) > MAX_VALUES_PER_PARAM:
        raise ValueError(f"{spec['label']}: max {MAX_VALUES_PER_PARAM} values")
    return out


def collect_sweep_grid(modes: dict, const_vals: dict, sweep_text: dict):
    """Return (engine_value_lists, display_lists, error) keyed by engine_key."""
    display = {}
    engine_lists = {}
    for spec in SWEEP_PARAMS:
        key = spec["key"]
        ek = spec["engine_key"]
        if modes[key] == "Constant":
            raw = [const_vals[key]]
        else:
            try:
                raw = parse_num_list(sweep_text[key], spec)
            except ValueError as e:
                return None, None, str(e)
        display[ek] = raw
        engine_lists[ek] = spec["to_engine"](raw)
    keys = [s["engine_key"] for s in SWEEP_PARAMS]
    combos = []
    for tup in itertools.product(*[engine_lists[k] for k in keys]):
        combos.append(dict(zip(keys, tup)))
    n_entry = len({(c["drop"], c["vol_mult"], c["ema21_dist"]) for c in combos})
    return combos, {"n_entry": n_entry, "display": display}, None


def run_sweep_grid(engine, prepared, combos, progress=None):
    groups = {}
    for c in combos:
        k = (c["drop"], c["vol_mult"], c["ema21_dist"])
        groups.setdefault(k, []).append(c)

    results = []
    n_groups = len(groups)
    for i, ((drop, vol, ema), subset) in enumerate(groups.items(), start=1):
        if progress:
            progress(i - 1, n_groups, drop, vol, ema)
        ensure_spike(engine, prepared, float(vol))
        entries = engine.find_entries(prepared, drop, float(vol), ema)
        for c in subset:
            filtered = [e for e in entries if e["days_below_sma200"] <= c["sma200_max_days"]]
            if c["min_entry_price"] > 0:
                filtered = [e for e in filtered if e["entry_price"] >= c["min_entry_price"]]
            trades = engine.apply_exit(filtered, prepared, int(c["exit_offset"]))
            stats = summarize(trades)
            stats.update(c)
            n = stats["n"]
            med = stats["median"]
            stats["balance"] = float(med * np.sqrt(n)) if n and pd.notna(med) else np.nan
            results.append(stats)
        if progress:
            progress(i, n_groups, drop, vol, ema)
    return results


def sweep_results_df(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    cols = [
        "drop",
        "vol_mult",
        "ema21_dist",
        "sma200_max_days",
        "exit_offset",
        "min_entry_price",
        "n",
        "mean",
        "median",
        "win",
        "gap",
        "skew",
        "balance",
        "n_tickers",
        "n_sub5",
        "max_hold",
        "max_earn_gap",
        "max",
        "min",
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols]


def pick_winner(df: pd.DataFrame, rank_by: str, min_n: int) -> pd.Series | None:
    if df.empty:
        return None
    feasible = df[df["n"] > min_n] if min_n else df
    pool = feasible if not feasible.empty else df
    col = {"Mean": "mean", "Median": "median", "Balance": "balance", "Win": "win", "n": "n"}[rank_by]
    return pool.sort_values(col, ascending=False).iloc[0]


def _fmt_pct(x) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    return f"{x:+.2%}"


def _fmt_num(x, nd=1) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    return f"{x:.{nd}f}"


def _combo_label(s: dict) -> str:
    return (
        f"drop {s['drop']:.0%}  vol {s['vol_mult']:.1f}x  "
        f"EMA21 {s['ema21_dist']:.0%}  SMA{int(s['sma200_max_days'])}d  "
        f"t+{int(s['exit_offset'])}  px≥${s['min_entry_price']:.0f}"
    )


CSS = """
<style>
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }
.block-container { padding-top: 1.75rem !important; padding-bottom: 2rem; }
h1 { letter-spacing: -0.03em; line-height: 1.25 !important; padding-top: 0.1rem; }
[data-testid="stCaptionContainer"] { margin-top: -0.55rem; opacity: 0.72; }
.fb-pills { display: flex; flex-wrap: wrap; gap: 7px; margin: 4px 0 16px; }
.fb-pill {
  background: var(--secondary-background-color, rgba(127,127,127,.12));
  border: 1px solid rgba(127,127,127,.16);
  border-radius: 999px;
  padding: 3px 11px;
  font-size: 0.84rem;
  font-variant-numeric: tabular-nums;
  line-height: 1.65;
}
.fb-pill i {
  font-style: normal; opacity: 0.45; margin-right: 6px;
  font-size: 0.68rem; letter-spacing: 0.05em; text-transform: uppercase;
}
.fb-dirty { font-size: 0.8rem; opacity: 0.5; margin: -8px 0 14px; }
.fb-hero, .fb-fine { display: grid; gap: 10px; margin-bottom: 10px; }
.fb-hero { grid-template-columns: repeat(4, minmax(0, 1fr)); }
.fb-fine { grid-template-columns: repeat(8, minmax(0, 1fr)); }
.fb-card {
  background: var(--secondary-background-color, rgba(127,127,127,.1));
  border: 1px solid rgba(127,127,127,.14);
  border-radius: 12px;
  padding: 12px 14px 11px;
  min-width: 0;
}
.fb-lbl {
  font-size: 0.7rem; opacity: 0.48; letter-spacing: 0.05em;
  text-transform: uppercase;
}
.fb-val {
  font-size: 1.55rem; font-weight: 650; letter-spacing: -0.03em;
  font-variant-numeric: tabular-nums; margin-top: 3px; line-height: 1.2;
}
.fb-fine .fb-val { font-size: 1.02rem; font-weight: 600; letter-spacing: -0.02em; }
.fb-sub { font-size: 0.75rem; opacity: 0.48; margin-top: 5px; }
.fb-val.pos { color: #3ecf8e; }
.fb-val.neg { color: #f07178; }
.fb-note { font-size: 0.84rem; opacity: 0.55; margin: 2px 0 12px; }
.fb-win { font-size: 0.9rem; opacity: 0.7; margin: 4px 0 12px; }
@media (max-width: 1200px) {
  .fb-hero { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .fb-fine { grid-template-columns: repeat(4, minmax(0, 1fr)); }
}
@media (max-width: 700px) {
  .fb-fine { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
</style>
"""


def _cls(x) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return ""
    if x > 0:
        return " pos"
    if x < 0:
        return " neg"
    return ""


def _card(label: str, value: str, sub: str = "", cls: str = "") -> str:
    sub_html = f'<div class="fb-sub">{escape(sub)}</div>' if sub else ""
    return (
        f'<div class="fb-card"><div class="fb-lbl">{escape(label)}</div>'
        f'<div class="fb-val{cls}">{escape(value)}</div>{sub_html}</div>'
    )


def _pills_html(drop, vol_mult, ema21_dist, sma200_max_days, exit_offset, min_entry_price) -> str:
    px = "off" if min_entry_price <= 0 else f"${min_entry_price:.0f}+"
    items = [
        ("Drop", f"{drop:.0%}"),
        ("Vol", f"{vol_mult:.1f}×"),
        ("EMA21", f"{ema21_dist:.0%}"),
        ("SMA200", f"≤{int(sma200_max_days)}d"),
        ("Exit", f"t+{int(exit_offset)}"),
        ("Floor", px),
    ]
    inner = "".join(
        f'<span class="fb-pill"><i>{escape(a)}</i>{escape(b)}</span>' for a, b in items
    )
    return f'<div class="fb-pills">{inner}</div>'


def _hold_sub(stats: dict) -> str:
    maxh = stats.get("max_hold")
    maxg = stats.get("max_earn_gap")
    bits = ["median sessions"]
    if maxh is not None and pd.notna(maxh):
        bits.append(f"max {int(maxh)}")
    if maxg is not None and pd.notna(maxg):
        bits.append(f"print gap ≤{int(maxg)}d")
    return " · ".join(bits)


def _results_html(stats: dict) -> str:
    win = f"{stats['win']:.1%}" if pd.notna(stats["win"]) else "—"
    sub5 = (
        f"{stats['n_sub5']:,} ({stats['pct_sub5']:.0%})"
        if pd.notna(stats["pct_sub5"])
        else "—"
    )
    hold = _fmt_num(stats["median_hold"], 0)
    hero = (
        _card("Trades", f"{stats['n']:,}", f"{stats['n_tickers']:,} tickers")
        + _card("Mean", _fmt_pct(stats["mean"]), cls=_cls(stats["mean"]))
        + _card("Median", _fmt_pct(stats["median"]), cls=_cls(stats["median"]))
        + _card("Win", win, f"{stats['n_doubles']:,} doubles")
    )
    fine = (
        _card("Gap", _fmt_pct(stats["gap"]), "mean − median", cls=_cls(stats["gap"]))
        + _card("Skew", _fmt_num(stats["skew"], 2))
        + _card("P10", _fmt_pct(stats["p10"]), cls=_cls(stats["p10"]))
        + _card("P90", _fmt_pct(stats["p90"]), cls=_cls(stats["p90"]))
        + _card("Min", _fmt_pct(stats["min"]), cls=_cls(stats["min"]))
        + _card("Max", _fmt_pct(stats["max"]), cls=_cls(stats["max"]))
        + _card(
            "Hold",
            f"{hold} sess" if hold != "—" else "—",
            _hold_sub(stats),
        )
        + _card("Sub-$5", sub5)
    )
    return f'<div class="fb-hero">{hero}</div><div class="fb-fine">{fine}</div>'


def cache_caption(meta, prepared, skipped):
    st.caption(
        f"Cache {meta.get('cache_dir')}\n\n"
        f"R3000 {meta.get('r3000_fp')} · prices {meta.get('prices_fp')} · "
        f"earnings {meta.get('earnings_fp')}\n\n"
        f"{len(prepared):,} prepared tickers  "
        f"(no px {skipped.get('no_price', 0)}, <2 earnings {skipped.get('no_earnings', 0)})\n\n"
        f"prepared via {meta.get('prepared_source', '?')}"
    )


def _apply_sweep_preset(name: str):
    locked_const = {
        "drop_pct": 25,
        "vol_mult": 7.0,
        "ema21_pct": 18,
        "sma200_max_days": 25,
        "exit_offset": 3,
        "min_entry_price": 0.0,
    }
    if name == "opt":
        modes = {
            "drop_pct": "Sweep",
            "vol_mult": "Constant",
            "ema21_pct": "Sweep",
            "sma200_max_days": "Sweep",
            "exit_offset": "Constant",
            "min_entry_price": "Constant",
        }
        lists = {
            "drop_pct": "20, 25",
            "vol_mult": "7, 10",
            "ema21_pct": "13, 15, 18",
            "sma200_max_days": "10, 20, 25",
            "exit_offset": "0, 3",
            "min_entry_price": "0, 5",
        }
    else:
        modes = {s["key"]: "Constant" for s in SWEEP_PARAMS}
        lists = {s["key"]: s["sweep_default"] for s in SWEEP_PARAMS}
    for k, v in modes.items():
        st.session_state[f"sw_mode_{k}"] = v
    for k, v in lists.items():
        st.session_state[f"sw_list_{k}"] = v
    for k, v in locked_const.items():
        st.session_state[f"sw_const_{k}"] = v


def render_trades(trades, *, download_name: str):
    tdf = pd.DataFrame(trades)
    show_cols = [
        "ticker",
        "earnings_start",
        "earnings_next",
        "earn_gap_days",
        "entry_date",
        "exit_date",
        "holding_days",
        "calendar_hold_days",
        "entry_price",
        "exit_price",
        "outcome_return",
        "entry_drawdown",
        "dist_ema21_pct",
        "days_below_sma200",
        "entry_pct",
    ]
    show_cols = [c for c in show_cols if c in tdf.columns]
    view = tdf[show_cols].copy()
    for col in ("entry_date", "exit_date", "earnings_start", "earnings_next"):
        if col in view.columns:
            view[col] = pd.to_datetime(view[col]).dt.strftime("%Y-%m-%d")

    left, right = st.columns(2)
    with left:
        st.markdown("**Year mix**")
        ym = year_mix(trades)
        if not ym.empty:
            ym_fmt = ym.copy()
            for col in ("mean", "median", "win"):
                ym_fmt[col] = ym_fmt[col].map(_fmt_pct)
            st.dataframe(ym_fmt, hide_index=True, use_container_width=True)
    with right:
        st.markdown("**Return buckets**")
        bins = [-1, -0.5, -0.2, 0, 0.2, 0.5, 1, 10]
        labels = ["≤−50%", "−50/−20", "−20/0", "0/+20", "+20/+50", "+50/+100", ">+100%"]
        tdf["bucket"] = pd.cut(tdf["outcome_return"], bins=bins, labels=labels, right=True)
        hist = tdf["bucket"].value_counts().reindex(labels).fillna(0).astype(int)
        st.bar_chart(hist)

    trade_cols = {
        "earnings_start": st.column_config.TextColumn("Print 1"),
        "earnings_next": st.column_config.TextColumn("Print 2"),
        "earn_gap_days": st.column_config.NumberColumn("Print gap", help="Calendar days between the two earnings prints. Cap is 140 (Yahoo hole, not one quarter)."),
        "holding_days": st.column_config.NumberColumn("Sessions", help="Trading bars from entry to next-earnings + t+N. A full quarter is often 40–90."),
        "calendar_hold_days": st.column_config.NumberColumn("Cal days", help="Calendar days from entry to exit. Q4 (Nov→Mar) + t+3 can reach ~140."),
        "outcome_return": st.column_config.NumberColumn("Return", format="+0.0%"),
        "entry_drawdown": st.column_config.NumberColumn("Drawdown", format="+0.0%"),
        "dist_ema21_pct": st.column_config.NumberColumn("EMA21 dist", format="+0.0%"),
        "entry_pct": st.column_config.NumberColumn("Entry % of window", format="0.0%"),
    }

    st.caption(
        "Exit = next earnings print + t+N (one quarter). Sessions = trading bars. "
        "Yahoo consecutive dates more than 140 calendar days apart are skipped (missing print, not a long quarter)."
    )

    st.markdown("**Worst 10 / best 10**")
    ranked = view.sort_values("outcome_return")
    b1, b2 = st.columns(2)
    with b1:
        st.caption("Worst")
        st.dataframe(ranked.head(10), hide_index=True, use_container_width=True, column_config=trade_cols)
    with b2:
        st.caption("Best")
        st.dataframe(ranked.tail(10).iloc[::-1], hide_index=True, use_container_width=True, column_config=trade_cols)

    with st.expander(f"All {len(view):,} trades"):
        st.dataframe(
            view.sort_values("outcome_return", ascending=False),
            hide_index=True,
            use_container_width=True,
            column_config=trade_cols,
        )

    csv = tdf.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download trades CSV",
        data=csv,
        file_name=download_name,
        mime="text/csv",
    )


def render_tuner(engine, prepared, meta, skipped):
    if "history" not in st.session_state:
        st.session_state.history = []

    with st.sidebar:
        st.header("Parameters")
        if st.button("Reset to default combo", use_container_width=True):
            for k, v in DEFAULTS.items():
                st.session_state[k] = v
            st.rerun()

        drop_pct = st.slider(
            "Drawdown from earnings close (%)",
            min_value=10,
            max_value=40,
            step=1,
            value=DEFAULTS["drop_pct"],
            key="drop_pct",
            help="First close ≤ this % below the post-print anchor AND far enough below EMA21.",
        )
        vol_mult = st.slider(
            "Volume-spike veto (× 20-day avg)",
            min_value=3.0,
            max_value=15.0,
            step=0.5,
            value=DEFAULTS["vol_mult"],
            key="vol_mult",
            help="Reject if any day from anchor through entry has volume > this × prior 20-day avg.",
        )
        ema21_pct = st.slider(
            "EMA21 distance required (% below)",
            min_value=5,
            max_value=30,
            step=1,
            value=DEFAULTS["ema21_pct"],
            key="ema21_pct",
            help="Entry close must be this far below EMA21. Bigger number = stricter.",
        )
        sma200_max_days = st.slider(
            "Max days below SMA200 (anchor → entry)",
            min_value=0,
            max_value=60,
            step=1,
            value=DEFAULTS["sma200_max_days"],
            key="sma200_max_days",
            help="Skip until SMA200 exists. Then reject if price spent more than N days below it.",
        )
        exit_offset = st.slider(
            "Exit offset (sessions after next-earnings priced bar)",
            min_value=0,
            max_value=10,
            step=1,
            value=DEFAULTS["exit_offset"],
            key="exit_offset",
            help="t+0 = first session that trades the next print. t+3 = three sessions later.",
        )
        min_entry_price = st.number_input(
            "Min entry price ($). 0 = no floor",
            min_value=0.0,
            max_value=50.0,
            step=1.0,
            value=DEFAULTS["min_entry_price"],
            key="min_entry_price",
        )
        run = st.button("Run", type="primary", use_container_width=True)

        st.divider()
        cache_caption(meta, prepared, skipped)

    drop = -drop_pct / 100.0
    ema21_dist = -ema21_pct / 100.0

    if run:
        with st.spinner("Scoring universe…"):
            trades, stats = run_combo(
                engine,
                prepared,
                drop=drop,
                vol_mult=float(vol_mult),
                ema21_dist=ema21_dist,
                sma200_max_days=int(sma200_max_days),
                exit_offset=int(exit_offset),
                min_entry_price=float(min_entry_price),
            )
        st.session_state.last_trades = trades
        st.session_state.last_stats = stats
        hist_row = {k: stats[k] for k in stats}
        st.session_state.history.insert(0, hist_row)

    stats = st.session_state.get("last_stats")
    trades = st.session_state.get("last_trades")

    pill_src = stats if stats is not None else {
        "drop": drop,
        "vol_mult": float(vol_mult),
        "ema21_dist": ema21_dist,
        "sma200_max_days": int(sma200_max_days),
        "exit_offset": int(exit_offset),
        "min_entry_price": float(min_entry_price),
    }
    st.markdown(
        _pills_html(
            pill_src["drop"],
            pill_src["vol_mult"],
            pill_src["ema21_dist"],
            pill_src["sma200_max_days"],
            pill_src["exit_offset"],
            pill_src["min_entry_price"],
        ),
        unsafe_allow_html=True,
    )

    if stats is None:
        st.info("Hit Run. First click after a cache rebuild is slower; after that, a few seconds.")
        return

    dirty = not (
        abs(drop - stats["drop"]) < 1e-9
        and abs(float(vol_mult) - stats["vol_mult"]) < 1e-9
        and abs(ema21_dist - stats["ema21_dist"]) < 1e-9
        and int(sma200_max_days) == int(stats["sma200_max_days"])
        and int(exit_offset) == int(stats["exit_offset"])
        and abs(float(min_entry_price) - stats["min_entry_price"]) < 1e-9
    )
    if dirty:
        st.markdown(
            '<div class="fb-dirty">Sliders moved — showing last run. Hit Run to refresh.</div>',
            unsafe_allow_html=True,
        )

    st.markdown(_results_html(stats), unsafe_allow_html=True)

    if trades:
        render_trades(trades, download_name="flow_b_dashboard_trades.csv")

    if st.session_state.history:
        st.markdown("**This-session runs**")
        hdf = pd.DataFrame(st.session_state.history)
        keep = [
            "drop",
            "vol_mult",
            "ema21_dist",
            "sma200_max_days",
            "exit_offset",
            "min_entry_price",
            "n",
            "mean",
            "median",
            "win",
            "gap",
            "skew",
            "n_tickers",
            "n_sub5",
        ]
        keep = [c for c in keep if c in hdf.columns]
        pretty = hdf[keep].copy()
        for col in ("drop", "ema21_dist", "mean", "median", "win", "gap"):
            if col in pretty.columns:
                pretty[col] = pretty[col].map(_fmt_pct)
        st.dataframe(pretty, hide_index=True, use_container_width=True)


def render_sweep(engine, prepared, meta, skipped):
    with st.sidebar:
        st.header("Sweep")
        rank_by = st.selectbox(
            "Rank winner by",
            ["Mean", "Median", "Balance", "Win", "n"],
            key="sw_rank",
        )
        min_n = st.number_input(
            "Winner needs n >",
            min_value=0,
            max_value=5000,
            value=500,
            step=50,
            key="sw_min_n",
        )
        force = st.checkbox("Allow large grids", value=False, key="sw_force")
        st.divider()
        cache_caption(meta, prepared, skipped)

    b1, b2 = st.columns(2)
    if b1.button("Preset: last opt grid", use_container_width=True):
        _apply_sweep_preset("opt")
        st.rerun()
    if b2.button("Preset: locked default combo", use_container_width=True):
        _apply_sweep_preset("locked")
        st.rerun()

    st.markdown(
        '<div class="fb-note">Constant = one value. Sweep = comma-separated list. '
        "Cartesian product. Slow part is drop × vol × EMA21 (~12s per unique triple).</div>",
        unsafe_allow_html=True,
    )

    for spec in SWEEP_PARAMS:
        st.session_state.setdefault(f"sw_mode_{spec['key']}", spec["mode_default"])
        st.session_state.setdefault(
            f"sw_const_{spec['key']}",
            int(spec["const_default"]) if spec["kind"] == "int" else float(spec["const_default"]),
        )
        st.session_state.setdefault(f"sw_list_{spec['key']}", spec["sweep_default"])

    modes, consts, texts = {}, {}, {}
    for spec in SWEEP_PARAMS:
        key = spec["key"]
        c1, c2, c3 = st.columns([1.15, 1.25, 2.2])
        with c1:
            st.markdown(f"**{spec['label']}**")
            st.caption(spec["unit"])
        with c2:
            modes[key] = st.radio(
                spec["label"] + " mode",
                ["Constant", "Sweep"],
                horizontal=True,
                key=f"sw_mode_{key}",
                label_visibility="collapsed",
            )
        with c3:
            if modes[key] == "Constant":
                if spec["kind"] == "int":
                    consts[key] = st.number_input(
                        spec["label"] + " value",
                        min_value=int(spec["min"]),
                        max_value=int(spec["max"]),
                        step=int(spec["step"]),
                        key=f"sw_const_{key}",
                        label_visibility="collapsed",
                    )
                else:
                    consts[key] = st.number_input(
                        spec["label"] + " value",
                        min_value=float(spec["min"]),
                        max_value=float(spec["max"]),
                        step=float(spec["step"]),
                        key=f"sw_const_{key}",
                        label_visibility="collapsed",
                    )
                texts[key] = st.session_state.get(f"sw_list_{key}", spec["sweep_default"])
            else:
                texts[key] = st.text_input(
                    spec["label"] + " list",
                    key=f"sw_list_{key}",
                    label_visibility="collapsed",
                    placeholder=spec["sweep_default"],
                )
                consts[key] = st.session_state.get(f"sw_const_{key}", spec["const_default"])

    combos, info, err = collect_sweep_grid(modes, consts, texts)
    if err:
        st.error(err)
        combos = []
        n_entry = 0
    else:
        n_entry = info["n_entry"]
    n_combos = len(combos)
    eta = n_entry * SEC_PER_ENTRY_PASS
    swept = [s["label"] for s in SWEEP_PARAMS if modes[s["key"]] == "Sweep"]
    locked = [s["label"] for s in SWEEP_PARAMS if modes[s["key"]] == "Constant"]
    st.markdown(
        f'<div class="fb-note">Sweep {", ".join(swept) or "—"} · '
        f'lock {", ".join(locked) or "—"} · '
        f"{n_combos} combos · {n_entry} entry passes · ~{eta:.0f}s</div>",
        unsafe_allow_html=True,
    )

    too_big = n_combos > MAX_SWEEP_COMBOS or n_entry > MAX_ENTRY_PASSES
    if too_big and not force:
        st.warning(
            f"Grid too big ({n_combos} combos, {n_entry} entry passes). "
            f"Cap is {MAX_SWEEP_COMBOS} combos / {MAX_ENTRY_PASSES} entry passes. "
            "Lock more params, shorten lists, or tick Allow large grids."
        )

    run = st.button(
        "Run sweep",
        type="primary",
        disabled=(not combos) or (too_big and not force),
    )

    if run and combos:
        bar = st.progress(0, text="Starting…")

        def progress(i, n, drop, vol, ema):
            frac = 0 if n == 0 else min(i / n, 1.0)
            bar.progress(
                frac,
                text=(
                    f"Entry pass {i}/{n}  drop {drop:.0%}  "
                    f"vol {vol:.1f}×  EMA21 {ema:.0%}"
                ),
            )

        rows = run_sweep_grid(engine, prepared, combos, progress=progress)
        bar.progress(1.0, text=f"Done · {len(rows)} rows")
        st.session_state.sweep_rows = rows

    rows = st.session_state.get("sweep_rows")
    if not rows:
        return

    full = pd.DataFrame(rows)
    rank_col = {"Mean": "mean", "Median": "median", "Balance": "balance", "Win": "win", "n": "n"}[rank_by]
    display = sweep_results_df(rows).sort_values(rank_col, ascending=False)

    winner = pick_winner(full, rank_by, int(min_n))
    if winner is not None:
        win_rate = f"{winner['win']:.1%}" if pd.notna(winner["win"]) else "—"
        feasible_n = int((full["n"] > int(min_n)).sum()) if min_n else len(full)
        st.markdown(
            f'<div class="fb-win">Winner ({escape(rank_by.lower())}, n&gt;{int(min_n)}'
            f", {feasible_n} feasible / {len(full)}): "
            f"{escape(_combo_label(winner))} · n={int(winner['n']):,} · "
            f"mean {_fmt_pct(winner['mean'])} · median {_fmt_pct(winner['median'])} · "
            f"win {win_rate}</div>",
            unsafe_allow_html=True,
        )
        if st.button("Load winner into Tuner"):
            st.session_state.drop_pct = int(round(-float(winner["drop"]) * 100))
            st.session_state.vol_mult = float(winner["vol_mult"])
            st.session_state.ema21_pct = int(round(-float(winner["ema21_dist"]) * 100))
            st.session_state.sma200_max_days = int(winner["sma200_max_days"])
            st.session_state.exit_offset = int(winner["exit_offset"])
            st.session_state.min_entry_price = float(winner["min_entry_price"])
            st.session_state.page = "Tuner"
            st.rerun()
        st.markdown(_results_html(winner.to_dict()), unsafe_allow_html=True)

    st.dataframe(
        display,
        hide_index=True,
        use_container_width=True,
        column_config={
            "drop": st.column_config.NumberColumn("Drop", format="%.0%"),
            "vol_mult": st.column_config.NumberColumn("Vol", format="%.1f"),
            "ema21_dist": st.column_config.NumberColumn("EMA21", format="%.0%"),
            "sma200_max_days": st.column_config.NumberColumn("SMA200 d"),
            "exit_offset": st.column_config.NumberColumn("t+"),
            "min_entry_price": st.column_config.NumberColumn("Px floor", format="$%.0f"),
            "n": st.column_config.NumberColumn("n", format="%d"),
            "mean": st.column_config.NumberColumn("Mean", format="+0.00%"),
            "median": st.column_config.NumberColumn("Median", format="+0.00%"),
            "win": st.column_config.NumberColumn("Win", format="0.0%"),
            "gap": st.column_config.NumberColumn("Gap", format="+0.00%"),
            "skew": st.column_config.NumberColumn("Skew", format="0.00"),
            "balance": st.column_config.NumberColumn("Balance", format="0.000"),
            "n_tickers": st.column_config.NumberColumn("Tickers", format="%d"),
            "n_sub5": st.column_config.NumberColumn("Sub-$5", format="%d"),
            "max_hold": st.column_config.NumberColumn("Max sess", format="%d"),
            "max_earn_gap": st.column_config.NumberColumn("Max print gap", format="%d"),
            "max": st.column_config.NumberColumn("Max", format="+0.0%"),
            "min": st.column_config.NumberColumn("Min", format="+0.0%"),
        },
    )
    st.download_button(
        "Download sweep CSV",
        data=display.to_csv(index=False).encode("utf-8"),
        file_name="flow_b_dashboard_sweep.csv",
        mime="text/csv",
    )


def main():
    st.set_page_config(page_title="Flow B tuner", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    try:
        engine, prepared, meta, skipped = get_prepared(_engine_stamp())
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.exception(e)
        st.stop()

    st.title("Flow B tuner")
    st.caption(
        f"Cache-only · {len(prepared):,} names · prices {meta.get('prices_fp') or 'unknown'} · no Yahoo"
    )

    page = st.radio(
        "page",
        ["Tuner", "Sweep"],
        horizontal=True,
        label_visibility="collapsed",
        key="page",
    )
    if page == "Tuner":
        render_tuner(engine, prepared, meta, skipped)
    else:
        render_sweep(engine, prepared, meta, skipped)


if __name__ == "__main__":
    main()
