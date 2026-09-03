---
title: Flow B Dashboard
emoji: 📉
colorFrom: blue
colorTo: slate
sdk: docker
app_port: 8501
pinned: false
---

# Flow B Dashboard

Streamlit tuner for the Flow B earnings-dip backtest. Reads pickle cache only. Does not hit Yahoo.

## Local

```powershell
cd C:\Users\ian09\Grok\flow-b-dashboard
python -m pip install -r requirements.txt
streamlit run flow_b_dashboard.py
```

Public URL while this PC is on:

```powershell
.\run-tunnel.ps1
```

## What is in this folder

| File | Role |
|---|---|
| `flow_b_dashboard.py` | Streamlit UI |
| `flow_b_engine.py` | Sweep engine (copy of the Downloads script) |
| `flow_b_cache/*.pkl` | R3000, prices, volume, earnings (~55 MB). **Not** the 401 MB prepared pickle |

First boot rebuilds moving averages and writes `prepared_dashboard.pkl` (gitignored).

SMA200 gate is a radio on Tuner. Sweep can include **both** families in one grid (union, not AND):

- **Occupancy** (Tuner default): last X sessions ending at entry, require Y% of closes ≥ SMA200. Default combo is **drop −20% / vol 3.0× / EMA21 −10% / SMA200 80%/20d / t+3 / px ≥ $0**.
- **Max consecutive days below**: consecutive closes under SMA200 ending at entry (walks backward through the print). A name that lived under SMA200 for months fails even if the post-print dump is 2 days.

Sweep default includes days-below `10, 20, 25` **and** occupancy `60, 80, 90` (lookback locked at 20). Drop × EMA21 still cartesian-products inside each family.

Optional EPS-beat filter (Yahoo `Surprise(%)` on the print you enter after, not GAAP NI):

```powershell
python flow_b_engine.py --fetch-surprise
```

Date-logic check (print gaps, BMO/AMC, hold cannot span a skipped quarter):

```powershell
python flow_b_engine.py --stress-dates
```

## Streamlit Community Cloud

1. Push this folder to GitHub (see `deploy.ps1`).
2. Open https://share.streamlit.io → **New app**.
3. Repo = this repo, file = `flow_b_dashboard.py`.
4. Free Cloud is ~1 GB RAM. If the app is killed on first MA build, use Hugging Face instead.

## Hugging Face Spaces

1. `hf auth login` (token: https://huggingface.co/settings/tokens ).
2. `.\deploy.ps1 -HuggingFace`
3. Space URL: `https://huggingface.co/spaces/<you>/flow-b-dashboard`

Docker listens on port **8501**.
