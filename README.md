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
