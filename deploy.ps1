param(
    [switch]$GitHub,
    [switch]$HuggingFace
)

Set-Location $PSScriptRoot

function Ensure-Git {
    if (-not (Test-Path .git)) {
        git init
        git lfs install
        git add .gitattributes
        git add -A
        git commit -m "Initial Flow B dashboard (cache-only, no Yahoo)"
    }
}

if (-not $GitHub -and -not $HuggingFace) {
    Write-Host @"
Usage:
  .\deploy.ps1 -GitHub        # commit + print Streamlit Cloud steps
  .\deploy.ps1 -HuggingFace   # create/upload HF Space (needs: hf auth login)

Logins (one-time):
  git:  install GitHub CLI then  gh auth login
  hf:   python -m huggingface_hub.commands.huggingface_cli login
        token: https://huggingface.co/settings/tokens  (write access)
"@
    exit 0
}

Ensure-Git

if ($GitHub) {
    $gh = Get-Command gh -ErrorAction SilentlyContinue
    if (-not $gh) {
        Write-Host "GitHub CLI not on PATH. Install: winget install GitHub.cli"
        Write-Host "Then: gh auth login"
        Write-Host "Then: gh repo create flow-b-dashboard --public --source=. --remote=origin --push"
        Write-Host ""
        Write-Host "After the repo exists, open https://share.streamlit.io"
        Write-Host "  New app -> this repo -> Main file path: flow_b_dashboard.py"
        exit 1
    }
    gh auth status
    if ($LASTEXITCODE -ne 0) { gh auth login }
    $exists = gh repo view flow-b-dashboard 2>$null
    if (-not $exists) {
        gh repo create flow-b-dashboard --public --source=. --remote=origin --push
    } else {
        git push -u origin HEAD
    }
    $url = gh repo view --json url -q .url
    Write-Host "GitHub: $url"
    Write-Host "Streamlit Cloud: https://share.streamlit.io  -> New app -> $url -> flow_b_dashboard.py"
}

if ($HuggingFace) {
    python -c "from huggingface_hub import whoami; print(whoami()['name'])"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Not logged in. Run: hf auth login"
        Write-Host "Create a write token: https://huggingface.co/settings/tokens"
        exit 1
    }
    python .\push_hf.py
}
