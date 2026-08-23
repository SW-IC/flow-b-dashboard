# Local Streamlit + public Cloudflare URL (PC must stay on).
Set-Location $PSScriptRoot
python -m pip install -q -r requirements.txt
$job = Start-Job -ScriptBlock {
    Set-Location $using:PSScriptRoot
    python -m streamlit run flow_b_dashboard.py --server.headless true --server.port 8501
}
Write-Host "Streamlit starting on http://localhost:8501 ..."
Start-Sleep -Seconds 4
npx --yes cloudflared tunnel --url http://localhost:8501
Stop-Job $job -ErrorAction SilentlyContinue
Remove-Job $job -ErrorAction SilentlyContinue
