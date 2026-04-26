# CrisisCompute UI — Start Script (PowerShell)
# Run from the project root: .\ui\start.ps1

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  CrisisCompute — Multi-Agent Command Center UI" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This script starts the React UI on http://localhost:3000" -ForegroundColor Yellow
Write-Host ""
Write-Host "Make sure your FastAPI backend is already running:" -ForegroundColor Gray
Write-Host "  cd 'd:\Meta Hack\multi-agent'" -ForegroundColor Gray
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor Gray
Write-Host "  uvicorn server.app:app --reload --port 7860" -ForegroundColor Gray
Write-Host ""

Set-Location $PSScriptRoot
npm run dev
