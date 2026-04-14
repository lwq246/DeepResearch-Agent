@echo off
powershell -ExecutionPolicy Bypass -File "%~dp0run-dev.ps1" %*
powershell -NoProfile -Command "$body = @{ message = 'Compare DPO and RLHF based on research findings.' } | ConvertTo-Json; Invoke-RestMethod -Method Post -Uri 'http://localhost:8000/chat/debug' -ContentType 'application/json' -Body $body | ConvertTo-Json -Depth 8"