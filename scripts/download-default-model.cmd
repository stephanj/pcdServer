@echo off
rem Downloads the default model into models\ next to this folder and verifies
rem its SHA-256 checksum. Uses curl.exe and certutil, built into Windows 10+.
setlocal
set "root=%~dp0.."
set "dst=%root%\models\Qwen3.5-0.8B-Q8_0.gguf"
set "url=https://huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q8_0.gguf"
set "expected=37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f"

if not exist "%root%\models" mkdir "%root%\models"
if exist "%dst%" (
    call :verify && goto :done
    echo existing download is incomplete or corrupt, resuming
)
curl.exe -L --fail --continue-at - -o "%dst%" "%url%" || goto :mismatch
echo checksum mismatch: %actual% 1>&2

:fail
call :verify || goto :mismatch

:done
echo %dst%
endlocal
exit /b 0

:mismatch
echo checksum mismatch: %actual% 1>&2

:fail
endlocal
exit /b 1

:verify
set "actual="
for /f "skip=1 delims=" %%h in ('certutil -hashfile "%dst%" SHA256') do if not defined actual set "actual=%%h"
if not defined actual exit /b 1
set "actual=%actual: =%"
if /i "%actual%"=="%expected%" exit /b 0
exit /b 1
