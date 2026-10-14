@echo off
setlocal
rem Installs what recomp needs on a PC that has no clang: Microsoft's "Build Tools for Visual
rem Studio 2022" with its C++ workload (the C runtime and the Windows libraries that linking
rem needs) and its optional "C++ Clang Compiler for Windows". The installer is Microsoft's and is
rem downloaded from Microsoft; this script only starts it with those choices. Windows asks for
rem permission, it needs a few GB of disk and takes 10 to 20 minutes.
rem "setup-toolchain.cmd /dry" says what it would do and does nothing.
set "HERE=%~dp0"
set "BOOT=%TEMP%\vs_BuildTools.exe"
set "URL=https://aka.ms/vs/17/release/vs_BuildTools.exe"
set "ARGS=--passive --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.Llvm.Clang --includeRecommended"

"%HERE%recomp.exe" doctor | findstr /b /c:"clang: not found" >nul
if errorlevel 1 goto already

echo recomp found no clang on this PC.
echo.
echo This installs "Build Tools for Visual Studio 2022" from Microsoft: the C++ build tools, the
echo Windows SDK and the C++ Clang compiler (a few GB, 10 to 20 minutes). Going on means you accept
echo Microsoft's license terms for it: https://visualstudio.microsoft.com/license-terms/
echo.
if /i "%~1"=="/dry" goto dry
choice /m "Install it now"
if errorlevel 2 goto declined

echo.
echo downloading the installer from Microsoft...
curl.exe -L --fail -o "%BOOT%" "%URL%"
if errorlevel 1 goto nodownload
echo starting it: Windows will ask for permission, then a progress window shows the install...
"%BOOT%" %ARGS%
set "CODE=%errorlevel%"
if "%CODE%"=="0" goto installed
if "%CODE%"=="3010" goto installed
echo.
echo the installer ended with code %CODE%.
goto manual

:installed
echo.
echo done. Run doctor.cmd: it should name a clang now. If it does not, restart the PC and try again.
pause
exit /b 0

:already
echo clang is already on this PC:
"%HERE%recomp.exe" doctor
echo.
if /i not "%~1"=="/dry" pause
exit /b 0

:dry
echo would download %URL%
echo and run: %BOOT% %ARGS%
exit /b 0

:declined
echo nothing was installed.
exit /b 1

:nodownload
echo the installer could not be downloaded; is the PC online?

:manual
echo.
echo To do it by hand: download "Build Tools for Visual Studio 2022" from
echo https://visualstudio.microsoft.com/downloads/ (under "Tools for Visual Studio"), tick
echo "Desktop development with C++" and, on the right, "C++ Clang tools for Windows".
pause
exit /b 1
