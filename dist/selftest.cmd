@echo off
setlocal
rem Builds the small program in the selftest folder and runs it. It needs what a game needs
rem (clang and the runtime library) and no game. The program computes the tenth Fibonacci number
rem and exits with it, 55.
set "HERE=%~dp0"
set "OUT=%TEMP%\recomp-selftest"
if exist "%OUT%" rmdir /s /q "%OUT%"

"%HERE%recomp.exe" doctor
echo.
echo building the self-test program...
"%HERE%recomp.exe" build --exefs "%HERE%selftest\exefs" --out "%OUT%" --name selftest
if errorlevel 1 goto failed

"%OUT%\selftest.exe"
set "CODE=%errorlevel%"
if not "%CODE%"=="55" goto wrongcode

echo.
echo selftest passed: the compiler, the runtime library and the linking work on this PC.
if /i not "%~1"=="/ci" pause
exit /b 0

:wrongcode
echo.
echo the self-test program exited with %CODE%, not 55.

:failed
echo.
echo selftest FAILED. Send the text above to whoever gave you this bundle.
if /i not "%~1"=="/ci" pause
exit /b 1
