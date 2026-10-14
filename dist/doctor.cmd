@echo off
rem Shows what this PC has for a build: the version of kiln, the C compiler it will run and
rem the runtime library that programs link. Nothing is built.
"%~dp0kiln.exe" doctor
echo.
if /i not "%~1"=="/ci" pause
