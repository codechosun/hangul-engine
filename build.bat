@echo off
rem ===========================================================================
rem  hangul-engine 빌드 스크립트
rem
rem    build.bat            엔진을 내보내고 데모(hello)까지 만든다
rem    build.bat d07        ch-D07-training 을 빌드해서 build\d07.exe 로
rem    build.bat d06 double USE_DOUBLE 로. D6 의 gradcheck 은 이게 필요하다
rem    build.bat all        서른 장을 전부 빌드한다
rem
rem  MSVC 가 필요하다. "x64 Native Tools Command Prompt" 를 쓰거나,
rem  이 스크립트가 알아서 vcvars64.bat 을 찾는다.
rem ===========================================================================
setlocal EnableDelayedExpansion

rem 콘솔을 UTF-8 로. 이 파일도, 예제 출력도 UTF-8 이다.
chcp 65001 >nul

cd /d "%~dp0"

rem ---- MSVC 준비 ----
where cl >nul 2>&1
if %ERRORLEVEL%==0 goto :ready

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo [!] Visual Studio 를 못 찾았다. C++ 워크로드를 설치할 것.
    exit /b 1
)

for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * ^
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 ^
    -property installationPath`) do set "VSPATH=%%i"

if not defined VSPATH (
    echo [!] MSVC 도구를 못 찾았다.
    exit /b 1
)

call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (
    echo [!] vcvars64.bat 실행 실패.
    exit /b 1
)

:ready
if not exist build mkdir build

rem 교재 전체가 쓰는 공통 옵션.
rem   /utf-8   한글 주석과 문자열 (A0)
rem   /GL      전체 프로그램 최적화. /LTCG 와 짝 (A4)
rem   /openmp  병렬 판 (C5, D8)
set "FLAGS=/nologo /EHsc /utf-8 /W3 /O2 /GL /openmp /D_CRT_SECURE_NO_WARNINGS /I common /I lib"
set "LINKFLAGS=/link /LTCG /SUBSYSTEM:CONSOLE"

set "LIBSRC="
for %%f in (lib\*.c lib\*.cpp) do set "LIBSRC=!LIBSRC! %%f"
for %%f in (common\*.c) do set "LIBSRC=!LIBSRC! %%f"

if "%~1"=="" goto :demo
if /i "%~1"=="all" goto :all
goto :one

rem ---------------------------------------------------------------- 한 장만
:one
set "TAG=%~1"
set "DIR="
for /d %%d in (ch-*) do (
    echo %%d | findstr /i /c:"%TAG%" >nul && set "DIR=%%d"
)

if not defined DIR (
    echo [!] "%TAG%" 에 맞는 장이 없다. 예: build.bat d07
    exit /b 1
)

echo.
echo === !DIR! ===
set "MAIN=!DIR!\Main.cpp"
if not exist "!MAIN!" set "MAIN=!DIR!\Main.c"
if not exist build\%TAG% mkdir build\%TAG%

set "EXTRA="
if /i "%~2"=="double" set "EXTRA=/DUSE_DOUBLE"
if defined EXTRA echo   (USE_DOUBLE)

cl %FLAGS% %EXTRA% "!MAIN!" !LIBSRC! psapi.lib /Fe:build\%TAG%.exe /Fo:build\%TAG%\ %LINKFLAGS%
if errorlevel 1 exit /b 1

echo.
echo 빌드 완료 : build\%TAG%.exe
echo 실행      : build\%TAG%.exe        (저장소 루트에서)
exit /b 0

rem ---------------------------------------------------------------- 전부
:all
for /d %%d in (ch-*) do (
    set "DIR=%%d"
    set "NAME=%%d"
    set "MAIN=%%d\Main.cpp"
    if not exist "!MAIN!" set "MAIN=%%d\Main.c"
    if not exist build\!NAME! mkdir build\!NAME!

    echo.
    echo === %%d ===
    cl %FLAGS% "!MAIN!" !LIBSRC! psapi.lib /Fe:build\!NAME!.exe /Fo:build\!NAME!\ %LINKFLAGS% >build\!NAME!.log 2>&1
    if errorlevel 1 (
        echo   [실패] 자세한 것은 build\!NAME!.log
    ) else (
        echo   [성공] build\!NAME!.exe
    )
)
exit /b 0

rem ---------------------------------------------------------------- 데모
:demo
echo.
echo === 1/3  E2 예제 — 모델을 훈련해 data\chat.hgen 을 굽는다 ===
if not exist build\e02 mkdir build\e02
cl %FLAGS% ch-E02-engine\Main.cpp !LIBSRC! psapi.lib /Fe:build\e02.exe /Fo:build\e02\ %LINKFLAGS%
if errorlevel 1 exit /b 1

build\e02.exe
if errorlevel 1 exit /b 1

echo.
echo === 2/3  엔진만 dist\hangul-engine\ 으로 내보낸다 ===
where python >nul 2>&1
if %ERRORLEVEL%==0 (
    python tools\export_engine.py
) else (
    echo   [!] python 이 없다. dist\ 가 이미 있으면 그대로 쓴다.
)
if not exist dist\hangul-engine\Engine.hpp (
    echo   [!] dist\hangul-engine\ 이 없다. python tools\export_engine.py 를 돌릴 것.
    exit /b 1
)

echo.
echo === 3/3  엔진만 보고 데모 앱을 빌드한다 ===
if not exist buildpp mkdir buildpp
cl /nologo /EHsc /utf-8 /W3 /O2 /D_CRT_SECURE_NO_WARNINGS /I dist\hangul-engine ^
   ch-E02-engine\app\main.cpp dist\hangul-engine\*.cpp dist\hangul-engine\*.c ^
   /Fe:build\hello.exe /Fo:build\app\ /link /SUBSYSTEM:CONSOLE
if errorlevel 1 exit /b 1

echo.
echo ============================================================
build\hello.exe
echo ============================================================
echo.
echo 데모 완료. build\hello.exe 를 다시 돌려도 된다.
exit /b 0
