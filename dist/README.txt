Kiln @VERSION@ (@SHA@, @DATE@) - Windows x64 bundle
=====================================================

What this is
------------
A tool that recompiles the program of a Nintendo Switch game that you dumped from your own
console into a native Windows program, and runs it. The bundle holds no game, no keys and no
firmware: you bring the files of the game. It is a test build: it is not known to be playable
(see "What to expect"). Please do not pass on a game you have built with it: what it makes is the
game's own code.

What is in this folder
----------------------
  kiln-gui.exe   the desktop front end: pick a game, recompile it, run it
  kiln.exe       the recompiler (the front end runs it; you can use it from a terminal too)
  nx_runtime.lib   the runtime every recompiled game links with (keep it beside kiln.exe)
  doctor.cmd       shows what this PC has for a build (the compiler, the runtime library)
  setup-toolchain.cmd  installs the compiler and the Windows libraries a build needs, if this PC
                   has none (it starts Microsoft's own installer; see below)
  selftest.cmd     builds and runs a tiny program that is part of the bundle, with no game:
                   it proves that the compiler and the linking work on this PC
  VERSION.txt      what version this is and where it came from

What your PC needs
------------------
1. Windows 10 or 11 (64-bit) and a graphics card and driver with Vulkan 1.3. It has only been
   run on one NVIDIA GeForce RTX 5070 Ti; AMD and Intel cards are untested. Plan on 8 GB of RAM
   or more (the running game uses about 4 GB).
2. clang and lld, and the Windows SDK libraries, to compile and link what kiln generates.
   They are not in this folder. Double-click setup-toolchain.cmd: if kiln finds no clang it
   offers to install "Build Tools for Visual Studio 2022" (free, from Microsoft; a few GB, 10 to 20
   minutes, Windows asks for permission) with the C++ workload and the C++ Clang compiler.
   By hand it is the same: in the installer of Build Tools for Visual Studio tick "Desktop
   development with C++" and, among the optional parts on the right, "C++ Clang tools for
   Windows". Kiln finds the clang inside it by itself. With another clang, put it on the PATH
   or give its path in the Settings screen of kiln-gui. (The script has not been run on a PC
   that had nothing: if it fails, do it by hand.)
   If kiln-gui.exe or a built game will not start because a vcruntime140 file is missing,
   install "Microsoft Visual C++ Redistributable 2015-2022 (x64)" (Build Tools installs it too).
3. The extracted ExeFS and RomFS of the game, from your own copy: a folder or an image for each.
   No keys are needed and none are read.

First run
---------
1. Double-click doctor.cmd. It should name a clang and the runtime library (nx_runtime.lib);
   if it says clang: not found, run setup-toolchain.cmd first.
2. Double-click selftest.cmd. It should end with "selftest passed".
3. Run kiln-gui.exe. Choose the folder that holds the game's exefs and romfs folders (or drop
   it on the window) and press Recompile; the output of the build streams into the window. The
   finished game appears in the Library: press Play. A window opens at once and shows the game
   when it has something to draw (a few seconds); the game has to start up first.
   Recompiling took about 40 seconds on the developer's PC. The game's settings (sound, keys,
   frame limit, logging) are in the Library next to it.

Keys
----
With the game's window focused, the keyboard is the console's handheld controls (the keys are
the physical ones, so the layout of your keyboard does not matter):

  Left stick         W A S D (click: Left Shift)
  Right stick        Arrow keys (click: Right Shift)
  A, B, X, Y         L, K, I, J   (X above, Y left, B below, A right)
  Directional pad    T up, F left, G down, H right
  L, R               Q, E
  ZL, ZR             Z, C
  Plus, Minus        Enter, Backspace

An Xbox-style (XInput) gamepad works without setup, with its face buttons by position (the
bottom one is B, as on the console).

What to expect
--------------
This is what has been seen on the developer's PC, one game, one graphics card:
- It drew the game's menu, its opening cutscene and its first area, and ran at 40 to 60 frames a
  second in that area (about 27 in the cutscene). What is past the first area has not been tried.
- The sound is decoded, mixed and sent to the default sound device; nobody has been able to check
  by ear that it is right. There is no reverb or other effect.
- A thin white bar at the top left of the first area's picture has no explanation yet.
- The game sometimes stops early; rare, and a stop prints a report of what every thread was doing.
- Nothing here is a claim that a game is playable.

If something goes wrong
-----------------------
Send: the text of the console in the kiln-gui window (a run that stops prints a report), the
output of doctor.cmd and selftest.cmd, and your graphics card and driver version. Do not send
game files.
