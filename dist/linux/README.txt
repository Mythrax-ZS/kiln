recomp @VERSION@ (@SHA@, @DATE@) - Linux x86-64 bundle
=====================================================

What this is
------------
A tool that recompiles the program of a Nintendo Switch game that you dumped from your own
console into a native Linux program, and runs it. The bundle holds no game, no keys and no
firmware: you bring the files of the game. It is a test build, and on Linux it is less tried
than on Windows: the game it was developed on has only been run on Windows, so on Linux what
is tested is the recompiler and the runtime (their unit tests, and a small built-in program),
not a game (see "What to expect"). Please do not pass on a game you have built with it: what
it makes is the game's own code.

Two ways to get it
------------------
  recomp-<version>-<commit>-x86_64.AppImage   one file: chmod +x it and run it (the desktop
                                              front end); `./recomp-*.AppImage recomp doctor`
                                              runs the command line tool inside it. An AppImage
                                              needs FUSE 2 (libfuse2); without it run
                                              `./recomp-*.AppImage --appimage-extract-and-run`.
  recomp-<version>-<commit>-linux-x86_64.tar.gz   this folder, with scripts

What is in the folder
---------------------
  recomp-gui        the desktop front end: pick a game, recompile it, run it
  recomp            the recompiler (the front end runs it; you can use it from a terminal too)
  libnx_runtime.a   the runtime every recompiled game links with (keep it beside recomp)
  doctor.sh         shows what this PC has for a build (the compiler, the runtime library)
  selftest.sh       builds and runs a tiny program that is part of the bundle, with no game:
                    it proves that the compiler and the linking work on this PC
  setup-toolchain.sh  installs clang and lld with your package manager (it asks first)
  VERSION.txt       what version this is and where it came from

What your PC needs
------------------
1. A 64-bit Linux with a graphics card and driver with Vulkan 1.3 (Mesa or NVIDIA), and for the
   front end X11 or Wayland with OpenGL, libxkbcommon and libvulkan: on Ubuntu or Debian
   `sudo apt install libvulkan1 libxkbcommon-x11-0 libgl1 mesa-vulkan-drivers`. Plan on 8 GB of
   RAM or more.
2. clang (a recent one: 21 and 22 are the ones tried) and lld, to compile and link what recomp generates. They are not in the
   bundle. Run setup-toolchain.sh (`sudo apt install clang lld` and its equivalents).
3. The extracted ExeFS and RomFS of the game, from your own copy: a folder or an image for each.
   No keys are needed and none are read.

First run
---------
1. Run ./doctor.sh. It should name a clang and the runtime library (libnx_runtime.a).
2. Run ./selftest.sh. It should end with "selftest passed".
3. Run recomp-gui (or the AppImage). Choose the folder that holds the game's exefs and romfs
   folders (or drop it on the window) and press Recompile; the output of the build streams into
   the window. The finished game appears in the Library: press Play. Games are kept in a "games"
   folder beside recomp-gui, or in ~/.local/share/recomp-gui/games when that folder cannot be
   written to (an AppImage), which the Settings screen shows and changes.

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

What to expect
--------------
- Developed and measured on Windows with one game and one graphics card (NVIDIA RTX 5070 Ti):
  the game's menu, opening cutscene and first area drew there, at 40 to 60 frames a second in the
  first area. Nothing of that has been run on Linux.
- No sound on Linux: the game's audio is mixed, but only the Windows build sends it to a sound
  device (the Linux build can write it to a WAV file with NX_AUDIO_DUMP=<file>).
- A gamepad is not read on Linux (only the keyboard).
- A thin white bar at the top left of the first area's picture has no explanation yet. The game
  sometimes stops early; rare, and a stop prints a report of what every thread was doing.
- Nothing here is a claim that a game is playable.

If something goes wrong
-----------------------
Send: the text of the console in the recomp-gui window (a run that stops prints a report), the
output of doctor.sh and selftest.sh, your distribution, and your graphics card and driver version.
Do not send game files.
