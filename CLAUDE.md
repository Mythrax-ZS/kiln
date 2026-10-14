# Kiln: working rules

Static recompiler that turns Horizon OS (Nintendo Switch) AArch64 programs into native x86_64
executables and shared libraries. LLVM IR is emitted as text and compiled with clang.

Crates:
- `kiln` (root): library + CLI. Input layer, loader, analysis, lifter, codegen.
- `nx_runtime` (`nx_runtime/`): everything shared between compile time and run time: storage, RomFS,
  NPDM, the A64 decoder and interpreter, guest memory and the Horizon emulation layer. Built as
  `rlib` and `staticlib`; every produced binary links against it.
- `nx_gpu` (`nx_gpu/`): the GPU front, with no knowledge of Horizon: the driver service's device nodes,
  the display buffer queue and a Maxwell command processor with a macro engine. `nx_runtime`
  depends on it. It draws nothing itself: draws, clears and displayed buffers go to a `Backend`.
- `nx_shader` (`nx_shader/`): Maxwell shader programs: header, decoder, control flow, a SPIR-V builder
  and the translation to SPIR-V. No dependencies and no knowledge of the GPU model that feeds it.
- `nx_vulkan` (`nx_vulkan/`): the Vulkan device, the compute harness the translator is tested
  with, and the renderer (a `Backend`) with the ASTC decoder (`ash`, loaded at run time).
- `kiln_gui` (`kiln_gui/`): the desktop front end (`egui`); it runs `kiln` and the games as
  processes. Run `cargo run -p kiln_gui`; `KILN_GUI_SCREENSHOT=<png>` saves its window. Its look (the
  icon, the mascot Cinder, the lettering) is pixel art in `kiln_gui/src/brand.rs`; `cargo run -p
  kiln_gui --example brand` writes `assets/brand/` and `dist/linux/kiln.png` from it, and
  `kiln_gui/build.rs` puts `assets/brand/kiln.ico` into `kiln-gui.exe`.

## Inputs (owner instruction, 2026-10-12)

The tool takes the **already extracted ExeFS and RomFS** of a game (a directory or an image for
each), or a homebrew NRO. It does **not use `prod.keys` at all** and has no NSP, NCA, ticket or
crypto code. Do not add any. The original brief's NCA/IVFC/key milestone M1 was replaced by an
input layer (ExeFS directory or PFS0, RomFS image parser, RomFS builder for extracted directories).

## Gate before every commit

```
cargo fmt --all --check
cargo lint            # alias for: cargo clippy --all-targets -- -D warnings
cargo test
```

`.github/workflows/ci.yml` runs this gate on Windows and (without `kiln_gui`) on Linux for every
push and pull request, and builds the Windows bundle (`tools/bundle.ps1`, `dist/`) and the Linux
bundle and AppImage (`tools/bundle.sh`, `dist/linux/`); see the README.
The workflow pins the Rust version (`RUST_VERSION`) and Unicorn: move them on purpose.

## Commits

Small chunks. Title at most 50 characters. No body. No attribution lines of any kind.

## Hard rules

- No firmware, keys or title data in the repository or in any committed output. Fixtures are
  synthetic or generated inside the test itself.
- Do not describe the output as "1:1 accurate". Describe measured test coverage instead.
- Every undocumented or assumed behavior goes in `docs/ASSUMPTIONS.md` with its source.
- An unknown SVC or service command stops the run and reports its address and number. Never
  return a made-up success.
- Where Unicorn disagrees with the Arm pseudocode, the pseudocode wins and the disagreement is
  recorded in `docs/UNICORN_DISAGREEMENTS.md`.
- Milestones M1..M10 (see `docs/reports/`): stop after each one, run the tests, write the report,
  then move on. M8 passed (2026-10-12): the owner approved the GPU front, the `nx_gpu` crate
  (`nvdrv` and its device nodes, the display and buffer-queue path, a tracing command processor
  with a macro engine). The owner approved shader translation (Maxwell SASS to SPIR-V, crate
  `nx_shader`, M9) and the Vulkan backend (`ash`, crate `nx_vulkan`, M10) on 2026-10-13; M9
  passed the same day, and M10 draws the title's menu, its opening cutscene and its first area
  (frames dumped and looked at), shows them in a window and takes keyboard and gamepad input; in
  the first area it runs at 40 to 60 frames a second on one RTX 5070 Ti, and what is past that
  area has not been tried (open, see `docs/reports/M10-vulkan-backend.md`). Never describe the
  output as playable before frames are measured; saying what a frame shows needs the frame
  itself (a dumped image looked at), not a count.
- Out of scope: online services, account checks, licensing checks, AArch32 (detect and error).

## External tools used by tests

Tests that need an external tool print a loud `SKIPPED` line and pass when the tool is missing.
Set `KILN_REQUIRE_EXTERNAL=1` to turn those skips into failures.

- clang with the x86-64 target: `KILN_CLANG`, then `PATH`, then the LLVM bundled with Visual Studio.
  Produced programs link `nx_runtime.lib`/`libnx_runtime.a`: `KILN_RUNTIME_LIB`, else it is built
  with `cargo build -p nx_runtime` (the native tests do this themselves).
- Native differential test sizes: `KILN_NATIVE_WORDS` (instructions per encoding group, default
  40), `KILN_NATIVE_STATES` (random states per instruction, 4), `KILN_NATIVE_PROGRAMS` (random
  control-flow programs, 150), `KILN_FLOAT_STATES` (random states for each word of the tests of
  the floating-point and vector forms the code generator lowers itself, 600). The IR-level tests
  scale with `KILN_LIFT_CASES` and `KILN_LIFT_PROGRAMS`.
- Real title (opt-in, `KILN_TITLE_DIR` = a folder holding `exefs/`): `tests/real_title.rs` checks
  the loader; `tests/real_code.rs` recompiles `main` and compares native with interpreted runs of
  sampled functions (`KILN_REAL_FUNCTIONS`, `KILN_REAL_STATES`, `KILN_REAL_ALL=1` for every
  function). Only aggregate numbers are printed; nothing from the title is stored.
- Unicorn differential test sizes: `KILN_DIFF_CASES` replaces the default number of cases of every
  group in `unicorn_diff`, `unicorn_fp` and `unicorn_simd` (the random-words test draws that many
  words).
- Kernel layer: `nx_runtime/src/horizon/tests.rs` drives the SVC interface with real host threads
  and needs no external tool. The homebrew tests in `src/build/e2e_tests.rs` build small NROs
  from assembled words and run them (clang needed). `NX_TRACE_SVC=1` logs every SVC and IPC
  request of a produced program; `NX_TRACE_LOG=1` prints what it logs through `lm`;
  `NX_TRACE_WAITS=<microseconds>` logs every blocking supervisor call (sleep, wait, lock, condition
  variable, address, request of a service) that took that long, with the thread and the time in
  milliseconds since the start, to see which threads wait for which;
  `NX_RANDOM_SEED=<n>` makes the kernel's entropy reproducible; `NX_ROMFS=<dir|image>` names the
  RomFS it reads at run time; `NX_WATCHDOG=<seconds>` says, when no frame has been presented for
  that long (and again every that many seconds), what every guest thread is doing (the supervisor
  call, the service and command of a request, the return addresses as of its last call) and
  where the command processor and the renderer are (`kiln symbols` names the addresses;
  `NX_WATCHDOG_ALWAYS=1` reports every that many seconds whether frames come or not, for a title
  that presents the frames of a loading screen for ever). `NX_JITTER=<percent>[,<microseconds>]`
  makes that share of the supervisor calls sleep for up to that long first (500 by default), to stir
  the threads' timing when looking for a race. A run
  that stops reports every thread (priority, core, what it is doing), then the registers of the
  thread that stopped and 256 bytes at each register that holds an address (and at what those
  point at in the heap), for reading a crash. `NX_CORES=per-core` (or `one`) makes a guest thread
  hold a core to run guest code and give it up at a supervisor call that can block or reschedule,
  threads waiting for a core being served in order (a model of the console's cores, off by
  default; priorities are not modelled).
- Audio: `NX_AUDIO=0` leaves the sound device out; `NX_AUDIO_DUMP=<file>` writes every sample the
  audio renderer mixes to a 16-bit stereo WAV file (kept readable while the program runs, so a killed
  program leaves a good file); `NX_TRACE_AUDIO=1` says every second how many voices played, in which
  formats, and how loud the mix was. The device test of `audio_out` feeds the machine's sound device
  silence, and prints `SKIPPED` without a device.
- GPU front: `nx_gpu`'s unit tests need no external tool. `NX_TRACE_GPU=1` prints every submission
  of a produced program and, when it ends, what the command processor counted (`NX_GPU_TOP=<n>` is
  how many of the busiest methods it lists, 0 for all); `NX_TRACE_GPU=2` prints every method write
  and macro call (`NX_TRACE_GPU_WINDOW=<first frame>:<last frame>[:<level>]` does so only while those
  frames are shown); `NX_TRACE_REGISTER=<hex byte offset>[,...]` prints every write to those 3D registers
  (with the draw count and the macro it came from) and `NX_TRACE_MACRO=<id>` every call of that macro
  with the writes it makes and its code, once; `NX_MAX_FRAMES=<n>` ends the run after `n` presented
  frames; `NX_SAVE_DIR=<dir>` is where save data goes (default `./nx_save`). `kiln symbols --exefs <dir> [--match <text>]
  [<address>...]` names the function around an address, for reading the return addresses a stop
  report ends with.
- Renderer: `nx_vulkan`'s tests need a Vulkan device (`NX_VULKAN_DEVICE` picks one; they print
  `SKIPPED` without one) and `spirv-val` for the translator's. `NX_RENDERER=vulkan` makes a
  produced program draw; `NX_TRACE_PERF=1` has the renderer say every 100 frames how long they took
  and where the time went (what `NX_TRACE_GPU` does too, without the line for every submission,
  which slows a run by about a third; the steps of a draw are timed only when one of them is set; a build with `--features alloc_stats` on `nx_runtime` counts
  the allocations of the draws' steps and the guest's calls too, and says how often memory was
  hashed); `NX_DUMP_FRAMES=<dir>` writes each displayed frame as a PNG (opaque, cropped
  to what is shown; `NX_DUMP_EVERY=<n>`, `NX_DUMP_COUNT=<n>`, `NX_DUMP_TARGETS=1` for every render
  target too: a PNG is uncompressed, so dump every n-th frame); `NX_TRACE_DRAWS=<frame>` lists the
  draws and clears of one frame (with `NX_TRACE_STATE=1` their state and vertices, `NX_DUMP_DRAWS=<dir>`
  and `NX_DUMP_FROM=<n>` a picture after each); `NX_TRACE_COMPUTE=1` says how long the device took over
  each compute launch (timestamps); `NX_TRACE_TEXTURES=1`, `NX_TRACE_CBUFS=1`,
  `NX_DEBUG_NO_CULL=1`, `NX_DEBUG_FULL_VIEWPORT=1`, `NX_DEBUG_SOLID_PIXEL=1` (every pixel program writes
  magenta; `NX_DEBUG_SOLID_KEEP=<hex address>[,...]` spares the programs at those addresses,
  `NX_DEBUG_SOLID_ONLY=<hex address>[,...]` does it to those programs alone,
  `NX_DEBUG_SKIP_PROGRAMS=<hex address>[,...]` draws nothing with those programs,
  `NX_DEBUG_DISCARD_PROGRAMS=<hex address>[,...]` runs the vertex stage of the draws of those vertex
  programs and rasterizes nothing, `NX_DEBUG_TRIVIAL_VERTEX=<hex address>[,...]` gives those vertex
  programs a body that writes a constant position, `NX_DEBUG_DISPATCH_LOOP=1` translates every
  program with the loop around a `switch` (A-114), `NX_SPIRV_OPT=<arguments>` passes every graphics
  program through `spirv-opt` first),
  `NX_DEBUG_EQUAL_AS_LEQUAL=1`, `NX_DEBUG_NO_DEPTH_TEST=1`, `NX_DEBUG_FLIP_FRONT=1`,
  `NX_TRACE_TARGETS=1` (each render target configuration, the first time it is drawn to) and
  `NX_GPU_SYNC=1` runs the commands of a submission in the call that makes it (with a renderer they
  otherwise run on a thread of their own, A-124); `NX_GPU_LENIENT=1` skips a draw the renderer
  cannot make (counted in the report of every 100 frames, and said once) instead of stopping the
  run: development runs set it, and so does `kiln-gui` for a game it starts to be played (its
  "Stop at a draw that cannot be rendered" setting turns that off);
  `NX_SPIRV_CHECK=1` look at one thing each; `NX_DEBUG_WAIT_FINISH=1` waits for the device at the end
  of every submission of the guest (the renderer otherwise lets the device run behind the
  recording, A-123); `NX_DEBUG_SYNC_DRAWS=<n>` waits for the device after
  every draw (slow), from the first frame with `n` draws on (`NX_DEBUG_SYNC_FROM=<frame>`: from
  that presented frame on), so that the draw that loses the device is named with its state,
  programs and textures, and how long after its submission the device was lost (a hang takes the
  2 s of the operating system's watchdog, a fault does not); `NX_DEBUG_CHECKPOINTS=1` (an NVIDIA device) marks the
  commands of a submission so that a lost device says which it was running;
  `NX_VULKAN_VALIDATE=1` turns the validation layer on; `NX_SHADER_DUMP=<dir>` writes the programs
  draws use and the compute programs launched, as `COMPUTE_<address>.bin` without a header (read
  them with `cargo run --release -p nx_shader --example sass -- <file>`; the `audit` example lists
  what the translator lacks). To find why a frame is wrong: `NX_DUMP_BUSY=<draws>` dumps the first
  frame with that many draws and every target, `NX_TRACE_AFTER_BUSY=1` traces the frame after it
  (what `NX_TRACE_DRAWS` does), `NX_DUMP_PASSES=<dir>` writes a picture of each pass of a traced
  frame, `NX_DUMP_DRAWS=<dir>` with `NX_DUMP_FROM=<n>` and `NX_DUMP_UNTIL=<n>` one after each
  command, `NX_TRACE_COMPUTE=1` prints every compute launch with its descriptor and
  `NX_TRACE_CBUFS=1` the constant buffers a traced draw or launch reads. `NX_INPUT_SCRIPT="<frame>
  <Action> [<frames>];..."` (or a file) presses actions of a keymap at presented frame numbers, for
  runs without a person (the frame rate decides when a menu is ready: a run that is 15 times faster
  needs the press 15 times later). Never `NX_DUMP_FRAMES` without `NX_DUMP_EVERY` for a long run.
- Window and input: `NX_WINDOW=1` (with `NX_RENDERER=vulkan`) shows the frames in a window
  (`NX_WINDOW_SIZE=<w>x<h>`, `NX_WINDOW_TITLE`, `NX_PRESENT_MODE=fifo|mailbox|immediate`); no test
  opens one. The window opens when the program starts (not when it first draws, seconds later),
  comes to the front and is painted black until the first frame is presented. The keyboard and an XInput gamepad are the handheld controls of `hid`: `NX_KEYMAP=<file>`
  rebinds keys, `NX_TRACE_KEYS=1` prints the keys and button changes, `NX_GAMEPAD=0` and
  `NX_GAMEPAD_LABELS=1` (docs/ASSUMPTIONS.md A-104). To drive a window from a script, post key messages
  to the game's own window and capture only that window (`PrintWindow`), never the screen, and bind
  keys nobody types (an `NX_KEYMAP` using F13 and up) so that the person at the keyboard cannot
  disturb the run; delete the trace logs afterwards, they hold the keys pressed.
- ASTC decoder: `nx_vulkan/tests/astc_reference.rs` compares it with ARM's `astcenc`
  (`KILN_ASTCENC=<path>`, or `astcenc` on the `PATH`; the tool may be unpacked from a distribution
  package without installing it); `KILN_ASTC_BATCHES` (default 4, 4,096 blocks each) scales it and
  `KILN_ASTC_ONLY=<n>` runs one footprint.
- A real title through the whole pipeline (opt-in, by hand; nothing from it is stored): build with
  `kiln build --exefs <exefs> --romfs <romfs> --out <dir>`, run `<dir>/<name>` with
  `NX_TRACE_SVC=1`, and use `kiln relink` / `--reuse-objects` to iterate on the runtime.
  `KILN_LINK_MAP=<file>` makes the link write a map of the executable, and on Windows
  `pwsh -File tools/sample_threads.ps1 -Map <file>` samples where every thread of the running
  program is (a generated function `f_<guest address>`, a Rust function of the runtime, a module
  of the process) to say what it is busy with (`-Callers` adds, to a sample in a system library,
  the place of the program that its stack points into), and `pwsh -File tools/guest_memory.ps1 -Map <file>
  -Ranges "<address>:<length>,..." [-Words] [-Find <value>]` reads the guest memory of the running
  program from outside (it walks the page table the map names, `nx_mem_l1`; a frame's saved
  registers on the stack and the objects they point to answer what a stuck loop is looping on;
  `-Find` lists where a 64-bit value is, scanning 3 GB in about 6 s: an object is found by the
  address of its table of virtual functions, its fields then say what state a game is in).
  `NX_HOLD_ON_STOP=1` keeps the process alive after a stop, so that its memory can still be read.
  `NX_TRAP_WRITES=<address>:<length>[,...]` (guest addresses, hexadecimal) says who writes into
  those pages: every guest write to them is reported once per call chain (and again at the 10th,
  100th... time) with the thread, the address, the value and the return addresses as offsets in
  the image, which the link map (`KILN_LINK_MAP`) names (`f_<guest address>` is a generated
  function). A page nobody writes says nothing; a page the guest writes through another address
  (an alias) is not seen.
  `pwsh -File tools/game_window.ps1 -ProcessId <pid> [-Capture <png>] [-Keys "F13:200,..."]` takes a
  picture of the program's own window (never of the screen) and posts keys to its message queue;
  run the program with an `NX_KEYMAP` that binds F13 and up so that nobody's typing disturbs it.
  `NX_CENSUS=<file>` counts the floating-point and SIMD instruction words that reach the runtime
  (`nx_vec`, `nx_vecmem`), the words the lifter left to the interpreter and the addresses the slow
  path interprets, and rewrites the file every ten seconds, most frequent first, with the decoded
  instruction: what to lower to native code next.
- Unicorn (AArch64 oracle): `UNICORN_DLL`, else `.venv` (`py -m venv .venv`, then
  `.venv\Scripts\pip install unicorn`). The shared library is loaded with `libloading`.
