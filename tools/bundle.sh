#!/usr/bin/env bash
# Make the Linux bundle: a tar.gz of what a person needs to recompile a game of their own, and an
# AppImage of the desktop front end, from a `cargo build --release --workspace` that is done.
#
# usage: tools/bundle.sh [--out <folder>] [--selftest <folder>] [--no-selftest]
#                        [--appimagetool <file> --runtime <file>]
#
# * --out is where the staging folders and the files go (default `dist-out`, ignored by git).
# * --selftest is the folder that the ignored test `build::e2e_tests::writes_the_selftest_program`
#   wrote (`RECOMP_SELFTEST_FIXTURE=<folder> cargo test -p recomp --lib -- --ignored --exact
#   build::e2e_tests::writes_the_selftest_program`); its `exefs` goes in the tarball for
#   `selftest.sh`. A bundle without one is refused unless --no-selftest says so.
# * --appimagetool and --runtime (the AppImage of appimagetool and the type 2 runtime, both of
#   the versions .github/workflows/ci.yml pins) also make `recomp-<version>-<commit>-x86_64.AppImage`.
#
# The bundle holds recomp-gui, recomp and libnx_runtime.a from target/release, the files of
# `dist/linux/`, and VERSION.txt. No game data, keys or firmware: nothing is read from anywhere but
# target/release, dist/ and --selftest.
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
out=dist-out
selftest=""
no_selftest=0
appimagetool=""
runtime=""
while [ $# -gt 0 ]; do
    case "$1" in
        --out) out="$2"; shift 2 ;;
        --selftest) selftest="$2"; shift 2 ;;
        --no-selftest) no_selftest=1; shift ;;
        --appimagetool) appimagetool="$2"; shift 2 ;;
        --runtime) runtime="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
cd "$repo"

release=target/release
for f in recomp-gui recomp libnx_runtime.a; do
    [ -f "$release/$f" ] || { echo "$f is not in $release: run cargo build --release --workspace first" >&2; exit 1; }
done
if [ "$no_selftest" = 0 ] && [ ! -f "$selftest/exefs/main" ]; then
    echo "--selftest must name the folder that holds exefs/main (see the usage), or pass --no-selftest" >&2
    exit 1
fi
if [ -n "$appimagetool" ] && { [ ! -x "$appimagetool" ] || [ ! -f "$runtime" ]; }; then
    echo "--appimagetool must be an executable file and --runtime a file" >&2
    exit 1
fi

version="$(sed -n 's/^version *= *"\(.*\)"/\1/p' Cargo.toml | head -n 1)"
sha="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
dirty=""
if [ -n "$(git status --porcelain 2>/dev/null)" ]; then dirty="+modified"; fi
date="$(date -u +%Y-%m-%d)"
name="recomp-$version-$sha-linux-x86_64"

mkdir -p "$out"
out="$(cd "$out" && pwd)"
stage="$out/$name"
rm -rf "$stage"
mkdir -p "$stage"

# Text with the version filled in.
copy_text() {
    sed -e "s/@VERSION@/$version/g" -e "s/@SHA@/$sha$dirty/g" -e "s/@DATE@/$date/g" "$1" > "$2"
}

install -m 755 "$release/recomp-gui" "$release/recomp" "$stage/"
install -m 644 "$release/libnx_runtime.a" "$stage/"
copy_text dist/linux/README.txt "$stage/README.txt"
for s in doctor.sh selftest.sh setup-toolchain.sh; do
    install -m 755 "dist/linux/$s" "$stage/$s"
done
if [ "$no_selftest" = 0 ]; then
    mkdir -p "$stage/selftest/exefs"
    cp -r "$selftest/exefs/." "$stage/selftest/exefs/"
fi
{
    echo "recomp $version"
    echo "commit $sha$dirty"
    echo "built $date with $(rustc --version 2>/dev/null || echo unknown)"
    echo "host $(uname -srm)"
} > "$stage/VERSION.txt"

tarball="$out/$name.tar.gz"
rm -f "$tarball"
tar --sort=name --owner=0 --group=0 --numeric-owner -C "$out" -czf "$tarball" "$name"
echo "bundle: $tarball ($(du -h "$tarball" | cut -f1))"

if [ -n "$appimagetool" ]; then
    appdir="$out/AppDir"
    rm -rf "$appdir"
    mkdir -p "$appdir/usr/bin" "$appdir/usr/share/doc/recomp"
    install -m 755 "$release/recomp-gui" "$release/recomp" "$appdir/usr/bin/"
    install -m 644 "$release/libnx_runtime.a" "$appdir/usr/bin/"
    install -m 755 dist/linux/AppRun "$appdir/AppRun"
    install -m 644 dist/linux/recomp.desktop "$appdir/recomp.desktop"
    install -m 644 dist/linux/recomp.png "$appdir/recomp.png"
    cp "$appdir/recomp.png" "$appdir/.DirIcon"
    copy_text dist/linux/README.txt "$appdir/usr/share/doc/recomp/README.txt"
    appimage="$out/recomp-$version-$sha-x86_64.AppImage"
    rm -f "$appimage"
    # appimagetool is an AppImage itself: run it without needing FUSE.
    ARCH=x86_64 "$appimagetool" --appimage-extract-and-run --runtime-file "$runtime" --no-appstream "$appdir" "$appimage"
    chmod +x "$appimage"
    echo "appimage: $appimage ($(du -h "$appimage" | cut -f1))"
fi

(cd "$out" && sha256sum ./*.tar.gz ./*.AppImage 2>/dev/null | sed 's# \./# #' > SHA256SUMS-linux.txt) || true
cat "$out/SHA256SUMS-linux.txt"
