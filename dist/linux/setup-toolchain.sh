#!/bin/sh
# Installs what recomp needs on a PC that has no clang or lld, with the package manager of the
# distribution (it asks first, and uses sudo). `setup-toolchain.sh --dry` says what it would run.
here="$(cd "$(dirname "$0")" && pwd)"
recomp="$here/recomp"
[ -x "$recomp" ] || recomp="$(command -v recomp 2>/dev/null)"

have_clang=1
if [ -n "$recomp" ] && "$recomp" doctor 2>/dev/null | grep -q '^clang: not found'; then
    have_clang=0
fi
have_lld=1
if ! command -v ld.lld >/dev/null 2>&1 && ! command -v lld >/dev/null 2>&1; then
    have_lld=0
fi
if [ "$have_clang" = 1 ] && [ "$have_lld" = 1 ]; then
    echo "clang and lld are already on this PC."
    [ -n "$recomp" ] && "$recomp" doctor
    exit 0
fi

if command -v apt-get >/dev/null 2>&1; then
    cmd="sudo apt-get install -y clang lld"
elif command -v dnf >/dev/null 2>&1; then
    cmd="sudo dnf install -y clang lld"
elif command -v pacman >/dev/null 2>&1; then
    cmd="sudo pacman -S --needed clang lld"
elif command -v zypper >/dev/null 2>&1; then
    cmd="sudo zypper install -y clang lld"
else
    echo "This distribution's package manager is not one this script knows."
    echo "Install clang and lld with it by hand (a recent clang; 21 and 22 are the ones tried), then run doctor.sh."
    exit 1
fi

echo "recomp needs clang and lld, and this PC is missing one of them. This would run:"
echo "    $cmd"
if [ "${1:-}" = "--dry" ]; then
    exit 0
fi
printf "Run it now? [y/N] "
read -r answer
case "$answer" in
    y | Y | yes | YES) ;;
    *)
        echo "nothing was installed."
        exit 1
        ;;
esac
$cmd || {
    echo
    echo "the install failed: do it by hand with your package manager (clang and lld)."
    exit 1
}
echo
echo "done. Run doctor.sh: it should name a clang now."
