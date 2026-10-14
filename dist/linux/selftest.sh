#!/bin/sh
# Builds the small program in the selftest folder and runs it. It needs what a game needs (clang,
# lld and the runtime library) and no game. The program computes the tenth Fibonacci number and
# exits with it, 55.
here="$(cd "$(dirname "$0")" && pwd)"
out="${TMPDIR:-/tmp}/kiln-selftest-$$"
rm -rf "$out"

failed() {
    echo
    echo "selftest FAILED. Send the text above to whoever gave you this bundle."
    rm -rf "$out"
    exit 1
}

"$here/kiln" doctor
echo
echo "building the self-test program..."
"$here/kiln" build --exefs "$here/selftest/exefs" --out "$out" --name selftest || failed
"$out/selftest"
code=$?
if [ "$code" != 55 ]; then
    echo
    echo "the self-test program exited with $code, not 55."
    failed
fi
echo
echo "selftest passed: the compiler, the runtime library and the linking work on this PC."
rm -rf "$out"
