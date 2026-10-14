#!/bin/sh
# Shows what this PC has for a build: the version of kiln, the C compiler it will run and the
# runtime library that programs link. Nothing is built.
exec "$(dirname "$0")/kiln" doctor
