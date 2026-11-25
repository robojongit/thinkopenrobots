#!/usr/bin/env bash
# Build the agent image and print its tag, inkling-agent:<version>. The version is a hash of
# docker/agent/ and sdk/inkling_robo/ (harness.sandbox.agent_image), so the harness always runs
# the image that matches its SDK; after changing either, build again.
#
# The context is only those two folders (no .venv, simulator or runs), sent as a tar stream.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
tag=$(python3 -c 'from harness.sandbox import agent_image; print(agent_image())')
tar -c --exclude=__pycache__ docker/agent sdk/inkling_robo \
    | docker build --quiet -t "$tag" -f docker/agent/Dockerfile - >&2
echo "$tag"
