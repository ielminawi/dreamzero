#!/bin/bash
# Stop the DreamZero policy server.
set -uo pipefail
source "$(dirname "$0")/scripts/_env.sh"
dz_stop_server
