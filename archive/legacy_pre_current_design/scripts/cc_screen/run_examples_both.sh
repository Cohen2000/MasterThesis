#!/usr/bin/env bash
# Resume the paired Claude disclosed_examples/mask cells sequentially,
# completing the tools arm before continuing with notools.
#
# Each arm uses its own answer file and cumulative cost cap. Usage-limit
# refusals are handled inside the runner by waiting for the reported reset;
# this wrapper does not poll or repeatedly call while a limit is active.

set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN_ONE="$REPO/scripts/cc_screen/run_examples_cell.sh"
STATUS=0

trap 'exit 130' INT TERM

for arm in tools notools; do
    echo
    echo "============================================================"
    echo "Claude examples arm: $arm"
    echo "============================================================"

    if [[ "$arm" == "notools" ]]; then
        ARM_CAP="${CC_EXAMPLES_NOTOOLS_COST_CAP:-${CC_EXAMPLES_COST_CAP:-50}}"
    else
        ARM_CAP="${CC_EXAMPLES_TOOLS_COST_CAP:-${CC_EXAMPLES_COST_CAP:-70}}"
    fi

    CC_EXAMPLES_COST_CAP="$ARM_CAP" bash "$RUN_ONE" "$arm"
    RC=$?
    if ((RC >= 128)); then
        exit "$RC"
    fi
    if ((RC != 0)); then
        STATUS="$RC"
        echo "arm=$arm ended with status $RC; continuing with the other arm" >&2
    fi
done

exit "$STATUS"
