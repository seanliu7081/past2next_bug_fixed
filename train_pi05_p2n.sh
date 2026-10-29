#!/usr/bin/env bash
# pi05_p2n launcher: Pi05_past with the Past2Next self-past AR head over 8 OAT tokens instead of the flow head;
# everything else (VLM regime, condition, self-past schedule, data, batch, schedule, evaluation) is Pi05_past's.
# Same flags as train_pi05_past.sh (it runs it with --variant pi05_p2n); defaults in oat/config/train_pi05_p2n.yaml.
# Design and evaluation: docs/PI05_P2N.md.
#
#   bash train_pi05_p2n.sh --help
#   bash train_pi05_p2n.sh --dry-run                             # resolve + validate the config only
#   bash train_pi05_p2n.sh --devices 0,1,2,3 --probe             # memory/throughput go/no-go, no checkpoints
#   bash train_pi05_p2n.sh --devices 0,1,2,3                     # pi05_p2n, 80k updates of batch 32
#   bash train_pi05_p2n.sh --devices 0,1,2,3 --use-past false    # AR no-past control, same recipe
set -euo pipefail
exec bash "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/train_pi05_past.sh" --variant pi05_p2n "$@"
