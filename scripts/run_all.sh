#!/usr/bin/env bash
set -euo pipefail
PY=.venv/bin/python
$PY scripts/build_manifest.py
# Screening then final selected protocol; skips completed results.
for a in A0 A1 A2; do $PY scripts/train.py --model complex --aug "$a" --loss weighted --seed 42 --epochs 40; done
for l in ce weighted; do $PY scripts/train.py --model complex --aug A1 --loss "$l" --seed 42 --epochs 40; done
for m in simple complex mobile; do for s in 42 43 44; do $PY scripts/train.py --model "$m" --aug A1 --loss weighted --seed "$s" --epochs 40; done; done
