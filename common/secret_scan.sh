#!/usr/bin/env bash
# Pre-commit secret/size scan of STAGED changes. Exit 1 on any hit. See secret_scan.py.
exec /opt/pytorch/bin/python "$(git rev-parse --show-toplevel)/common/secret_scan.py" "$@"
