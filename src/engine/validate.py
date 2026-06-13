#!/usr/bin/env python3
"""
Validation entrypoint placeholder for MVAA Task 1.

Current training engine already performs internal validation.
This module exists as a clean future extension point for standalone validation.
"""

from __future__ import annotations


def validate_task1() -> None:
    raise NotImplementedError(
        "Standalone validation will be implemented after inference/submission pipeline is stable. "
        "For now, use src/engine/train.py internal validation."
    )
