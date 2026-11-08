"""slotwise: offline, approval-gated send-slot optimizer for social-media posts.

The package chooses *when* to publish a fixed set of posts across channels over a
planning horizon so as to maximize a transparent, simulated-engagement objective,
subject to per-day and per-channel caps and an audience-fatigue penalty for posting
too frequently on the same channel.

Everything runs offline and deterministically: synthetic scenarios are seeded,
there is no network access at runtime, and the "publish" step is a dry-run outbox
that refuses to act without an explicit human approval.
"""

from __future__ import annotations

from slotwise.models import (
    Approval,
    ApprovalRequired,
    Channel,
    FatigueParams,
    Placement,
    PublishResult,
    Scenario,
    Schedule,
)
from slotwise.optimize import optimize_schedule
from slotwise.publish import publish

__version__ = "0.1.0"

__all__ = [
    "Approval",
    "ApprovalRequired",
    "Channel",
    "FatigueParams",
    "Placement",
    "PublishResult",
    "Scenario",
    "Schedule",
    "optimize_schedule",
    "publish",
    "__version__",
]
