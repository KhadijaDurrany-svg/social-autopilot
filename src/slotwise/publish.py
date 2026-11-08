"""The approval-gated publish step.

slotwise never talks to any social platform. :func:`publish` turns a schedule into a
dry-run "outbox" -- a list of human-readable send instructions -- but only after an
explicit, confirmed human approval. Without one it raises :class:`ApprovalRequired`.

This is the single place where the tool would ever "act", and it is deliberately inert:
the safety contract is that an unapproved or unconfirmed call *always raises* and never
produces an outbox, so no accidental or automated send can slip through.
"""

from __future__ import annotations

from slotwise.models import (
    Approval,
    ApprovalRequired,
    PublishResult,
    Scenario,
    Schedule,
)


def _render_outbox(scenario: Scenario, schedule: Schedule) -> tuple[str, ...]:
    lines: list[str] = []
    for p in sorted(schedule.placements, key=lambda x: (x.channel, x.slot)):
        day = scenario.day_of(p.slot)
        hour = p.slot % scenario.slots_per_day
        lines.append(f"[DRY-RUN] {p.channel}: publish on day {day} at slot {hour} (global slot {p.slot})")
    return tuple(lines)


def publish(
    scenario: Scenario,
    schedule: Schedule,
    approval: Approval | None = None,
) -> PublishResult:
    """Produce a dry-run outbox for ``schedule``; raise unless explicitly approved.

    Parameters
    ----------
    approval:
        Must be a :class:`Approval` with ``confirmed=True`` and a non-empty
        ``approved_by``. ``None``, an unconfirmed approval, or a blank approver all raise
        :class:`ApprovalRequired`.

    Notes
    -----
    This function performs no network I/O under any circumstances. Even when approved it
    only returns a labelled dry-run plan.
    """
    if approval is None:
        raise ApprovalRequired("publish requires an explicit human Approval; none was given")
    if not isinstance(approval, Approval):
        raise ApprovalRequired("approval must be a slotwise Approval instance")
    if not approval.confirmed:
        raise ApprovalRequired("approval.confirmed must be True to publish")
    if not approval.approved_by.strip():
        raise ApprovalRequired("approval.approved_by must identify a human approver")

    schedule.validate_against(scenario)
    outbox = _render_outbox(scenario, schedule)
    return PublishResult(
        dry_run=True,
        approved_by=approval.approved_by,
        placements=schedule.placements,
        outbox=outbox,
    )


__all__ = ["publish"]
