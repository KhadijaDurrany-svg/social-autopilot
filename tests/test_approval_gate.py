"""Safety tests for the approval-gated publish step.

The core contract: ``publish`` must never produce an outbox unless an explicit, confirmed
human approval is supplied. These tests fail loudly if that gate is ever weakened.
"""

from __future__ import annotations

import pytest

from slotwise.generate import generate_scenario
from slotwise.models import Approval, ApprovalRequired
from slotwise.optimize import optimize_schedule
from slotwise.publish import publish


def _fixture():
    scenario = generate_scenario(7)
    return scenario, optimize_schedule(scenario)


def test_publish_without_approval_raises():
    scenario, sched = _fixture()
    with pytest.raises(ApprovalRequired):
        publish(scenario, sched)


def test_publish_with_none_approval_raises():
    scenario, sched = _fixture()
    with pytest.raises(ApprovalRequired):
        publish(scenario, sched, approval=None)


def test_publish_with_unconfirmed_approval_raises():
    scenario, sched = _fixture()
    appr = Approval(approved_by="khadija", confirmed=False)
    with pytest.raises(ApprovalRequired):
        publish(scenario, sched, approval=appr)


def test_publish_with_blank_approver_raises():
    scenario, sched = _fixture()
    appr = Approval(approved_by="   ", confirmed=True)
    with pytest.raises(ApprovalRequired):
        publish(scenario, sched, approval=appr)


def test_publish_rejects_non_approval_object():
    scenario, sched = _fixture()
    with pytest.raises(ApprovalRequired):
        publish(scenario, sched, approval="yes please")  # type: ignore[arg-type]


def test_publish_with_valid_approval_is_dry_run():
    scenario, sched = _fixture()
    appr = Approval(approved_by="khadija", confirmed=True)
    result = publish(scenario, sched, approval=appr)
    assert result.dry_run is True
    assert result.approved_by == "khadija"
    assert len(result.placements) == len(sched.placements)
    assert len(result.outbox) == len(sched.placements)
    assert all(line.startswith("[DRY-RUN]") for line in result.outbox)


def test_publish_validates_schedule_feasibility():
    from slotwise.models import Placement, Schedule

    scenario, _ = _fixture()
    appr = Approval(approved_by="khadija", confirmed=True)
    bad = Schedule(placements=(Placement(channel="x", slot=99999),))
    with pytest.raises(ValueError):
        publish(scenario, bad, approval=appr)
