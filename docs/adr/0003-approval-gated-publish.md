# ADR 0003: Approval-gated, dry-run publish

## Status

Accepted.

## Context

A scheduler that can publish on a person's behalf is a safety-sensitive component: an
automated or accidental send is hard to undo and can damage a brand. Even though slotwise
is offline and has no platform integrations today, we want the "act" boundary designed
correctly from the start so that adding a real adapter later cannot silently bypass human
oversight.

## Decision

All acting flows through a single function, `publish()`, which:

* requires an explicit `Approval` object with `confirmed=True` and a named human approver;
* raises `ApprovalRequired` for `None`, an unconfirmed approval, a blank approver, or any
  non-`Approval` value -- there is no "default approve" path;
* even when approved, only returns a clearly labelled **dry-run outbox** and performs no
  network I/O.

`Approval.confirmed` defaults to `False`, so a default-constructed approval cannot
publish. The CLI mirrors this: `slotwise publish` exits non-zero and prints `REFUSED`
unless `--confirm` and `--approved-by` are both given.

## Consequences

* The human-in-the-loop gate is enforced in one place and covered by dedicated safety
  tests (`tests/test_approval_gate.py`) that fail loudly if the gate is weakened.
* A future real platform adapter must be called from inside `publish()` after the gate,
  inheriting the approval requirement by construction.
* The tool is safe to run in automation and CI: the worst case without approval is an
  exception, never a send.
