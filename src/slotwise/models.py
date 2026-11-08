"""Typed data model for slotwise.

The model is intentionally small and explicit. A :class:`Scenario` fully describes a
scheduling problem: a planning horizon (``days`` x ``slots_per_day`` time slots), a set
of :class:`Channel` objects each with a per-slot expected-activity curve, how many posts
must be placed on each channel, per-day caps, and the fatigue parameters shared by all
channels. A :class:`Schedule` is an assignment of posts to time slots.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FatigueParams(BaseModel):
    """Audience-fatigue model.

    The multiplier applied to a placement's engagement rises linearly from ``floor``
    (when the previous same-channel post was one slot ago) up to ``1.0`` once the gap
    reaches ``recover_slots``. The first post on a channel is never penalized.
    """

    model_config = ConfigDict(frozen=True)

    recover_slots: int = Field(ge=1, description="Gap (in slots) at which engagement fully recovers.")
    floor: float = Field(gt=0.0, lt=1.0, description="Multiplier for a back-to-back post (gap == 1).")

    def multiplier(self, gap: int) -> float:
        """Return the fatigue multiplier for a post placed ``gap`` slots after the previous one.

        ``gap`` must be >= 1. A gap >= ``recover_slots`` returns ``1.0``; use a large gap
        (or the sentinel used by the optimizer) to represent "no previous post".
        """
        if gap < 1:
            raise ValueError("gap must be >= 1")
        if gap >= self.recover_slots:
            return 1.0
        return self.floor + (1.0 - self.floor) * (gap / self.recover_slots)


class Channel(BaseModel):
    """A single publishing channel with its expected per-slot activity curve."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    activity: tuple[float, ...] = Field(description="Expected engagement weight per time slot, length == horizon.")
    posts: int = Field(ge=0, description="Number of posts that must be scheduled on this channel.")
    daily_cap: int = Field(ge=1, description="Maximum posts allowed on this channel per day.")

    @model_validator(mode="after")
    def _check_activity(self) -> Channel:
        if len(self.activity) == 0:
            raise ValueError("activity curve must be non-empty")
        if any(a < 0.0 for a in self.activity):
            raise ValueError("activity values must be non-negative")
        return self


class Scenario(BaseModel):
    """A complete, self-contained scheduling problem."""

    model_config = ConfigDict(frozen=True)

    slots_per_day: int = Field(ge=1)
    days: int = Field(ge=1)
    channels: tuple[Channel, ...] = Field(min_length=1)
    fatigue: FatigueParams

    @property
    def horizon(self) -> int:
        """Total number of time slots in the planning window."""
        return self.slots_per_day * self.days

    def day_of(self, slot: int) -> int:
        """Return the day index (0-based) containing ``slot``."""
        return slot // self.slots_per_day

    def capacity(self, channel: Channel) -> int:
        """Maximum number of posts that can be placed on ``channel`` given its daily cap."""
        return self.days * channel.daily_cap

    @model_validator(mode="after")
    def _check(self) -> Scenario:
        names = [c.name for c in self.channels]
        if len(set(names)) != len(names):
            raise ValueError("channel names must be unique")
        for c in self.channels:
            if len(c.activity) != self.horizon:
                raise ValueError(
                    f"channel {c.name!r} activity length {len(c.activity)} != horizon {self.horizon}"
                )
        return self


class Placement(BaseModel):
    """A single post placed on a channel at a time slot."""

    model_config = ConfigDict(frozen=True)

    channel: str
    slot: int = Field(ge=0)


class Schedule(BaseModel):
    """An assignment of posts to time slots, plus the name of the strategy that built it."""

    model_config = ConfigDict(frozen=True)

    placements: tuple[Placement, ...] = ()
    strategy: str = "unknown"

    def slots_for(self, channel: str) -> list[int]:
        """Return the sorted list of slots used on ``channel``."""
        return sorted(p.slot for p in self.placements if p.channel == channel)

    def count_for(self, channel: str) -> int:
        return sum(1 for p in self.placements if p.channel == channel)

    def validate_against(self, scenario: Scenario) -> None:
        """Raise ``ValueError`` if this schedule violates the scenario's hard constraints.

        Checks the horizon bound, no double-booking a channel in the same slot, and the
        per-day cap. Does not require that every post be placed.
        """
        by_channel: dict[str, list[int]] = {c.name: [] for c in scenario.channels}
        for p in self.placements:
            if p.channel not in by_channel:
                raise ValueError(f"unknown channel {p.channel!r}")
            if not (0 <= p.slot < scenario.horizon):
                raise ValueError(f"slot {p.slot} out of range for horizon {scenario.horizon}")
            by_channel[p.channel].append(p.slot)
        for c in scenario.channels:
            slots = by_channel[c.name]
            if len(set(slots)) != len(slots):
                raise ValueError(f"channel {c.name!r} double-books a slot")
            per_day: dict[int, int] = {}
            for s in slots:
                d = scenario.day_of(s)
                per_day[d] = per_day.get(d, 0) + 1
                if per_day[d] > c.daily_cap:
                    raise ValueError(f"channel {c.name!r} exceeds daily cap on day {d}")


class Approval(BaseModel):
    """An explicit human approval required before :func:`slotwise.publish.publish` will act.

    ``confirmed`` must be set to ``True`` *explicitly*; a default-constructed Approval is
    not valid for publishing. This is the human-in-the-loop gate.
    """

    model_config = ConfigDict(frozen=True)

    approved_by: str = Field(min_length=1, description="Identity of the human who approved the send.")
    confirmed: bool = False
    note: str = ""


class ApprovalRequired(RuntimeError):
    """Raised when a publish is attempted without a valid, confirmed approval."""


class PublishResult(BaseModel):
    """The outcome of a (dry-run, offline) publish."""

    model_config = ConfigDict(frozen=True)

    dry_run: bool = True
    approved_by: str
    placements: tuple[Placement, ...]
    outbox: tuple[str, ...]
