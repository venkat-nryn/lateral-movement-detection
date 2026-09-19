"""Deterministic synthetic temporal security scenarios (Module M2.2).

The scenarios in this module are deterministic software fixtures used
to validate graph functionality. They are not derived from LANL, do
not represent real attacks, and must not be used as evidence of
detection performance.

Every generator returns freshly constructed, validated
:class:`~ml.preprocessing.schema.CanonicalEvent` objects with fixed
identifiers, timestamps, and ordering. No random number generation is
used anywhere in this module, so repeated calls produce identical
output. These fixtures exist solely for software testing,
graph-engine testing, demonstrations, and future unit tests.
"""

from __future__ import annotations

from typing import List

from ml.preprocessing.schema import (
    EVENT_TYPE_AUTHENTICATION,
    CanonicalEvent,
)

_BASE_TIME = 1_000_000.0


def _event(
    suffix: str,
    offset: float,
    user: str,
    source_host: str,
    destination_host: str,
    success: bool = True,
    base: float = _BASE_TIME,
) -> CanonicalEvent:
    return CanonicalEvent(
        event_id=f"{suffix}-{offset:09.1f}",
        timestamp=base + offset,
        user=user,
        source_host=source_host,
        destination_host=destination_host,
        event_type=EVENT_TYPE_AUTHENTICATION,
        success=success,
    )


def create_benign_scenario() -> List[CanonicalEvent]:
    """Three independent users logging on to their own workstations.

    All events succeed; timestamps differ per user.
    """
    return [
        _event("benign", 10.0, "User_A", "Workstation_A", "Workstation_A"),
        _event("benign", 20.0, "User_B", "Workstation_B", "Workstation_B"),
        _event("benign", 30.0, "User_C", "Workstation_C", "Workstation_C"),
    ]


def create_multihop_scenario() -> List[CanonicalEvent]:
    """One user authenticating into a sequence of four hosts.

    This is a software test scenario representing a multi-step
    movement pattern; it is not labelled as or derived from any real
    attack. Each event's ``source_host`` is the previous destination,
    so the movement chain is preserved in the event context.
    Timestamps are strictly increasing.
    """
    chain = [
        ("Workstation_A", "Workstation_B"),
        ("Workstation_B", "Server_A"),
        ("Server_A", "Critical_Server"),
    ]
    events = [
        _event(
            "multihop", 100.0, "User_A", "Workstation_A", "Workstation_A"
        )
    ]
    for index, (source, destination) in enumerate(chain, start=1):
        events.append(
            _event(
                "multihop",
                100.0 + 60.0 * index,
                "User_A",
                source,
                destination,
            )
        )
    return events


def create_mixed_scenario() -> List[CanonicalEvent]:
    """Normal activity, repetition, a multi-step sequence, and failures.

    Deterministically mixes benign single logons, repeated
    authentication between the same entities at different times, one
    short multi-step sequence, and both successful and failed events
    across multiple users and hosts.
    """
    events: List[CanonicalEvent] = []

    # Normal independent activity.
    events.append(_event("mix", 10.0, "User_A", "Workstation_A", "Workstation_A"))
    events.append(_event("mix", 20.0, "User_B", "Workstation_B", "Workstation_B"))
    events.append(_event("mix", 30.0, "User_C", "Workstation_C", "Workstation_C"))

    # Repeated authentication: same entities, different times.
    events.append(_event("mix", 40.0, "User_A", "Workstation_A", "Server_A"))
    events.append(_event("mix", 80.0, "User_A", "Workstation_A", "Server_A"))

    # A failed attempt followed by a successful retry.
    events.append(
        _event(
            "mix", 120.0, "User_D", "Workstation_D", "Server_B", success=False
        )
    )
    events.append(_event("mix", 150.0, "User_D", "Workstation_D", "Server_B"))

    # Short multi-step sequence by User_A.
    events.append(_event("mix", 200.0, "User_A", "Server_A", "Critical_Server"))
    events.append(
        _event("mix", 260.0, "User_A", "Critical_Server", "Backup_Server")
    )

    return events


GENERATORS = {
    "benign": create_benign_scenario,
    "multihop": create_multihop_scenario,
    "mixed": create_mixed_scenario,
}
