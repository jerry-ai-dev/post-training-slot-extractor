import json
from collections.abc import Iterable, Iterator

from .models import CompareEvent


def encode_event(event: CompareEvent) -> str:
    return json.dumps(
        {
            "side": event.side,
            "comparable": event.comparable,
            "seq": event.event.seq,
            "type": event.event.kind,
            "payload": event.event.payload,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def encode_events(events: Iterable[CompareEvent]) -> Iterator[str]:
    for event in events:
        yield encode_event(event) + "\n"


def encode_side_status(side: str, status: str, comparable: bool, **payload: object) -> str:
    return json.dumps(
        {
            "side": side,
            "comparable": comparable,
            "seq": -1,
            "type": "side_status",
            "payload": {"status": status, **payload},
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ) + "\n"
